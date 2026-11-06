"""POM nodes for ComfyUI: use the models the POM cluster serves.

Installed by the POM ComfyUI plugin into `custom_nodes/pom_nodes`, and
rewritten on every start. The node's gateway address and key arrive only in
this process's environment (`POM_OPENAI_BASE_URL`, `POM_API_KEY`); they never
reach the browser, a workflow file or a log.

ComfyUI here runs behind POM, on the node, not on the viewer's computer: its
"Download" button for a missing model would save the file in the browser.
`web/pom.js` hands that button to `/pom/models/download`, which downloads the
model into this server's models directory instead.
"""
import base64
import io
import json
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

CATEGORY = "POM"
UNAVAILABLE = "(POM gateway unavailable)"
NO_MODELS = "(no model deployed)"
THINK = re.compile(r"<think>.*?</think>\s*", re.S)

ENHANCE_SYSTEM = (
    "You write prompts for text-to-image diffusion models. Rewrite the user's idea as one "
    "detailed prompt: subject, composition, lighting, style, lens and mood, as comma "
    "separated phrases. Answer with the prompt only, no preamble and no quotes."
)


def _gateway():
    base = os.environ.get("POM_OPENAI_BASE_URL", "").rstrip("/")
    key = os.environ.get("POM_API_KEY", "")
    if not base or not key:
        raise RuntimeError("POM gateway is not configured for ComfyUI; reopen it from POM")
    return base, key


def _call(path, body=None, timeout=600):
    base, key = _gateway()
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, method="GET" if body is None else "POST")
    request.add_header("Authorization", f"Bearer {key}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read()[:500].decode(errors="replace")
        raise RuntimeError(f"POM answered HTTP {error.code}: {detail}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"POM gateway unreachable: {error.reason}") from None


def pom_models():
    """The ids `/v1/models` lists now; asked again whenever ComfyUI reloads
    its node definitions (opening or refreshing the page)."""
    try:
        listing = _call("/models", timeout=5)
    except Exception:
        return [UNAVAILABLE]
    ids = sorted({entry.get("id") for entry in listing.get("data", []) if entry.get("id")})
    return ids or [NO_MODELS]


def chat(model, messages, temperature, max_tokens, seed):
    if model in (UNAVAILABLE, NO_MODELS):
        raise RuntimeError("Deploy a model in POM first, then refresh ComfyUI")
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": False}
    if seed:
        body["seed"] = seed
    reply = _call("/chat/completions", body)
    try:
        text = reply["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise RuntimeError("POM returned no answer") from None
    return THINK.sub("", text).strip()


def _model_input():
    return (pom_models(), {"tooltip": "A model deployed on the POM cluster"})


def _sampling():
    return {
        "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
        "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 131072}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                         "tooltip": "Change it to ask again; 0 lets the model choose"}),
    }


class POMChat:
    """Ask a POM model anything; the answer is a STRING for any text input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": _model_input(),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "system": ("STRING", {"multiline": True, "default": ""}),
            **_sampling(),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, prompt, system, temperature, max_tokens, seed):
        messages = ([{"role": "system", "content": system}] if system.strip() else [])
        messages.append({"role": "user", "content": prompt})
        return (chat(model, messages, temperature, max_tokens, seed),)


class POMPromptEnhancer:
    """Turn a short idea into a detailed text-to-image prompt with a POM model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": _model_input(),
            "idea": ("STRING", {"multiline": True, "default": "a lighthouse in a storm"}),
            **_sampling(),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, idea, temperature, max_tokens, seed):
        messages = [{"role": "system", "content": ENHANCE_SYSTEM}, {"role": "user", "content": idea}]
        return (chat(model, messages, temperature, max_tokens, seed),)


def _png_data_url(image):
    """ComfyUI's IMAGE is a [batch, height, width, channels] float tensor."""
    import numpy as np
    from PIL import Image

    array = (image[0].cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class POMDescribeImage:
    """Ask a multimodal POM model about an image (captions, tags, critique)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": _model_input(),
            "image": ("IMAGE",),
            "question": ("STRING", {"multiline": True,
                                    "default": "Describe this image in one detailed paragraph."}),
            **_sampling(),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, image, question, temperature, max_tokens, seed):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": _png_data_url(image)}},
        ]}]
        return (chat(model, messages, temperature, max_tokens, seed),)


NODE_CLASS_MAPPINGS = {
    "POMChat": POMChat,
    "POMPromptEnhancer": POMPromptEnhancer,
    "POMDescribeImage": POMDescribeImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "POMChat": "POM Chat (LLM)",
    "POMPromptEnhancer": "POM Prompt Enhancer",
    "POMDescribeImage": "POM Describe Image",
}


# --- Missing models: download on the server ---------------------------------

WEB_DIRECTORY = "./web"
# The hosts and file types ComfyUI's own missing-model dialog offers to download.
MODEL_HOSTS = {"huggingface.co", "civitai.com", "civitai.red"}
MODEL_SUFFIXES = (".safetensors", ".sft", ".ckpt", ".pth", ".pt", ".gguf")
_downloads = {}
_downloads_lock = threading.Lock()


HEADROOM = 512 * 2**20


def _validate(url, name, directory):
    """Refuse what the dialog must not download, with a message it can show."""
    import folder_paths

    parsed = urllib.parse.urlparse(url or "")
    if parsed.scheme != "https" or (parsed.hostname or "").lower() not in MODEL_HOSTS:
        raise ValueError("Only Hugging Face and Civitai models can be downloaded here")
    if not name or name != os.path.basename(name) or name.startswith(".") or "\\" in name:
        raise ValueError("Invalid model file name")
    if not name.lower().endswith(MODEL_SUFFIXES):
        raise ValueError("Not a model file")
    if directory not in folder_paths.folder_names_and_paths:
        raise ValueError(f"Unknown model folder: {directory}")


def relative_destination(url, name):
    """Where a download goes inside a POM model folder, in the layout the
    folders already use: `<org>/<repo>/<file>` from a Hugging Face URL,
    `civitai/<version id>/<file>` from a Civitai one."""
    parsed = urllib.parse.urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if (parsed.hostname or "").lower().startswith("civitai."):
        version = parts[-1] if parts else "models"
        return "/".join(["civitai", version, name])
    if len(parts) >= 2:
        return "/".join([parts[0], parts[1], name])
    return name


def choose_root(dirs, size, free=shutil.disk_usage):
    """The first folder with room for the file and some headroom."""
    for root in dirs:
        try:
            if free(root).free >= size + HEADROOM:
                return root
        except OSError:
            continue
    return None


def _model_dirs():
    return [d for d in os.environ.get("POM_MODEL_DIRS", "").split("\n") if d]


def _comfy_target(name, directory):
    """Without POM's folders (an older node): ComfyUI's own models folder."""
    import folder_paths

    paths = folder_paths.get_folder_paths(directory)
    folder = next((path for path in paths if os.path.basename(path) == directory), paths[0])
    return os.path.join(folder, name)


def _existing(url, name, directory):
    """The file when it is already on this machine, else None."""
    dirs = _model_dirs()
    if not dirs:
        target = _comfy_target(name, directory)
        return target if os.path.exists(target) else None
    relative = relative_destination(url, name)
    for root in dirs:
        target = os.path.join(root, *relative.split("/"))
        if os.path.exists(target):
            return target
    return None


def _host_api_post(path):
    base = os.environ.get("POM_HOST_API_URL", "")
    token = os.environ.get("POM_HOST_API_TOKEN", "")
    if not base or not token:
        return
    request = urllib.request.Request(base + path, data=b"{}", method="POST")
    request.add_header("x-pom-plugin-host-token", token)
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


def sync_links():
    """Expose POM's catalog in ComfyUI's folders (filled in by the link sync)."""


def _after_download():
    try:
        _host_api_post("/catalog/refresh")
    except Exception as error:
        print(f"[POM] catalog refresh failed: {error}")
    try:
        sync_links()
    except Exception as error:
        print(f"[POM] model link sync failed: {error}")


def _publish(state):
    from server import PromptServer

    PromptServer.instance.send_sync("pom-download", _public(state))


def _public(state):
    return {key: value for key, value in state.items() if key != "url"}


def new_state(url, name, directory):
    return {"url": url, "name": name, "directory": directory, "path": "", "total": 0,
            "received": 0, "status": "downloading", "error": ""}


def _fetch(state):
    partial = None
    try:
        request = urllib.request.Request(state["url"], headers={"User-Agent": "POM-ComfyUI"})
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            state["total"] = total
            dirs = _model_dirs()
            if dirs:
                root = choose_root(dirs, total)
                if root is None:
                    raise RuntimeError(
                        f"Not enough disk space in any model folder for {total / 2**30:.1f} GiB")
                relative = relative_destination(state["url"], state["name"])
                target = os.path.join(root, *relative.split("/"))
            else:
                target = _comfy_target(state["name"], state["directory"])
                os.makedirs(os.path.dirname(target), exist_ok=True)
                free = shutil.disk_usage(os.path.dirname(target)).free
                if total and total > free:
                    raise RuntimeError(
                        f"Not enough disk space: {total / 2**30:.1f} GiB needed, "
                        f"{free / 2**30:.1f} GiB free")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            state["path"] = target
            partial = target + ".part"
            published = 0.0
            with open(partial, "wb") as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    state["received"] += len(chunk)
                    if time.monotonic() - published >= 1:
                        published = time.monotonic()
                        _publish(state)
        if total and state["received"] != total:
            raise RuntimeError("The download was cut short")
        os.replace(partial, target)
        state["status"] = "done"
        _after_download()
    except Exception as error:
        state["status"] = "error"
        state["error"] = str(error)[:300]
        if partial:
            try:
                os.remove(partial)
            except OSError:
                pass
    _publish(state)


def start_download(url, name, directory):
    _validate(url, name, directory)
    key = (url, name)
    with _downloads_lock:
        current = _downloads.get(key)
        if current and current["status"] == "downloading":
            return _public(current)
        state = new_state(url, name, directory)
        existing = _existing(url, name, directory)
        if existing:
            size = os.path.getsize(existing)
            state.update(status="done", path=existing, total=size, received=size)
            _downloads[key] = state
            return _public(state)
        _downloads[key] = state
    threading.Thread(target=_fetch, args=(state,), daemon=True).start()
    return _public(state)


def _register_routes():
    from aiohttp import web
    from server import PromptServer

    routes = PromptServer.instance.routes

    @routes.post("/pom/models/download")
    async def download(request):
        try:
            body = await request.json()
            state = start_download(body.get("url"), body.get("name"), body.get("directory"))
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        except Exception:
            return web.json_response({"error": "Invalid request"}, status=400)
        return web.json_response(dict(state))

    @routes.get("/pom/models/downloads")
    async def downloads(request):
        with _downloads_lock:
            return web.json_response([_public(state) for state in _downloads.values()])


def _prepare_user_directory():
    """ComfyUI answers 404 (a console error in the browser) while these
    directories do not exist yet."""
    import folder_paths

    user = os.path.join(folder_paths.get_user_directory(), "default")
    for directory in ("workflows", "subgraphs"):
        os.makedirs(os.path.join(user, directory), exist_ok=True)


try:
    _register_routes()
    _prepare_user_directory()
except Exception as error:  # imported outside a running ComfyUI server
    print(f"[POM] model downloads unavailable: {error}")
