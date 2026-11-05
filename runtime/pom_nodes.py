"""POM nodes for ComfyUI: use the models the POM cluster serves.

Installed by the POM ComfyUI plugin into `custom_nodes/pom_nodes`, and
rewritten on every start. The node's gateway address and key arrive only in
this process's environment (`POM_OPENAI_BASE_URL`, `POM_API_KEY`); they never
reach the browser, a workflow file or a log.
"""
import base64
import io
import json
import os
import re
import urllib.error
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
