# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 The POM authors. Runs inside ComfyUI; see runtime/LICENSE.
"""Workflows POM's Chat runs on this machine (installed as
`custom_nodes/pom_nodes/workflows.py`).

- Built-in: `builtin:<deploy_id>`, a text to image graph wired to an image
  deployment's network components (`POM ▸ <name> ▸ <file>` in ComfyUI's
  loaders), with the defaults of the model's family.
- Published: `pub:<slug>`, a graph an admin published from ComfyUI ("Publish
  to Chat"), stored in `<home>/user/pom-workflows/<slug>.json` with the node
  that takes the prompt.

`POST /pom/workflow/run` (behind the bridge's token, called by the POM node)
queues one in this ComfyUI, streams NDJSON progress (`{"p": n, "of": m}`) and
ends with the images (`{"images": [{mime, width, height, data}]}`) or
`{"error"}`. The images are base64 PNG bytes; the node keeps them.
"""
import base64
import json
import os
import random
import re
import threading
import time

PREFIX = "POM ▸ "
SEPARATOR = " ▸ "
WORKFLOWS_DIR = os.path.join("user", "pom-workflows")
MAX_COUNT = 16
RUN_TIMEOUT = 20 * 60

# steps, cfg, sampler, scheduler, size, CLIP type, latent node
FAMILIES = {
    "z_image": {"steps": 8, "cfg": 1.0, "sampler": "res_multistep", "scheduler": "simple",
                "width": 1024, "height": 1024, "clip_type": "lumina2", "latent": "EmptySD3LatentImage"},
    "qwen_image": {"steps": 20, "cfg": 2.5, "sampler": "euler", "scheduler": "simple",
                   "width": 1328, "height": 1328, "clip_type": "qwen_image", "latent": "EmptySD3LatentImage"},
    "flux": {"steps": 20, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
             "width": 1024, "height": 1024, "clip_type": "flux", "latent": "EmptySD3LatentImage"},
}
DEFAULT = {"steps": 20, "cfg": 5.0, "sampler": "euler", "scheduler": "normal",
           "width": 1024, "height": 1024, "clip_type": "stable_diffusion", "latent": "EmptyLatentImage"}
BUILTIN_INPUTS = ["prompt", "negative", "width", "height", "seed", "image"]
# How far image to image moves away from the base image (1 ignores it).
DEFAULT_STRENGTH = 0.6
MAX_BASE_IMAGE = 16 * 1024 * 1024


def family_of(names):
    """The model family, from the deployment's file names (best effort)."""
    text = " ".join(names).lower().replace("-", "_")
    for family in ("z_image", "qwen_image", "flux"):
        if family in text:
            return family
    return None


def defaults(family):
    return dict(FAMILIES.get(family, DEFAULT))


def name_of(entry):
    files = entry.get("files") or ["?"]
    return f"{PREFIX}{entry.get('name', '')}{SEPARATOR}{files[0]}"


def builtin(deploy_id, entries):
    """The text to image graph (ComfyUI's API format) for a deployment, from
    the network components it is made of. None when a stage is missing."""
    parts = {}
    for entry in entries:
        if entry.get("deploy_id") == deploy_id and entry.get("component") not in parts:
            parts[entry["component"]] = entry
    if not all(c in parts for c in ("text_encoder", "diffusion", "vae")):
        return None
    files = [f for e in parts.values() for f in (e.get("files") or [])]
    d = defaults(family_of(files))
    return {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": name_of(parts["text_encoder"]), "type": d["clip_type"]}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": name_of(parts["diffusion"]), "weight_dtype": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": name_of(parts["vae"])}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": ""}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 0], "text": ""}},
        "6": {"class_type": d["latent"], "inputs": {"width": d["width"], "height": d["height"], "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["2", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
            "seed": 0, "steps": d["steps"], "cfg": d["cfg"], "sampler_name": d["sampler"],
            "scheduler": d["scheduler"], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "pom-chat/pom"}},
    }


BUILTIN_MAP = {"prompt": "4", "negative": "5"}


def with_base_image(graph):
    """The built-in graph turned into image to image: the base image is
    loaded (node 10), scaled to about one megapixel in steps the latent
    accepts (11) and encoded by the deployment's VAE (12); the sampler starts
    from it instead of an empty latent."""
    graph = json.loads(json.dumps(graph))
    graph.pop("6", None)
    graph["10"] = {"class_type": "LoadImage", "inputs": {"image": ""}}
    graph["11"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {
        "image": ["10", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 16}}
    graph["12"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["11", 0], "vae": ["3", 0]}}
    graph["7"]["inputs"]["latent_image"] = ["12", 0]
    graph["7"]["inputs"]["denoise"] = DEFAULT_STRENGTH
    return graph


BUILTIN_IMAGE_MAP = {**BUILTIN_MAP, "image": "10"}


def base_image_bytes(data):
    """The bytes of a base image sent as base64 (a data URL or plain), and its
    extension; only PNG and JPEG, at most 16 MiB."""
    if not isinstance(data, str) or not data:
        raise ValueError("the base image is empty")
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(data, validate=True)
    except ValueError:
        raise ValueError("the base image is not valid base64") from None
    if len(raw) > MAX_BASE_IMAGE:
        raise ValueError("the base image is larger than 16 MiB")
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return raw, "png"
    if raw.startswith(b"\xff\xd8\xff"):
        return raw, "jpg"
    raise ValueError("the base image must be PNG or JPEG")


def save_base_image(data, input_dir):
    """Writes the base image under ComfyUI's input folder as
    `pom-chat/<sha256>.<ext>` and answers that name (what LoadImage takes)."""
    import hashlib

    raw, ext = base_image_bytes(data)
    name = f"pom-chat/{hashlib.sha256(raw).hexdigest()}.{ext}"
    path = os.path.join(input_dir, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path + ".part", "wb") as handle:
            handle.write(raw)
        os.replace(path + ".part", path)
    return name


def substitute(graph, inputs_map, values):
    """A copy of `graph` with the Chat's values set: the prompt (and negative)
    text, the size and batch on latent nodes, the seed on samplers (random
    when not given)."""
    graph = json.loads(json.dumps(graph))
    image_node = inputs_map.get("image")
    if image_node and image_node in graph and values.get("image_name"):
        graph[image_node]["inputs"]["image"] = values["image_name"]
    strength = values.get("strength")
    if strength is not None:
        for node in graph.values():
            if node.get("class_type") == "KSampler" and "denoise" in node.get("inputs", {}):
                node["inputs"]["denoise"] = max(0.05, min(float(strength), 1.0))
    for key in ("prompt", "negative"):
        node = inputs_map.get(key)
        if node and node in graph:
            graph[node]["inputs"]["text"] = values.get(key) or ""
    seed = values.get("seed")
    if seed is None:
        seed = random.randrange(0, 2**53)
    count = values.get("count")
    for node in graph.values():
        inputs = node.get("inputs", {})
        for field in ("seed", "noise_seed"):
            if field in inputs and not isinstance(inputs[field], list):
                inputs[field] = int(seed)
        if "width" in inputs and "height" in inputs and "batch_size" in inputs:
            if values.get("width"):
                inputs["width"] = int(values["width"])
            if values.get("height"):
                inputs["height"] = int(values["height"])
            if count:
                inputs["batch_size"] = max(1, min(int(count), MAX_COUNT))
    return graph


def slug(name):
    text = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return text[:48] or "workflow"


PARAM_TYPES = ("text", "number", "integer", "boolean", "choice", "image")
MAX_PARAMS = 24
MAX_TEXT = 8000
TRUE_WORDS = ("true", "1", "yes", "on")
FALSE_WORDS = ("false", "0", "no", "off", "")
# The keys the earlier publish dialog mapped to a node, and the input each sets.
LEGACY_KEYS = (("prompt", "text", "text", "Prompt"), ("negative", "text", "text", "Negative prompt"),
               ("image", "image", "image", "Base image"))


def _short(value, limit):
    return value.strip()[:limit] if isinstance(value, str) else ""


def _legacy_params(inputs):
    """The earlier `{prompt, negative?, image?}` choice as parameters."""
    out = []
    for key, input_name, kind, label in LEGACY_KEYS:
        node = (inputs or {}).get(key)
        if node:
            out.append({"key": key, "label": label, "node": str(node), "input": input_name, "type": kind})
    return out


def check_params(graph, params):
    """Validated copies of the parameters; raises ValueError."""
    if not isinstance(params, list) or len(params) > MAX_PARAMS:
        raise ValueError(f"a workflow takes at most {MAX_PARAMS} parameters")
    out, keys = [], set()
    for raw in params:
        if not isinstance(raw, dict):
            raise ValueError("a parameter must be an object")
        label = _short(raw.get("label"), 60) or _short(raw.get("key"), 60)
        key = slug(raw.get("key") or label).replace("-", "_")[:40]
        if not label or key in keys:
            raise ValueError(f"the parameter keys must be unique ({key})")
        node, input_name, kind = str(raw.get("node", "")), raw.get("input"), raw.get("type")
        inputs = (graph.get(node) or {}).get("inputs") or {}
        if not isinstance(input_name, str) or input_name not in inputs or isinstance(inputs[input_name], list):
            raise ValueError(f"{label}: the node has no editable input {input_name!r}")
        if kind not in PARAM_TYPES:
            raise ValueError(f"{label}: unknown type {kind!r}")
        if kind == "image" and input_name != "image":
            raise ValueError(f"{label}: an image parameter must be an image loader's input")
        param = {"key": key, "label": label, "node": node, "input": input_name, "type": kind}
        description = _short(raw.get("description"), 300)
        if description:
            param["description"] = description
        if kind == "choice":
            options = raw.get("options")
            if not isinstance(options, list) or not options or len(options) > 200:
                raise ValueError(f"{label}: a choice needs its options")
            param["options"] = [o for o in options if isinstance(o, (str, int, float))][:200]
        for bound in ("min", "max"):
            if kind in ("number", "integer") and isinstance(raw.get(bound), (int, float)) and not isinstance(raw.get(bound), bool):
                param[bound] = raw[bound]
        if "default" in raw and kind != "image":
            param["default"] = coerce(param, raw["default"])
        keys.add(key)
        out.append(param)
    return out


def coerce(param, value):
    """A value of a parameter's type; numbers are clamped to min/max, a choice
    must be one of its options. Raises ValueError."""
    kind, label = param["type"], param.get("label", param["key"])
    if kind == "text":
        return str(value)[:MAX_TEXT]
    if kind in ("number", "integer"):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a number") from None
        if number != number:
            raise ValueError(f"{label} must be a number")
        if "min" in param:
            number = max(number, param["min"])
        if "max" in param:
            number = min(number, param["max"])
        return int(round(number)) if kind == "integer" else float(number)
    if kind == "boolean":
        if isinstance(value, bool):
            return value
        word = str(value).strip().lower()
        if word in TRUE_WORDS:
            return True
        if word in FALSE_WORDS:
            return False
        raise ValueError(f"{label} must be true or false")
    if kind == "choice":
        for option in param.get("options", []):
            if option == value or str(option) == str(value):
                return option
        raise ValueError(f"{label} must be one of: {', '.join(map(str, param.get('options', [])[:20]))}")
    raise ValueError(f"{label}: an image is given as a PNG or JPEG data URL")


def params_of(record):
    """A published record's parameters; one published before parameters
    existed has its `map` read as parameters."""
    if isinstance(record.get("params"), list):
        return record["params"]
    return _legacy_params(record.get("map"))


def apply_params(graph, params, values, save_image):
    """A copy of `graph` with each given value set on its node input (coerced;
    an image goes through `save_image(data) -> name`). A seed no parameter
    sets is random, so a run never answers ComfyUI's cached result."""
    graph = json.loads(json.dumps(graph))
    seeded = set()
    for param in params:
        node = graph.get(param["node"])
        if node is None:
            continue
        if param["type"] in ("integer", "number") and param["input"] in ("seed", "noise_seed"):
            seeded.add((param["node"], param["input"]))
        value = (values or {}).get(param["key"])
        if value is None or (param["type"] == "image" and not value):
            continue
        if param["type"] == "image":
            value = save_image(value)
        else:
            value = coerce(param, value)
        node.setdefault("inputs", {})[param["input"]] = value
    for node_id, node in graph.items():
        inputs = node.get("inputs", {})
        for field in ("seed", "noise_seed"):
            if field in inputs and not isinstance(inputs[field], list) and (node_id, field) not in seeded:
                inputs[field] = random.randrange(0, 2**53)
    return graph


def publish(home, name, graph, params, description=""):
    """Stores a graph for the Chat with its parameters (a list, or the earlier
    `{prompt, negative?, image?}` node choice). Answers its id."""
    name = (name or "").strip()
    if not name or len(name) > 80:
        raise ValueError("a workflow needs a name of up to 80 characters")
    if not isinstance(graph, dict) or not graph:
        raise ValueError("the workflow is empty")
    if isinstance(params, dict):
        prompt = params.get("prompt")
        if prompt not in graph or "text" not in graph[prompt].get("inputs", {}):
            raise ValueError("choose the text node that receives the prompt")
        params = _legacy_params(params)
    params = check_params(graph, params)
    if not params:
        raise ValueError("choose at least one input the Chat fills")
    workflow_id = "pub:" + slug(name)
    folder = os.path.join(home, WORKFLOWS_DIR)
    os.makedirs(folder, exist_ok=True)
    keys = [p["key"] for p in params]
    mapped = {p["key"]: p["node"] for p in params
              if p["key"] in ("prompt", "negative", "image") and p["type"] in ("text", "image")}
    record = {"id": workflow_id, "name": name, "description": _short(description, 500),
              "inputs": keys, "map": mapped, "params": params, "graph": graph}
    path = os.path.join(folder, workflow_id[4:] + ".json")
    with open(path + ".part", "w") as handle:
        json.dump(record, handle)
    os.replace(path + ".part", path)
    return workflow_id


def published(home):
    folder = os.path.join(home, WORKFLOWS_DIR)
    found = []
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return found
    for file_name in names:
        if not file_name.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, file_name)) as handle:
                found.append(json.load(handle))
        except (OSError, ValueError):
            continue
    return found


def resolve(home, workflow_id, entries, base_image=False):
    """(graph, inputs map) of a workflow id, or raises ValueError. With a base
    image the built-in one becomes image to image, and a published one must
    have a node that takes it."""
    if workflow_id.startswith("builtin:"):
        graph = builtin(workflow_id[len("builtin:"):], entries)
        if graph is None:
            raise ValueError("this deployment is missing a stage on the network")
        if base_image:
            return with_base_image(graph), BUILTIN_IMAGE_MAP
        return graph, BUILTIN_MAP
    for record in published(home):
        if record.get("id") == workflow_id:
            mapped = record.get("map", {})
            if base_image and "image" not in mapped:
                raise ValueError("this workflow takes no base image")
            return record["graph"], mapped
    raise ValueError("no such workflow on this machine")


# --- Progress of a queued prompt ----------------------------------------------

_listeners = {}
_listeners_lock = threading.Lock()


def _on_event(event, data):
    if not isinstance(data, dict):
        return
    prompt_id = data.get("prompt_id")
    with _listeners_lock:
        listener = _listeners.get(prompt_id)
    if listener is not None:
        listener(event, data)


def _watch_progress():
    from server import PromptServer

    instance = PromptServer.instance
    original = instance.send_sync

    def send_sync(event, data, sid=None):
        try:
            _on_event(event, data)
        except Exception as error:  # never break ComfyUI's own events
            print(f"[POM] workflow progress failed: {error}")
        return original(event, data, sid)

    instance.send_sync = send_sync


MAX_MEDIA = 16
MAX_MEDIA_BYTES = 32 * 1024 * 1024
OUTPUT_KEYS = ("images", "gifs", "videos", "video", "audio")


def media_kind(head):
    """(kind, mime) of a file from its first bytes, or None."""
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image", "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image", "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image", "image/webp"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio", "audio/wav"
    if head[4:8] == b"ftyp":
        return "video", "video/mp4"
    if head.startswith(b"\x1aE\xdf\xa3"):
        return "video", "video/webm"
    if head.startswith(b"ID3") or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return "audio", "audio/mpeg"
    if head.startswith(b"fLaC"):
        return "audio", "audio/flac"
    if head.startswith(b"OggS"):
        return "audio", "audio/ogg"
    return None


def _size_of(path):
    try:
        from PIL import Image

        with Image.open(path) as picture:
            return picture.size
    except Exception:
        return None


def collect_media(history_entry, base):
    """Every file the run saved under `base` (ComfyUI's output folder): images,
    animations and videos, audio; at most 16 files of 32 MiB, each a
    `{kind, mime, data, width?, height?}` with base64 data."""
    out = []
    root = os.path.normpath(base) + os.sep
    for output in (history_entry.get("outputs") or {}).values():
        for key in OUTPUT_KEYS:
            for item in output.get(key) or []:
                if len(out) >= MAX_MEDIA:
                    return out
                if not isinstance(item, dict) or item.get("type") != "output" or not item.get("filename"):
                    continue
                path = os.path.normpath(os.path.join(base, item.get("subfolder") or "", item["filename"]))
                if not path.startswith(root) or not os.path.isfile(path):
                    continue
                if os.path.getsize(path) > MAX_MEDIA_BYTES:
                    continue
                with open(path, "rb") as handle:
                    data = handle.read()
                found = media_kind(data[:16])
                if found is None:
                    continue
                entry = {"kind": found[0], "mime": found[1], "data": base64.b64encode(data).decode()}
                if found[0] == "image":
                    size = _size_of(path)
                    if size:
                        entry["width"], entry["height"] = size
                out.append(entry)
    return out


def _media_of(history_entry):
    import folder_paths

    return collect_media(history_entry, folder_paths.get_output_directory())


def listing(record):
    """What the service lists of a published workflow (never its graph)."""
    return {"id": record["id"], "name": record["name"], "description": record.get("description", ""),
            "inputs": record.get("inputs", []), "params": params_of(record)}


def find_published(home, workflow_id):
    for record in published(home):
        if record.get("id") == workflow_id:
            return record
    return None


def prepare(home, body):
    """The graph a run request queues (blocking: reads files, the network's
    components). Raises ValueError."""
    from . import remote

    workflow_id = str(body.get("workflow_id", ""))
    record = find_published(home, workflow_id) if workflow_id.startswith("pub:") else None

    def save_image(data):
        import folder_paths

        return save_base_image(data, folder_paths.get_input_directory())

    if record is not None and isinstance(record.get("params"), list):
        params = record["params"]
        values = dict(body.get("params") or {}) if isinstance(body.get("params"), dict) else {}
        for key in ("prompt", "negative", "image"):
            if values.get(key) is None and body.get(key):
                values[key] = body[key]
        if body.get("image") and not any(p["type"] == "image" for p in params):
            raise ValueError("this workflow takes no base image")
        return apply_params(record["graph"], params, values, save_image)

    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("write what the image should show")
    has_image = bool(body.get("image"))
    entries = remote.network_components() if workflow_id.startswith("builtin:") else []
    graph, inputs_map = resolve(home, workflow_id, entries, base_image=has_image)
    values = dict(body)
    if has_image:
        values["image_name"] = save_image(body["image"])
        values.setdefault("strength", DEFAULT_STRENGTH)
    return substitute(graph, inputs_map, values)


def final_line(media):
    """The run's last NDJSON line: every output, and its images in the earlier
    `images` form for nodes that read only that."""
    images = [{"mime": m["mime"], "width": m.get("width", 0), "height": m.get("height", 0), "data": m["data"]}
              for m in media if m["kind"] == "image"]
    return {"media": media, "images": images[:MAX_COUNT]}


def register(home):
    """The aiohttp routes on ComfyUI's server."""
    import asyncio
    import uuid

    from aiohttp import web
    from server import PromptServer

    _watch_progress()
    routes = PromptServer.instance.routes

    @routes.post("/pom/workflow/run")
    async def run(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "the request is not JSON"}, status=400)
        loop = asyncio.get_running_loop()
        try:
            graph = await loop.run_in_executor(None, prepare, home, body)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)

        response = web.StreamResponse(headers={"content-type": "application/x-ndjson"})
        await response.prepare(request)

        async def line(value):
            await response.write((json.dumps(value) + "\n").encode())

        queue = asyncio.Queue()
        prompt_id = str(uuid.uuid4())

        def listener(event, data):
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        with _listeners_lock:
            _listeners[prompt_id] = listener
        try:
            server = PromptServer.instance
            import execution

            valid = execution.validate_prompt(prompt_id, graph, None)
            if asyncio.iscoroutine(valid):
                valid = await valid
            if not valid[0]:
                await line({"error": f"the workflow is not valid: {valid[1].get('message', valid[1])}"})
                return response
            number = server.number
            server.number += 1
            server.prompt_queue.put((number, prompt_id, graph, {"client_id": "pom-chat"}, valid[2], {}))
            deadline = time.monotonic() + RUN_TIMEOUT
            while time.monotonic() < deadline:
                history = server.prompt_queue.get_history(prompt_id=prompt_id)
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        messages = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                        reason = messages[-1][1].get("exception_message", "the workflow failed") if messages else "the workflow failed"
                        await line({"error": str(reason)[:500]})
                        return response
                    media = await loop.run_in_executor(None, _media_of, entry)
                    if not media:
                        await line({"error": "the workflow saved no image, video or audio"})
                    else:
                        await line(final_line(media))
                    return response
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if event == "progress":
                    await line({"p": data.get("value"), "of": data.get("max")})
            await line({"error": "the workflow took too long"})
            return response
        finally:
            with _listeners_lock:
                _listeners.pop(prompt_id, None)

    @routes.post("/pom/workflow/publish")
    async def publish_route(request):
        loop = asyncio.get_running_loop()
        role = await loop.run_in_executor(None, nodes_role)
        if not role.get("admin"):
            return web.json_response({"error": "only an admin of the network publishes workflows"}, status=403)
        try:
            body = await request.json()
            params = body.get("params")
            workflow_id = publish(home, body.get("name"), body.get("graph"),
                                  params if isinstance(params, list) else (body.get("inputs") or {}),
                                  body.get("description") or "")
        except (ValueError, AttributeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response({"id": workflow_id})

    @routes.get("/pom/workflow/list")
    async def list_route(request):
        return web.json_response({"workflows": [
            listing(r) for r in published(home)
        ]})


def nodes_role():
    """This node's role, through the POM nodes package (`deploy_role`)."""
    import importlib

    package = importlib.import_module(__package__) if __package__ else None
    if package is None or not hasattr(package, "deploy_role"):
        return {"admin": False}
    return package.deploy_role()
