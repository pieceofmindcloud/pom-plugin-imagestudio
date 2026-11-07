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
BUILTIN_INPUTS = ["prompt", "negative", "width", "height", "seed"]


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


def substitute(graph, inputs_map, values):
    """A copy of `graph` with the Chat's values set: the prompt (and negative)
    text, the size and batch on latent nodes, the seed on samplers (random
    when not given)."""
    graph = json.loads(json.dumps(graph))
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


def publish(home, name, graph, inputs):
    """Stores a graph for the Chat; the prompt node must be a text node of
    the graph. Answers its id."""
    name = (name or "").strip()
    if not name or len(name) > 80:
        raise ValueError("a workflow needs a name of up to 80 characters")
    if not isinstance(graph, dict) or not graph:
        raise ValueError("the workflow is empty")
    prompt = (inputs or {}).get("prompt")
    if prompt not in graph or "text" not in graph[prompt].get("inputs", {}):
        raise ValueError("choose the text node that receives the prompt")
    negative = (inputs or {}).get("negative")
    if negative and (negative not in graph or "text" not in graph[negative].get("inputs", {})):
        raise ValueError("the negative prompt node is not a text node")
    workflow_id = "pub:" + slug(name)
    folder = os.path.join(home, WORKFLOWS_DIR)
    os.makedirs(folder, exist_ok=True)
    listed = ["prompt"] + (["negative"] if negative else [])
    record = {"id": workflow_id, "name": name, "inputs": listed,
              "map": {"prompt": prompt, **({"negative": negative} if negative else {})}, "graph": graph}
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


def resolve(home, workflow_id, entries):
    """(graph, inputs map) of a workflow id, or raises ValueError."""
    if workflow_id.startswith("builtin:"):
        graph = builtin(workflow_id[len("builtin:"):], entries)
        if graph is None:
            raise ValueError("this deployment is missing a stage on the network")
        return graph, BUILTIN_MAP
    for record in published(home):
        if record.get("id") == workflow_id:
            return record["graph"], record.get("map", {})
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


def _images_of(history_entry):
    import folder_paths
    from PIL import Image

    out = []
    for output in (history_entry.get("outputs") or {}).values():
        for image in output.get("images") or []:
            if image.get("type") != "output":
                continue
            base = folder_paths.get_output_directory()
            path = os.path.normpath(os.path.join(base, image.get("subfolder", ""), image["filename"]))
            if not path.startswith(os.path.normpath(base) + os.sep):
                continue
            with open(path, "rb") as handle:
                data = handle.read()
            with Image.open(path) as picture:
                width, height = picture.size
            out.append({"mime": "image/png", "width": width, "height": height,
                        "data": base64.b64encode(data).decode()})
    return out


def register(home):
    """The aiohttp routes on ComfyUI's server."""
    import asyncio
    import uuid

    from aiohttp import web
    from server import PromptServer

    from . import remote

    _watch_progress()
    routes = PromptServer.instance.routes

    @routes.post("/pom/workflow/run")
    async def run(request):
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "the request is not JSON"}, status=400)
        prompt = (body.get("prompt") or "").strip()
        if not prompt:
            return web.json_response({"error": "write what the image should show"}, status=400)
        loop = asyncio.get_running_loop()
        try:
            entries = await loop.run_in_executor(None, remote.network_components)
            graph, inputs_map = resolve(home, str(body.get("workflow_id", "")), entries)
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=404)
        graph = substitute(graph, inputs_map, body)

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
                    images = await loop.run_in_executor(None, _images_of, entry)
                    if not images:
                        await line({"error": "the workflow made no image"})
                    else:
                        await line({"images": images[:MAX_COUNT]})
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
            workflow_id = publish(home, body.get("name"), body.get("graph"), body.get("inputs") or {})
        except (ValueError, AttributeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response({"id": workflow_id})

    @routes.get("/pom/workflow/list")
    async def list_route(request):
        return web.json_response({"workflows": [
            {"id": r["id"], "name": r["name"], "inputs": r.get("inputs", [])} for r in published(home)
        ]})


def nodes_role():
    """This node's role, through the POM nodes package (`deploy_role`)."""
    import importlib

    package = importlib.import_module(__package__) if __package__ else None
    if package is None or not hasattr(package, "deploy_role"):
        return {"admin": False}
    return package.deploy_role()
