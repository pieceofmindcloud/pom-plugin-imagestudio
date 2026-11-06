"""POM image model components served by this ComfyUI to the POM network.

Installed by the POM ComfyUI plugin as `custom_nodes/pom_nodes/components.py`
and rewritten on every start. The plugin writes `<home>/pom-served.json` (what
POM asked this machine to serve); POM's node posts each op to the bridge,
which checks the plugin token and forwards it here:

    POST /pom/component/{deploy_id}/{component}/{op}

Bodies both ways are a safetensors file: the tensors carry the data and
`__metadata__["args"]` a JSON string with every other value. Never pickle.

- diffusion `load`  -> how the client rebuilds the model without weights
- diffusion `apply` -> the model's `apply_model` on the client's inputs
- text_encoder `encode` -> the conditioning of a prompt
- vae `load`, `decode`, `encode`
"""
import json
import os
import threading

SERVED_FILE = "pom-served.json"


# --- Wire ---------------------------------------------------------------------

def pack(tensors, args, lists=None):
    """`tensors`: name -> tensor; `lists`: name -> [tensor...]; `args`: JSON."""
    from safetensors.torch import save

    flat = {name: tensor.detach().to("cpu").contiguous() for name, tensor in tensors.items()}
    sizes = {}
    for name, items in (lists or {}).items():
        sizes[name] = len(items)
        for index, tensor in enumerate(items):
            flat[f"{name}.{index}"] = tensor.detach().to("cpu").contiguous()
    meta = {"args": json.dumps(args), "lists": json.dumps(sizes)}
    return save(flat, metadata=meta)


def unpack(body):
    """The inverse of `pack`: `(tensors, args, lists)`."""
    import struct

    from safetensors.torch import load

    flat = load(body)
    header_len = struct.unpack("<Q", body[:8])[0]
    meta = json.loads(body[8:8 + header_len]).get("__metadata__", {}) or {}
    args = json.loads(meta.get("args", "{}"))
    sizes = json.loads(meta.get("lists", "{}"))
    lists = {name: [flat.pop(f"{name}.{index}") for index in range(count)] for name, count in sizes.items()}
    return flat, args, lists


# --- What this machine serves -------------------------------------------------

class ServedList:
    """`pom-served.json`, read again whenever it changes."""

    def __init__(self, home):
        self.path = os.path.join(home, SERVED_FILE)
        self.mtime = None
        self.entries = []
        self.lock = threading.Lock()

    def get(self, deploy_id, component):
        with self.lock:
            try:
                mtime = os.stat(self.path).st_mtime_ns
            except OSError:
                self.entries, self.mtime = [], None
                return None
            if mtime != self.mtime:
                try:
                    with open(self.path) as handle:
                        self.entries = json.load(handle)
                except (OSError, ValueError):
                    self.entries = []
                self.mtime = mtime
                evict_unserved(self.entries)
            for entry in self.entries:
                if entry.get("deploy_id") == deploy_id and entry.get("component") == component:
                    return entry
            return None


# --- Running the ops (ComfyUI side) -------------------------------------------

_cache = {}
_cache_lock = threading.Lock()
# ComfyUI's model management (`load_models_gpu`, the loaded-model list) is not
# thread-safe: every op this machine serves runs one at a time, whatever the
# deployment or component.
_gpu_lock = threading.Lock()


def _lock_for(key):
    return _gpu_lock


# A serving machine shares its GPU with POM's language models (Strata,
# llama.cpp): ComfyUI keeps a loaded model in VRAM after an op, so after a
# minute without calls the models go back to RAM. The next op reloads them
# from RAM in seconds.
IDLE_UNLOAD_SECONDS = 60


class IdleUnload:
    """When the GPU was used for the network and then left idle."""

    def __init__(self, seconds=IDLE_UNLOAD_SECONDS):
        self.seconds = seconds
        self.last = None
        self.lock = threading.Lock()

    def note_op(self, now):
        with self.lock:
            self.last = now

    def check(self, now, unload):
        """Run `unload` once when an op happened and none since `seconds`."""
        with self.lock:
            if self.last is None or now - self.last < self.seconds:
                return False
            self.last = None
        unload()
        return True


_idle = IdleUnload()


def _unload_gpu():
    import comfy.model_management as mm

    with _gpu_lock:
        mm.unload_all_models()
        mm.soft_empty_cache()
    print("[POM] network models moved off the GPU after a minute idle")


def _watch_idle():
    import time

    while True:
        time.sleep(10)
        try:
            _idle.check(time.monotonic(), _unload_gpu)
        except Exception as error:
            print(f"[POM] could not free the GPU: {error}")


def evict_unserved(entries):
    """Drop the loaded models no served entry uses any more, so a released
    component frees its RAM and VRAM without restarting ComfyUI."""
    served = set()
    for entry in entries if isinstance(entries, list) else []:
        for path in entry.get("paths") or []:
            served.add(path)
    with _cache_lock:
        for key in list(_cache):
            paths = key[1] if isinstance(key[1], tuple) else (key[1],)
            if not any(path in served for path in paths):
                del _cache[key]


def _json_safe(value):
    """`unet_config` and sampling settings hold dtypes and tuples."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"__repr__": str(value)}


def _weight_dtype(options):
    import torch

    name = (options or {}).get("weight_dtype", "default")
    return {
        "fp8_e4m3fn": torch.float8_e4m3fn,
        "fp8_e4m3fn_fast": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }.get(name)


def _diffusion(entry):
    import comfy.sd

    options = entry.get("options") or {}
    key = ("diffusion", entry["paths"][0], json.dumps(options, sort_keys=True))
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    model_options = {}
    dtype = _weight_dtype(options)
    if dtype is not None:
        model_options["dtype"] = dtype
    patcher = comfy.sd.load_diffusion_model(entry["paths"][0], model_options=model_options)
    with _cache_lock:
        _cache[key] = patcher
    return patcher


def _clip(entry, clip_type):
    import comfy.sd
    import folder_paths

    key = ("text_encoder", tuple(entry["paths"]), clip_type)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    kind = getattr(comfy.sd.CLIPType, (clip_type or "stable_diffusion").upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
    clip = comfy.sd.load_clip(
        ckpt_paths=entry["paths"],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=kind,
    )
    with _cache_lock:
        _cache[key] = clip
    return clip


def _vae(entry):
    import comfy.sd
    import comfy.utils

    key = ("vae", entry["paths"][0])
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    sd, metadata = comfy.utils.load_torch_file(entry["paths"][0], return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    vae.throw_exception_if_invalid()
    with _cache_lock:
        _cache[key] = vae
    return vae


def _vae_shape(vae):
    def plain(value):
        return value if isinstance(value, (int, float, list, tuple)) else None

    return {
        "downscale_ratio": _json_safe(plain(vae.downscale_ratio)),
        "upscale_ratio": _json_safe(plain(vae.upscale_ratio)),
        "latent_channels": vae.latent_channels,
        "latent_dim": getattr(vae, "latent_dim", 2),
        "output_channels": getattr(vae, "output_channels", 3),
        "downscale_index_formula": _json_safe(getattr(vae, "downscale_index_formula", None)),
        "upscale_index_formula": _json_safe(getattr(vae, "upscale_index_formula", None)),
    }


def run_op(entry, component, op, body):
    """One op on a served component; returns the answer body."""
    import torch

    tensors, args, lists = unpack(body) if body else ({}, {}, {})
    if component == "diffusion":
        patcher = _diffusion(entry)
        model = patcher.model
        if op == "load":
            return pack({}, {
                "config": type(model.model_config).__name__,
                "unet_config": _json_safe(model.model_config.unet_config),
                "sampling_settings": _json_safe(getattr(model.model_config, "sampling_settings", {})),
                "model_class": type(model).__name__,
                "model_type": model.model_type.name,
                "dtype": str(model.get_dtype()),
                "manual_cast_dtype": str(model.manual_cast_dtype) if model.manual_cast_dtype is not None else None,
            })
        if op == "apply":
            import comfy.model_management as mm

            mm.load_models_gpu([patcher])
            device = patcher.load_device
            kwargs = {name: value.to(device) for name, value in tensors.items() if name not in ("x", "t", "c_concat", "c_crossattn")}
            for name, items in lists.items():
                kwargs[name] = [item.to(device) for item in items]
            kwargs.update(args.get("plain", {}))
            with torch.no_grad():
                out = model.apply_model(
                    tensors["x"].to(device),
                    tensors["t"].to(device),
                    c_concat=tensors["c_concat"].to(device) if "c_concat" in tensors else None,
                    c_crossattn=tensors["c_crossattn"].to(device) if "c_crossattn" in tensors else None,
                    control=None,
                    transformer_options={},
                    **kwargs,
                )
            return pack({"out": out}, {})
    if component == "text_encoder" and op == "encode":
        clip = _clip(entry, args.get("clip_type"))
        with torch.no_grad():
            scheduled = clip.encode_from_tokens_scheduled(clip.tokenize(args.get("text", "")))
        out_tensors, extras = {}, []
        for index, (cond, extra) in enumerate(scheduled):
            out_tensors[f"cond.{index}"] = cond
            plain = {}
            for name, value in extra.items():
                if torch.is_tensor(value):
                    out_tensors[f"extra.{index}.{name}"] = value
                else:
                    plain[name] = _json_safe(value)
            extras.append(plain)
        return pack(out_tensors, {"count": len(scheduled), "extras": extras})
    if component == "vae":
        vae = _vae(entry)
        if op == "load":
            return pack({}, _vae_shape(vae))
        with torch.no_grad():
            if op == "decode":
                return pack({"images": vae.decode(tensors["samples"])}, {})
            if op == "encode":
                return pack({"latent": vae.encode(tensors["pixels"])}, {})
    raise ValueError(f"{component} has no op {op}")


def register(home):
    """The aiohttp route on ComfyUI's server."""
    import asyncio

    from aiohttp import web
    from server import PromptServer

    served = ServedList(home)
    threading.Thread(target=_watch_idle, name="pom-idle-unload", daemon=True).start()
    routes = PromptServer.instance.routes

    @routes.post("/pom/component/{deploy_id}/{component}/{op}")
    async def component(request):
        deploy_id = request.match_info["deploy_id"]
        name = request.match_info["component"]
        op = request.match_info["op"]
        entry = served.get(deploy_id, name)
        if entry is None:
            return web.json_response({"error": f"{name} of {deploy_id} is not served here"}, status=404)
        body = await request.read()

        def work():
            import time

            try:
                with _lock_for((deploy_id, name)):
                    return run_op(entry, name, op, body)
            finally:
                _idle.note_op(time.monotonic())

        try:
            answer = await asyncio.get_running_loop().run_in_executor(None, work)
        except Exception as error:  # the client shows it with this machine's name
            return web.json_response({"error": f"{type(error).__name__}: {error}"[:500]}, status=500)
        return web.Response(body=answer, content_type="application/octet-stream")
