"""Image model components served on the POM network, in ComfyUI's ordinary
loader nodes.

Installed by the POM ComfyUI plugin as `custom_nodes/pom_nodes/remote.py`.
`install()` appends `POM ▸ <name> ▸ <file>` to the lists of Load Diffusion
Model, Load CLIP and Load VAE, and makes those loaders return stand-ins for
such a name:

- a `MODEL` built from ComfyUI's own model classes without weights, whose
  `apply_model` (one denoising step) runs on the machine serving it;
- a `CLIP` whose text encoding runs remotely;
- a `VAE` whose decode and encode run remotely.

Every call goes to this node's POM API (`POM_HOST_API_URL`), which forwards it
to the serving machine. Bodies are safetensors (`components.pack`/`unpack`).
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request

PREFIX = "POM ▸ "
SEPARATOR = " ▸ "
# ComfyUI folder -> component listed in it.
FOLDERS = {"diffusion_models": "diffusion", "text_encoders": "text_encoder", "vae": "vae"}
LIST_TTL = 5.0
CALL_TIMEOUT = 15 * 60

_components = {"at": 0.0, "list": []}
_components_lock = threading.Lock()


def forget_components():
    with _components_lock:
        _components["at"], _components["list"] = 0.0, []


def _host_api():
    return os.environ.get("POM_HOST_API_URL", ""), os.environ.get("POM_HOST_API_TOKEN", "")


def network_components():
    """The network's components, asked again at most every few seconds; an
    unreachable node lists nothing."""
    with _components_lock:
        if time.monotonic() - _components["at"] < LIST_TTL:
            return list(_components["list"])
    base, token = _host_api()
    found = []
    if base and token:
        try:
            request = urllib.request.Request(base + "/components")
            request.add_header("x-pom-plugin-host-token", token)
            with urllib.request.urlopen(request, timeout=5) as response:
                found = json.loads(response.read()).get("components", [])
        except Exception:
            found = []
    with _components_lock:
        _components["at"], _components["list"] = time.monotonic(), found
    return list(found)


def name_of(entry):
    files = entry.get("files") or ["?"]
    return f"{PREFIX}{entry.get('name', '')}{SEPARATOR}{files[0]}"


def target_of(name, component):
    """The network component a listed name stands for, or None."""
    if not isinstance(name, str) or not name.startswith(PREFIX):
        return None
    for entry in network_components():
        if entry.get("component") == component and name_of(entry) == name:
            return entry
    return None


def _current_job():
    try:
        from server import PromptServer

        return getattr(PromptServer.instance, "last_prompt_id", None)
    except Exception:
        return None


def call(target, op, body):
    """Run `op` on the machine serving `target`; the answer's body, or a
    RuntimeError naming the component and the machine."""
    base, token = _host_api()
    where = f"{target.get('component')} on {target.get('hostname')}"
    if not base or not token:
        raise RuntimeError(f"{where}: this ComfyUI is not connected to its POM node")
    url = f"{base}/component/{target.get('deploy_id')}/{target.get('component')}/{op}"
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("x-pom-plugin-host-token", token)
    request.add_header("Content-Type", "application/octet-stream")
    job = _current_job()
    if job:
        request.add_header("x-pom-job", str(job))
    try:
        with urllib.request.urlopen(request, timeout=CALL_TIMEOUT) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read()[:500].decode(errors="replace")
        try:
            detail = json.loads(detail).get("error", detail)
        except ValueError:
            pass
        raise RuntimeError(f"{where}: {detail}") from None
    except urllib.error.URLError as error:
        raise RuntimeError(f"{where}: {error.reason}") from None


# --- Stand-ins ------------------------------------------------------------------

def _wire():
    from . import components  # noqa: PLC0415 - ComfyUI side only

    return components


def _restore(value):
    """`components._json_safe` writes dtypes as {"__repr__": "torch.bfloat16"}."""
    import torch

    if isinstance(value, dict):
        if set(value) == {"__repr__"}:
            text = value["__repr__"]
            if text.startswith("torch."):
                return getattr(torch, text[len("torch."):], text)
            return text
        return {k: _restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore(v) for v in value]
    return value


def _dtype(text):
    import torch

    if not text or not str(text).startswith("torch."):
        return None
    return getattr(torch, str(text)[len("torch."):], None)


def remote_model(target, weight_dtype):
    """A MODEL whose denoising steps run on the serving machine."""
    import torch

    import comfy.model_base
    import comfy.model_patcher
    import comfy.supported_models

    wire = _wire()
    _, info, _ = wire.unpack(call(target, "load", wire.pack({}, {"options": {"weight_dtype": weight_dtype}})))
    config = getattr(comfy.supported_models, info["config"])(_restore(info["unet_config"]))
    config.sampling_settings.update(_restore(info.get("sampling_settings") or {}))
    config.unet_config["disable_unet_model_creation"] = True
    config.manual_cast_dtype = _dtype(info.get("manual_cast_dtype"))
    model_class = getattr(comfy.model_base, info["model_class"])
    model_type = comfy.model_base.ModelType[info["model_type"]]
    model = model_class(config, model_type=model_type, device=torch.device("cpu"))
    dtype = _dtype(info.get("dtype")) or torch.float32

    class Weightless(torch.nn.Module):
        """Stands where the network's weights would be."""

        def __init__(self):
            super().__init__()
            self.dtype = dtype

    model.diffusion_model = Weightless()
    model.pom_remote = True
    model.get_dtype = lambda: dtype

    def apply_model(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        if control is not None:
            raise RuntimeError("ControlNet on a POM network model is not supported yet")
        tensors, lists, plain = {"x": x, "t": t}, {}, {}
        if c_concat is not None:
            tensors["c_concat"] = c_concat
        if c_crossattn is not None:
            tensors["c_crossattn"] = c_crossattn
        for name, value in kwargs.items():
            if torch.is_tensor(value):
                tensors[name] = value
            elif isinstance(value, (list, tuple)) and value and all(torch.is_tensor(v) for v in value):
                lists[name] = list(value)
            else:
                plain[name] = value
        try:
            body = wire.pack(tensors, {"plain": plain}, lists)
        except TypeError:
            raise RuntimeError("this model's conditioning cannot cross the network yet") from None
        out, _, _ = wire.unpack(call(target, "apply", body))
        return out["out"].to(x.device)

    model.apply_model = apply_model
    patcher = comfy.model_patcher.ModelPatcher(model, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    return patcher


class RemoteCLIP:
    """A CLIP whose text encoding runs on the serving machine."""

    def __init__(self, target, clip_type):
        self.target = target
        self.clip_type = clip_type
        self.patcher = None

    def clone(self):
        return self

    def tokenize(self, text, return_word_ids=False, **kwargs):
        if kwargs.get("images"):
            raise RuntimeError("Remote text encoders take text only")
        return {"pom_text": text}

    def _encode(self, tokens):
        wire = _wire()
        body = wire.pack({}, {"text": tokens.get("pom_text", ""), "clip_type": self.clip_type})
        tensors, args, _ = wire.unpack(call(self.target, "encode", body))
        out = []
        for index in range(args.get("count", 0)):
            extra = dict(_restore((args.get("extras") or [{}])[index] or {}))
            prefix = f"extra.{index}."
            for name, value in tensors.items():
                if name.startswith(prefix):
                    extra[name[len(prefix):]] = value
            out.append([tensors[f"cond.{index}"], extra])
        return out

    def encode_from_tokens_scheduled(self, tokens, unprojected=False, add_dict={}, show_pbar=True):
        out = self._encode(tokens)
        for entry in out:
            entry[1].update(add_dict)
        return out

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        cond, extra = self._encode(tokens)[0]
        if return_dict:
            return {"cond": cond, **extra}
        if return_pooled:
            return cond, extra.get("pooled_output")
        return cond

    def clip_layer(self, layer_idx):
        raise RuntimeError("Set CLIP last layer is not supported on a POM network text encoder yet")

    def add_patches(self, *args, **kwargs):
        raise RuntimeError("LoRAs and model patches are not supported on a POM network model yet")

    def load_model(self, *args, **kwargs):
        raise RuntimeError("A POM network text encoder has no local weights to load")


class RemoteVAE:
    """A VAE whose decode and encode run on the serving machine."""

    def __init__(self, target):
        wire = _wire()
        self.target = target
        _, shape, _ = wire.unpack(call(target, "load", wire.pack({}, {})))
        for name, value in shape.items():
            setattr(self, name, tuple(value) if isinstance(value, list) else value)
        self.output_device = None

    def decode(self, samples_in, vae_options={}):
        wire = _wire()
        images, _, _ = wire.unpack(call(self.target, "decode", wire.pack({"samples": samples_in}, {})))
        return images["images"]

    def encode(self, pixel_samples):
        wire = _wire()
        latent, _, _ = wire.unpack(call(self.target, "encode", wire.pack({"pixels": pixel_samples}, {})))
        return latent["latent"]

    def spacial_compression_decode(self):
        ratio = self.downscale_ratio
        return ratio[-1] if isinstance(ratio, tuple) else ratio

    def spacial_compression_encode(self):
        return self.spacial_compression_decode()

    def temporal_compression_decode(self):
        return None

    def throw_exception_if_invalid(self):
        return None


# --- Hooks ----------------------------------------------------------------------

_installed = False


def install():
    """List the network's components in ComfyUI's loaders and answer them with
    stand-ins; patches on a remote MODEL are refused."""
    global _installed
    if _installed:
        return
    _installed = True
    import folder_paths
    import nodes

    original_list = folder_paths.get_filename_list

    def get_filename_list(folder):
        names = original_list(folder)
        component = FOLDERS.get(folder)
        if component is None:
            return names
        remote = [name_of(entry) for entry in network_components() if entry.get("component") == component]
        return list(names) + [name for name in remote if name not in names]

    folder_paths.get_filename_list = get_filename_list

    load_unet = nodes.UNETLoader.load_unet

    def unet(self, unet_name, weight_dtype="default", *args, **kwargs):
        target = target_of(unet_name, "diffusion")
        if target is None:
            return load_unet(self, unet_name, weight_dtype, *args, **kwargs)
        return (remote_model(target, weight_dtype),)

    nodes.UNETLoader.load_unet = unet

    load_clip = nodes.CLIPLoader.load_clip

    def clip(self, clip_name, type="stable_diffusion", device="default", *args, **kwargs):
        target = target_of(clip_name, "text_encoder")
        if target is None:
            return load_clip(self, clip_name, type, device, *args, **kwargs)
        return (RemoteCLIP(target, type),)

    nodes.CLIPLoader.load_clip = clip

    load_vae = nodes.VAELoader.load_vae

    def vae(self, vae_name, *args, **kwargs):
        target = target_of(vae_name, "vae")
        if target is None:
            return load_vae(self, vae_name, *args, **kwargs)
        return (RemoteVAE(target),)

    nodes.VAELoader.load_vae = vae

    import comfy.model_patcher

    add_patches = comfy.model_patcher.ModelPatcher.add_patches

    def guarded(self, patches, *args, **kwargs):
        if patches and getattr(self.model, "pom_remote", False):
            raise RuntimeError("LoRAs and model patches are not supported on a POM network model yet")
        return add_patches(self, patches, *args, **kwargs)

    comfy.model_patcher.ModelPatcher.add_patches = guarded
