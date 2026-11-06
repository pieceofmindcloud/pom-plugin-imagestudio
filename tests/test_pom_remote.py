"""Tests of runtime/pom_remote.py with ComfyUI stubbed (`folder_paths`,
`nodes`, `comfy`) and a fake node API on loopback.

Run: python3 -m unittest discover -s tests -p 'test_*.py'
"""
import http.server
import importlib.util
import json
import os
import sys
import threading
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _stubs():
    # Shared with the other test modules of this run: add, never replace.
    folder_paths = sys.modules.get("folder_paths") or types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda folder: [f"local-{folder}.safetensors"]

    nodes = types.ModuleType("nodes")

    class UNETLoader:
        def load_unet(self, unet_name, weight_dtype="default"):
            return (f"local:{unet_name}",)

    class CLIPLoader:
        def load_clip(self, clip_name, type="stable_diffusion", device="default"):
            return (f"local:{clip_name}",)

    class VAELoader:
        def load_vae(self, vae_name):
            return (f"local:{vae_name}",)

    nodes.UNETLoader, nodes.CLIPLoader, nodes.VAELoader = UNETLoader, CLIPLoader, VAELoader

    comfy = types.ModuleType("comfy")
    model_patcher = types.ModuleType("comfy.model_patcher")

    class ModelPatcher:
        def __init__(self, model):
            self.model = model
            self.patches = {}

        def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
            self.patches.update(patches)
            return list(patches)

    model_patcher.ModelPatcher = ModelPatcher
    comfy.model_patcher = model_patcher
    return {"folder_paths": folder_paths, "nodes": nodes, "comfy": comfy, "comfy.model_patcher": model_patcher}


for _name, _module in _stubs().items():
    sys.modules[_name] = _module
_spec = importlib.util.spec_from_file_location("pom_remote", ROOT / "runtime" / "pom_remote.py")
pom_remote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pom_remote)

COMPONENTS = [
    {"deploy_id": "img-1", "name": "Z-Image", "component": "diffusion", "files": ["z.safetensors"],
     "status": "ready", "node_id": "a", "hostname": "gpu-a", "plugin_code": "comfyui"},
    {"deploy_id": "img-1", "name": "Z-Image", "component": "vae", "files": ["ae.safetensors"],
     "status": "ready", "node_id": "c", "hostname": "gpu-c", "plugin_code": "comfyui"},
]


class _Node:
    """A fake node API: the components listing, and op calls that answer
    `self.status` with `self.body`."""

    def __init__(self):
        self.status, self.body, self.calls = 200, b"", []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self, status, body, kind="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._answer(200, json.dumps({"components": COMPONENTS}).encode())

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                owner.calls.append((self.path, self.headers.get("x-pom-plugin-host-token"), self.rfile.read(length)))
                self._answer(owner.status, owner.body, "application/octet-stream")

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/api/plugin-host"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


class Remote(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = _Node()
        os.environ["POM_HOST_API_URL"] = cls.node.url
        os.environ["POM_HOST_API_TOKEN"] = "host-token"
        pom_remote.install()

    def setUp(self):
        pom_remote.forget_components()
        self.node.status, self.node.body, self.node.calls[:] = 200, b"", []

    def test_network_components_are_listed_after_local_files(self):
        import folder_paths

        names = folder_paths.get_filename_list("diffusion_models")
        self.assertEqual(names[0], "local-diffusion_models.safetensors")
        self.assertIn("POM ▸ Z-Image ▸ z.safetensors", names)
        self.assertNotIn("POM ▸ Z-Image ▸ ae.safetensors", names)
        self.assertIn("POM ▸ Z-Image ▸ ae.safetensors", folder_paths.get_filename_list("vae"))
        self.assertEqual(folder_paths.get_filename_list("loras"), ["local-loras.safetensors"])

    def test_a_remote_name_returns_a_stand_in_and_a_local_name_the_original(self):
        import nodes

        self.assertEqual(nodes.VAELoader().load_vae("mine.safetensors"), ("local:mine.safetensors",))
        target = pom_remote.target_of("POM ▸ Z-Image ▸ ae.safetensors", "vae")
        self.assertEqual(target["hostname"], "gpu-c")
        self.assertIsNone(pom_remote.target_of("mine.safetensors", "vae"))

    def test_remote_failure_names_the_machine(self):
        self.node.status, self.node.body = 502, json.dumps({"error": "gpu-c did not answer"}).encode()
        target = pom_remote.target_of("POM ▸ Z-Image ▸ ae.safetensors", "vae")
        with self.assertRaises(RuntimeError) as raised:
            pom_remote.call(target, "decode", b"")
        message = str(raised.exception)
        self.assertIn("vae on gpu-c", message)
        self.assertIn("did not answer", message)
        path, token, _ = self.node.calls[-1]
        self.assertEqual(path, "/api/plugin-host/component/img-1/vae/decode")
        self.assertEqual(token, "host-token")

    def test_patches_on_a_remote_model_are_refused(self):
        import comfy.model_patcher

        remote = types.SimpleNamespace(pom_remote=True)
        with self.assertRaises(RuntimeError) as raised:
            comfy.model_patcher.ModelPatcher(remote).add_patches({"w": 1})
        self.assertIn("not supported on a POM network model", str(raised.exception))
        local = comfy.model_patcher.ModelPatcher(types.SimpleNamespace())
        self.assertEqual(local.add_patches({"w": 1}), ["w"])
        self.assertEqual(comfy.model_patcher.ModelPatcher(remote).add_patches({}), [])

    def test_images_on_a_remote_text_encoder_are_refused(self):
        clip = pom_remote.RemoteCLIP({"component": "text_encoder", "hostname": "gpu-b"}, "lumina2")
        self.assertEqual(clip.tokenize("a cat")["pom_text"], "a cat")
        with self.assertRaises(RuntimeError) as raised:
            clip.tokenize("a cat", images=[object()])
        self.assertIn("text only", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
