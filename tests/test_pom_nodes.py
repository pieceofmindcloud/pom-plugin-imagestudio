"""Tests of runtime/pom_nodes.py without ComfyUI: `folder_paths` and
`server` are stubbed before the module is loaded by path.

Run: python3 -m unittest discover -s tests -p 'test_*.py'
"""
import http.server
import importlib.util
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMFY_MODELS = tempfile.mkdtemp(prefix="pom-comfy-models-")
COMFY_USER = tempfile.mkdtemp(prefix="pom-comfy-user-")


def _stub_folder_paths():
    module = types.ModuleType("folder_paths")
    folders = ["checkpoints", "diffusion_models", "text_encoders", "vae", "loras"]
    module.folder_names_and_paths = {
        name: ([os.path.join(COMFY_MODELS, name)], {".safetensors"}) for name in folders
    }
    module.get_folder_paths = lambda name: module.folder_names_and_paths[name][0]
    module.get_user_directory = lambda: COMFY_USER
    module.models_dir = COMFY_MODELS
    return module


sys.modules["folder_paths"] = _stub_folder_paths()
sys.modules.setdefault("server", types.ModuleType("server"))
_spec = importlib.util.spec_from_file_location("pom_nodes", ROOT / "runtime" / "pom_nodes.py")
pom_nodes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pom_nodes)
pom_nodes._publish = lambda state: None  # no PromptServer here


class _Server:
    """A loopback server: GET /file.safetensors serves bytes, POST
    /api/plugin-host/catalog/refresh records the token header."""

    def __init__(self, payload):
        self.payload = payload
        self.refreshes = []
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(owner.payload)))
                self.end_headers()
                self.wfile.write(owner.payload)

            def do_POST(self):
                owner.refreshes.append((self.path, self.headers.get("x-pom-plugin-host-token")))
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


class Destinations(unittest.TestCase):
    def test_hugging_face_layout(self):
        url = ("https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/"
               "split_files/vae/ae.safetensors")
        self.assertEqual(pom_nodes.relative_destination(url, "ae.safetensors"),
                         "Comfy-Org/z_image_turbo/ae.safetensors")

    def test_civitai_layout(self):
        for host in ("civitai.com", "civitai.red"):
            url = f"https://{host}/api/download/models/12345?type=Model"
            self.assertEqual(pom_nodes.relative_destination(url, "x.safetensors"),
                             "civitai/12345/x.safetensors")

    def test_picks_the_first_folder_with_room(self):
        gib = 2**30
        space = {"/full": 1 * gib, "/roomy": 50 * gib}
        free = lambda path: types.SimpleNamespace(free=space[path])
        self.assertEqual(pom_nodes.choose_root(["/full", "/roomy"], 10 * gib, free), "/roomy")
        self.assertIsNone(pom_nodes.choose_root(["/full"], 10 * gib, free))
        self.assertIsNone(pom_nodes.choose_root([], 1, free))


class Downloads(unittest.TestCase):
    def setUp(self):
        self.server = _Server(b"x" * 1000)
        self.root = tempfile.mkdtemp(prefix="pom-model-dir-")
        os.environ["POM_MODEL_DIRS"] = self.root
        os.environ["POM_HOST_API_URL"] = self.server.url + "/api/plugin-host"
        os.environ["POM_HOST_API_TOKEN"] = "host-token"

    def tearDown(self):
        self.server.close()
        for key in ("POM_MODEL_DIRS", "POM_HOST_API_URL", "POM_HOST_API_TOKEN"):
            os.environ.pop(key, None)

    def _fetch(self, url, name="ae.safetensors", directory="vae"):
        state = pom_nodes.new_state(url, name, directory)
        pom_nodes._fetch(state)
        return state

    def test_a_download_lands_in_the_model_folder_and_refreshes_the_catalog(self):
        url = self.server.url + "/Comfy-Org/z_image_turbo/resolve/main/vae/ae.safetensors"
        state = self._fetch(url)
        self.assertEqual(state["status"], "done", state["error"])
        self.assertEqual(state["path"],
                         os.path.join(self.root, "Comfy-Org", "z_image_turbo", "ae.safetensors"))
        self.assertEqual(os.path.getsize(state["path"]), 1000)
        self.assertFalse(os.path.exists(state["path"] + ".part"))
        self.assertIn(("/api/plugin-host/catalog/refresh", "host-token"), self.server.refreshes)

    def test_no_folder_with_room_fails_before_writing(self):
        original = pom_nodes.choose_root
        pom_nodes.choose_root = lambda dirs, size, free=None: None
        try:
            state = self._fetch(self.server.url + "/a/b/resolve/main/ae.safetensors")
        finally:
            pom_nodes.choose_root = original
        self.assertEqual(state["status"], "error")
        self.assertIn("disk space", state["error"])
        self.assertEqual(os.listdir(self.root), [])

    def test_falls_back_to_comfyui_models_without_dirs(self):
        os.environ.pop("POM_MODEL_DIRS")
        state = self._fetch(self.server.url + "/a/b/resolve/main/ae.safetensors")
        self.assertEqual(state["status"], "done", state["error"])
        self.assertEqual(state["path"], os.path.join(COMFY_MODELS, "vae", "ae.safetensors"))


if __name__ == "__main__":
    unittest.main()
