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
                body = (b'{"image_files": []}' if self.path.startswith("/api/plugin-host/")
                        else owner.payload)
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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

    def test_dot_segments_cannot_escape(self):
        for url in ("https://huggingface.co/../../evil/resolve/main/x.safetensors",
                    "https://huggingface.co/./../repo/resolve/main/x.safetensors",
                    "https://huggingface.co/x.safetensors",
                    "https://civitai.com/api/download/models/..",
                    "https://civitai.com/"):
            with self.assertRaises(ValueError, msg=url):
                pom_nodes.relative_destination(url, "x.safetensors")

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


def _file(path, category, error=None, name=None):
    return {"path": path, "name": name or os.path.basename(path), "category": category,
            "metadata_error": error}


class Links(unittest.TestCase):
    def setUp(self):
        self.models = tempfile.mkdtemp(prefix="pom-comfy-links-")
        self.store = tempfile.mkdtemp(prefix="pom-store-")
        self.vae = os.path.join(self.store, "ae.safetensors")
        self.unet = os.path.join(self.store, "z.safetensors")
        for path in (self.vae, self.unet):
            Path(path).write_bytes(b"x")

    def test_plan_links_maps_categories(self):
        plan = pom_nodes.plan_links([
            _file(self.vae, "vae", name="Comfy-Org/z/ae.safetensors"),
            _file(self.unet, "diffusion", name="Comfy-Org/z/z.safetensors"),
            _file(self.unet, "other", name="x/y/other.safetensors"),
            _file(self.vae, "vae", error="cut short", name="x/y/broken.safetensors"),
        ], self.models)
        self.assertEqual(plan, {
            os.path.join(self.models, "vae", "pom", "Comfy-Org", "z", "ae.safetensors"): self.vae,
            os.path.join(self.models, "diffusion_models", "pom", "Comfy-Org", "z",
                         "z.safetensors"): self.unet,
        })
        self.assertEqual(pom_nodes.CATEGORY_FOLDERS["text_encoder"], "text_encoders")
        self.assertEqual(pom_nodes.CATEGORY_FOLDERS["upscale"], "upscale_models")
        self.assertNotIn("other", pom_nodes.CATEGORY_FOLDERS)

    def test_sync_creates_the_links(self):
        plan = pom_nodes.plan_links([_file(self.vae, "vae", name="a/b/ae.safetensors")], self.models)
        pom_nodes.sync_links_to(self.models, plan)
        link = os.path.join(self.models, "vae", "pom", "a", "b", "ae.safetensors")
        self.assertEqual(os.path.realpath(link), os.path.realpath(self.vae))

    def test_stale_links_are_removed(self):
        first = pom_nodes.plan_links([_file(self.vae, "vae", name="a/b/ae.safetensors")], self.models)
        pom_nodes.sync_links_to(self.models, first)
        pom_nodes.sync_links_to(self.models, {})
        self.assertFalse(os.path.lexists(os.path.join(self.models, "vae", "pom", "a", "b",
                                                      "ae.safetensors")))
        self.assertFalse(os.path.exists(os.path.join(self.models, "vae", "pom", "a")))

    def test_only_links_under_pom_are_touched(self):
        mine = os.path.join(self.models, "vae", "mine.safetensors")
        os.makedirs(os.path.dirname(mine))
        Path(mine).write_bytes(b"user")
        inside = os.path.join(self.models, "vae", "pom", "regular.safetensors")
        os.makedirs(os.path.dirname(inside))
        Path(inside).write_bytes(b"user")
        pom_nodes.sync_links_to(self.models, {})
        self.assertTrue(os.path.isfile(mine))
        self.assertTrue(os.path.isfile(inside))

    def test_without_symlinks_the_folder_is_registered(self):
        registered = []
        folder_paths = sys.modules["folder_paths"]
        folder_paths.add_model_folder_path = lambda folder, path: registered.append((folder, path))
        original = os.symlink

        def refuse(*args, **kwargs):
            raise OSError("A required privilege is not held by the client")

        os.symlink = refuse
        try:
            plan = pom_nodes.plan_links([_file(self.vae, "vae", name="a/b/ae.safetensors")],
                                        self.models)
            pom_nodes.sync_links_to(self.models, plan)
        finally:
            os.symlink = original
            del folder_paths.add_model_folder_path
        self.assertEqual(registered, [("vae", os.path.dirname(self.vae))])

    def test_templates_find_the_file_by_its_plain_name(self):
        files = [_file(self.vae, "vae", name="Comfy-Org/z/ae.safetensors")]
        pom_nodes.sync_links_to(self.models, pom_nodes.plan_links(files, self.models),
                                pom_nodes.plan_aliases(files, self.models))
        alias = os.path.join(self.models, "vae", "ae.safetensors")
        self.assertEqual(os.path.realpath(alias), os.path.realpath(self.vae))

    def test_an_alias_never_replaces_a_users_file(self):
        mine = os.path.join(self.models, "vae", "ae.safetensors")
        os.makedirs(os.path.dirname(mine))
        Path(mine).write_bytes(b"user")
        files = [_file(self.vae, "vae", name="a/b/ae.safetensors")]
        pom_nodes.sync_links_to(self.models, {}, pom_nodes.plan_aliases(files, self.models))
        self.assertFalse(os.path.islink(mine))
        self.assertEqual(Path(mine).read_bytes(), b"user")

    def test_stale_aliases_are_removed_and_only_ours(self):
        files = [_file(self.vae, "vae", name="a/b/ae.safetensors")]
        pom_nodes.sync_links_to(self.models, {}, pom_nodes.plan_aliases(files, self.models))
        other = os.path.join(self.models, "vae", "theirs.safetensors")
        os.symlink(self.unet, other)
        pom_nodes.sync_links_to(self.models, {}, {})
        self.assertFalse(os.path.lexists(os.path.join(self.models, "vae", "ae.safetensors")))
        self.assertTrue(os.path.islink(other), "a link POM did not make stays")

    def test_two_files_with_one_name_get_no_alias(self):
        files = [_file(self.vae, "vae", name="a/b/ae.safetensors"),
                 _file(self.unet, "vae", name="c/d/ae.safetensors")]
        self.assertEqual(pom_nodes.plan_aliases(files, self.models), {})

    def test_sync_is_idempotent(self):
        plan = pom_nodes.plan_links([_file(self.vae, "vae", name="a/b/ae.safetensors"),
                                     _file(self.unet, "diffusion", name="a/b/z.safetensors")],
                                    self.models)
        pom_nodes.sync_links_to(self.models, plan)
        before = sorted(str(p) for p in Path(self.models).rglob("*"))
        pom_nodes.sync_links_to(self.models, plan)
        self.assertEqual(sorted(str(p) for p in Path(self.models).rglob("*")), before)


if __name__ == "__main__":
    unittest.main()
