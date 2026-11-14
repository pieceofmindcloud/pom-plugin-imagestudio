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

        def add_hook_patches(self, hook, patches, strength_patch=1.0, strength_model=1.0):
            self.patches.update(patches)

    model_patcher.ModelPatcher = ModelPatcher
    comfy.model_patcher = model_patcher

    execution = types.ModuleType("execution")

    class PromptExecutor:
        """ComfyUI's executor: `add_message` records every status event,
        and sends it to a browser only when one queued the prompt."""

        def __init__(self):
            self.status_messages = []

        def add_message(self, event, data, broadcast):
            self.status_messages.append((event, data))

    execution.PromptExecutor = PromptExecutor
    return {"folder_paths": folder_paths, "nodes": nodes, "comfy": comfy, "comfy.model_patcher": model_patcher,
            "execution": execution}


for _name, _module in _stubs().items():
    if _name == "folder_paths":
        sys.modules[_name] = _module
    else:
        sys.modules[_name] = _module


class _PromptServer:
    """ComfyUI's server as the client sees it: the running prompt and the
    events it sends to the browser."""

    def __init__(self):
        self.last_prompt_id = None
        self.sent = []

    def send_sync(self, event, data, sid=None):
        self.sent.append((event, data))


_server = sys.modules.get("server") or types.ModuleType("server")
_server.PromptServer = types.SimpleNamespace(instance=_PromptServer())
sys.modules["server"] = _server
_spec = importlib.util.spec_from_file_location("pom_remote", ROOT / "runtime" / "pom_remote.py")
pom_remote = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pom_remote)

COMPONENTS = [
    {"deploy_id": "img-1", "name": "Z-Image", "component": "diffusion", "files": ["z.safetensors"],
     "status": "ready", "node_id": "a", "hostname": "gpu-a", "plugin_code": "imagestudio"},
    {"deploy_id": "img-1", "name": "Z-Image", "component": "vae", "files": ["ae.safetensors"],
     "status": "ready", "node_id": "c", "hostname": "gpu-c", "plugin_code": "imagestudio"},
]


class _Node:
    """A fake node API: the components listing, and op calls that answer
    `self.status` with `self.body`."""

    def __init__(self):
        self.status, self.body, self.calls = 200, b"", []
        self.jobs, self.refuse_begin, self.drop = [], False, False
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
                body = self.rfile.read(length)
                if self.path.endswith("/image-job/begin") or self.path.endswith("/image-job/complete"):
                    owner.jobs.append((self.path.rsplit("/", 1)[1], json.loads(body)))
                    if self.path.endswith("/begin") and owner.refuse_begin:
                        self._answer(429, json.dumps({"error": "Image budget exhausted"}).encode())
                    else:
                        self._answer(200, b"{}")
                    return
                owner.calls.append((self.path, self.headers.get("x-pom-plugin-host-token"), body,
                                    self.headers.get("x-pom-job")))
                if owner.drop:
                    # The machine goes away halfway through its answer.
                    self.send_response(200)
                    self.send_header("Content-Length", "1000")
                    self.end_headers()
                    self.wfile.write(b"abc")
                    self.wfile.flush()
                    self.connection.shutdown(2)
                    return
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
        pom_remote.forget_jobs()
        self.node.status, self.node.body, self.node.calls[:] = 200, b"", []
        self.node.jobs[:], self.node.refuse_begin, self.node.drop = [], False, False
        self.server = sys.modules["server"].PromptServer.instance
        self.server.last_prompt_id = None

    def test_the_prompt_is_read_from_the_graph_before_the_encoder_runs(self):
        graph = {
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "a red fox"}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
            "7": {"class_type": "KSampler", "inputs": {"positive": ["4", 0], "negative": ["5", 0]}},
        }
        self.assertEqual(pom_remote.prompt_text_of(graph), "a red fox")
        graph["7"]["inputs"]["positive"] = ["5", 0]
        self.assertEqual(pom_remote.prompt_text_of(graph), "blurry")
        self.assertEqual(pom_remote.prompt_text_of({"1": {"inputs": {"seed": 1}}}), "")

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
        path, token, _, _ = self.node.calls[-1]
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

    def test_a_dropped_answer_names_the_machine(self):
        self.node.drop = True
        target = pom_remote.target_of("POM ▸ Z-Image ▸ ae.safetensors", "vae")
        with self.assertRaises(RuntimeError) as raised:
            pom_remote.call(target, "decode", b"")
        self.assertIn("vae on gpu-c", str(raised.exception))

    def test_only_loading_waits_long(self):
        self.assertGreater(pom_remote.timeout_of("load"), pom_remote.timeout_of("apply"))
        self.assertLessEqual(pom_remote.timeout_of("apply"), 5 * 60)

    def test_attention_and_block_patches_on_a_remote_model_are_refused(self):
        for options in ({"patches": {"attn1_patch": [object()]}},
                        {"patches_replace": {"attn1": {("input", 1): object()}}}):
            with self.assertRaises(RuntimeError) as raised:
                pom_remote.refuse_patches(options)
            self.assertIn("not supported on a POM network model", str(raised.exception))
        pom_remote.refuse_patches({"cond_or_uncond": [0, 1], "patches": {}, "patches_replace": {}})

    def test_hook_patches_on_a_remote_model_are_refused(self):
        import comfy.model_patcher

        remote = types.SimpleNamespace(pom_remote=True)
        with self.assertRaises(RuntimeError):
            comfy.model_patcher.ModelPatcher(remote).add_hook_patches(object(), {"w": 1})
        comfy.model_patcher.ModelPatcher(types.SimpleNamespace()).add_hook_patches(object(), {"w": 1})

    def test_images_on_a_remote_text_encoder_are_refused(self):
        clip = pom_remote.RemoteCLIP({"component": "text_encoder", "hostname": "gpu-b"}, "lumina2")
        self.assertEqual(clip.tokenize("a cat")["pom_text"], "a cat")
        with self.assertRaises(RuntimeError) as raised:
            clip.tokenize("a cat", images=[object()])
        self.assertIn("text only", str(raised.exception))


class Generations(Remote):
    def target(self):
        return pom_remote.target_of("POM ▸ Z-Image ▸ ae.safetensors", "vae")

    def test_a_generation_begins_once_and_completes_with_its_images(self):
        self.server.last_prompt_id = "p1"
        pom_remote.call(self.target(), "decode", b"x")
        pom_remote.call(self.target(), "decode", b"x")
        pom_remote.note_images("p1", 2)
        self.server.send_sync("execution_success", {"prompt_id": "p1"})
        kinds = [kind for kind, _ in self.node.jobs]
        self.assertEqual(kinds, ["begin", "complete"])
        begin, complete = self.node.jobs[0][1], self.node.jobs[1][1]
        self.assertEqual(begin["job_id"], "p1")
        self.assertEqual(begin["model"], "Z-Image")
        self.assertEqual(complete["status"], "ok")
        self.assertEqual(complete["images"], 2)
        self.assertEqual(self.server.sent[-1][0], "execution_success", "the browser still gets it")

    def test_a_spent_budget_stops_before_the_first_call(self):
        self.server.last_prompt_id = "p2"
        self.node.refuse_begin = True
        with self.assertRaises(RuntimeError) as raised:
            pom_remote.call(self.target(), "decode", b"x")
        self.assertIn("Image budget", str(raised.exception))
        self.assertEqual(self.node.calls, [], "no GPU call went out")

    def test_every_call_carries_the_job(self):
        self.server.last_prompt_id = "p3"
        pom_remote.call(self.target(), "decode", b"x")
        self.assertEqual(self.node.calls[-1][3], "p3")

    def test_an_error_completes_the_job_with_its_message(self):
        self.server.last_prompt_id = "p4"
        pom_remote.call(self.target(), "decode", b"x")
        self.server.send_sync("execution_error", {"prompt_id": "p4", "exception_message": "out of memory"})
        complete = self.node.jobs[-1][1]
        self.assertEqual(complete["status"], "error")
        self.assertIn("out of memory", complete["error"])

    def test_a_generation_queued_without_a_browser_is_closed_too(self):
        import execution

        self.server.last_prompt_id = "p6"
        pom_remote.call(self.target(), "decode", b"x")
        pom_remote.note_images("p6", 1)
        # ComfyUI sends nothing to a browser here: no client queued it.
        execution.PromptExecutor().add_message("execution_success", {"prompt_id": "p6"}, False)
        self.assertEqual([kind for kind, _ in self.node.jobs], ["begin", "complete"])
        self.assertEqual(self.node.jobs[-1][1]["images"], 1)

    def test_a_generation_without_network_models_reports_nothing(self):
        self.server.last_prompt_id = "p5"
        self.server.send_sync("execution_success", {"prompt_id": "p5"})
        self.assertEqual(self.node.jobs, [])


if __name__ == "__main__":
    unittest.main()
