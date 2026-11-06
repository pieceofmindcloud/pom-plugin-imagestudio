"""Tests of runtime/pom_components.py. The wire tests need torch and
safetensors: run them with ComfyUI's venv (the live test does), they are
skipped by a plain interpreter.

Run: python3 -m unittest discover -s tests -p 'test_*.py'
"""
import importlib.util
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.modules.setdefault("server", types.ModuleType("server"))
_spec = importlib.util.spec_from_file_location("pom_components", ROOT / "runtime" / "pom_components.py")
pom_components = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pom_components)

try:
    import torch  # noqa: F401
    import safetensors  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch and safetensors are in ComfyUI's venv only")
class Wire(unittest.TestCase):
    def test_pack_round_trips_tensors_lists_and_args(self):
        import torch

        tensors = {"x": torch.ones(2, 3), "t": torch.tensor([0.5])}
        lists = {"ref_latents": [torch.zeros(1, 4), torch.full((2,), 7.0)]}
        args = {"text": "a lighthouse", "n": 3, "shape": [[1, 16, 8, 8]]}
        body = pom_components.pack(tensors, args, lists)
        got_tensors, got_args, got_lists = pom_components.unpack(body)
        self.assertTrue(torch.equal(got_tensors["x"], tensors["x"]))
        self.assertTrue(torch.equal(got_tensors["t"], tensors["t"]))
        self.assertEqual(len(got_lists["ref_latents"]), 2)
        self.assertTrue(torch.equal(got_lists["ref_latents"][1], lists["ref_latents"][1]))
        self.assertEqual(got_args, args)

    def test_bfloat16_crosses_unchanged(self):
        import torch

        x = torch.randn(4, 4).to(torch.bfloat16)
        got, _, _ = pom_components.unpack(pom_components.pack({"x": x}, {}))
        self.assertEqual(got["x"].dtype, torch.bfloat16)
        self.assertTrue(torch.equal(got["x"], x))


class Served(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="pom-comfy-home-")
        self.served = pom_components.ServedList(self.home)

    def write(self, entries):
        path = os.path.join(self.home, "pom-served.json")
        with open(path, "w") as handle:
            json.dump(entries, handle)
        # A later mtime than the previous write, even on a coarse clock.
        later = time.time() + len(entries) + 1
        os.utime(path, (later, later))

    def test_unserved_components_are_refused(self):
        self.write([{"deploy_id": "img-1", "component": "vae", "paths": ["/m/ae.safetensors"], "options": {}}])
        self.assertEqual(self.served.get("img-1", "vae")["paths"], ["/m/ae.safetensors"])
        self.assertIsNone(self.served.get("img-1", "diffusion"))
        self.assertIsNone(self.served.get("img-2", "vae"))

    def test_served_list_is_reread_when_it_changes(self):
        self.write([{"deploy_id": "img-1", "component": "vae", "paths": ["/a"], "options": {}}])
        self.assertIsNotNone(self.served.get("img-1", "vae"))
        self.write([])
        self.assertIsNone(self.served.get("img-1", "vae"))

    def test_a_missing_file_serves_nothing(self):
        self.assertIsNone(self.served.get("img-1", "vae"))

    def test_a_released_model_leaves_the_cache(self):
        cache = pom_components._cache
        cache.clear()
        cache[("vae", "/m/ae.safetensors")] = "kept"
        cache[("diffusion", "/m/old.safetensors", "{}")] = "released"
        cache[("text_encoder", ("/m/qwen.safetensors",), "lumina2")] = "kept too"
        self.write([
            {"deploy_id": "img-1", "component": "vae", "paths": ["/m/ae.safetensors"], "options": {}},
            {"deploy_id": "img-1", "component": "text_encoder", "paths": ["/m/qwen.safetensors"], "options": {}},
        ])
        self.served.get("img-1", "vae")
        self.assertEqual(sorted(cache.values()), ["kept", "kept too"])

    def test_an_idle_machine_gives_the_gpu_back(self):
        unloads = []
        idle = pom_components.IdleUnload(seconds=60)
        self.assertFalse(idle.check(100.0, lambda: unloads.append("x")), "nothing served yet")
        idle.note_op(100.0)
        self.assertFalse(idle.check(150.0, lambda: unloads.append("x")), "still in use")
        self.assertTrue(idle.check(161.0, lambda: unloads.append("x")))
        self.assertFalse(idle.check(500.0, lambda: unloads.append("x")), "once per use")
        idle.note_op(600.0)
        self.assertTrue(idle.check(700.0, lambda: unloads.append("x")))
        self.assertEqual(unloads, ["x", "x"])

    def test_every_op_on_this_machine_runs_one_at_a_time(self):
        self.assertIs(pom_components._lock_for(("img-1", "vae")), pom_components._lock_for(("img-2", "diffusion")))


if __name__ == "__main__":
    unittest.main()
