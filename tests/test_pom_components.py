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


if __name__ == "__main__":
    unittest.main()
