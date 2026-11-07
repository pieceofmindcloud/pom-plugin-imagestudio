"""Tests of runtime/pom_workflows.py (stdlib only; ComfyUI is not needed).

Run: python3 -m unittest discover -s tests -p 'test_*.py'
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("pom_workflows", ROOT / "runtime" / "pom_workflows.py")
pom_workflows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pom_workflows)

ENTRIES = [
    {"deploy_id": "img-1", "name": "Z", "component": "vae", "files": ["ae.safetensors"]},
    {"deploy_id": "img-1", "name": "Z", "component": "diffusion", "files": ["z_image_turbo_bf16.safetensors"]},
    {"deploy_id": "img-1", "name": "Z", "component": "text_encoder", "files": ["qwen_3_4b.safetensors"]},
    {"deploy_id": "img-2", "name": "Other", "component": "vae", "files": ["ae.safetensors"]},
]


class WorkflowTests(unittest.TestCase):
    def test_builtin_wires_network_components_in_flow_order(self):
        graph = pom_workflows.builtin("img-1", ENTRIES)
        self.assertEqual(graph["1"]["inputs"]["clip_name"], "POM ▸ Z ▸ qwen_3_4b.safetensors")
        self.assertEqual(graph["2"]["inputs"]["unet_name"], "POM ▸ Z ▸ z_image_turbo_bf16.safetensors")
        self.assertEqual(graph["3"]["inputs"]["vae_name"], "POM ▸ Z ▸ ae.safetensors")
        self.assertEqual(graph["7"]["inputs"]["model"], ["2", 0])
        self.assertEqual(graph["8"]["inputs"]["vae"], ["3", 0])
        self.assertEqual(graph["9"]["class_type"], "SaveImage")
        self.assertIsNone(pom_workflows.builtin("img-2", ENTRIES), "a deployment missing stages")

    def test_family_defaults(self):
        self.assertEqual(pom_workflows.family_of(["z_image_turbo_bf16.safetensors"]), "z_image")
        graph = pom_workflows.builtin("img-1", ENTRIES)
        self.assertEqual(graph["7"]["inputs"]["steps"], 8)
        self.assertEqual(graph["7"]["inputs"]["cfg"], 1.0)
        self.assertEqual(graph["1"]["inputs"]["type"], "lumina2")
        self.assertEqual(pom_workflows.defaults(None)["steps"], 20)

    def test_substitute_sets_prompt_and_seed(self):
        graph = pom_workflows.builtin("img-1", ENTRIES)
        out = pom_workflows.substitute(graph, pom_workflows.BUILTIN_MAP,
                                       {"prompt": "a red fox", "negative": "blur", "seed": 7, "width": 512, "count": 3})
        self.assertEqual(out["4"]["inputs"]["text"], "a red fox")
        self.assertEqual(out["5"]["inputs"]["text"], "blur")
        self.assertEqual(out["7"]["inputs"]["seed"], 7)
        self.assertEqual(out["6"]["inputs"]["width"], 512)
        self.assertEqual(out["6"]["inputs"]["height"], 1024)
        self.assertEqual(out["6"]["inputs"]["batch_size"], 3)
        self.assertEqual(graph["4"]["inputs"]["text"], "", "the original is untouched")
        random_seed = pom_workflows.substitute(graph, pom_workflows.BUILTIN_MAP, {"prompt": "x"})
        self.assertIsInstance(random_seed["7"]["inputs"]["seed"], int)

    def test_publish_and_list(self):
        with tempfile.TemporaryDirectory() as home:
            graph = {"10": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}}, "11": {"class_type": "KSampler", "inputs": {"seed": 1}}}
            workflow_id = pom_workflows.publish(home, "My Sketch!", graph, {"prompt": "10"})
            self.assertEqual(workflow_id, "pub:my-sketch")
            listed = pom_workflows.published(home)
            self.assertEqual([w["id"] for w in listed], ["pub:my-sketch"])
            self.assertEqual(listed[0]["inputs"], ["prompt"])
            resolved, inputs_map = pom_workflows.resolve(home, "pub:my-sketch", [])
            self.assertEqual(inputs_map, {"prompt": "10"})
            with self.assertRaises(ValueError):
                pom_workflows.resolve(home, "pub:nope", [])

    def test_publish_refuses_a_graph_without_the_prompt_node(self):
        with tempfile.TemporaryDirectory() as home:
            graph = {"11": {"class_type": "KSampler", "inputs": {"seed": 1}}}
            with self.assertRaises(ValueError):
                pom_workflows.publish(home, "x", graph, {"prompt": "11"})
            with self.assertRaises(ValueError):
                pom_workflows.publish(home, "", {"10": {"inputs": {"text": ""}}}, {"prompt": "10"})
            self.assertFalse(os.path.exists(os.path.join(home, "user", "pom-workflows")))


if __name__ == "__main__":
    unittest.main()
