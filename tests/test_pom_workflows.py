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

    def test_unpublish_removes_only_the_published_copy(self):
        with tempfile.TemporaryDirectory() as home:
            graph = {"10": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}}}
            workflow_id = pom_workflows.publish(home, "My Sketch!", graph, {"prompt": "10"})
            result = pom_workflows.unpublish(home, workflow_id)
            self.assertEqual(result, {"id": workflow_id, "name": "My Sketch!"})
            self.assertEqual(pom_workflows.published(home), [])
            self.assertFalse(os.path.exists(os.path.join(home, "user", "pom-workflows", "my-sketch.json")))
            self.assertIsNone(pom_workflows.unpublish(home, workflow_id))
            with self.assertRaises(ValueError):
                pom_workflows.unpublish(home, "pub:../other")

    def test_publish_refuses_a_graph_without_the_prompt_node(self):
        with tempfile.TemporaryDirectory() as home:
            graph = {"11": {"class_type": "KSampler", "inputs": {"seed": 1}}}
            with self.assertRaises(ValueError):
                pom_workflows.publish(home, "x", graph, {"prompt": "11"})
            with self.assertRaises(ValueError):
                pom_workflows.publish(home, "", {"10": {"inputs": {"text": ""}}}, {"prompt": "10"})
            self.assertFalse(os.path.exists(os.path.join(home, "user", "pom-workflows")))


class BaseImageTests(unittest.TestCase):
    PNG = b"\x89PNG\r\n\x1a\n" + b"rest"

    def test_builtin_becomes_image_to_image(self):
        graph = pom_workflows.with_base_image(pom_workflows.builtin("img-1", ENTRIES))
        self.assertNotIn("6", graph)
        self.assertEqual(graph["10"]["class_type"], "LoadImage")
        self.assertEqual(graph["11"]["inputs"]["image"], ["10", 0])
        self.assertEqual(graph["12"]["inputs"], {"pixels": ["11", 0], "vae": ["3", 0]})
        self.assertEqual(graph["7"]["inputs"]["latent_image"], ["12", 0])
        self.assertEqual(graph["7"]["inputs"]["denoise"], 0.6)
        out = pom_workflows.substitute(graph, pom_workflows.BUILTIN_IMAGE_MAP,
                                       {"prompt": "x", "image_name": "pom-chat/a.png", "strength": 0.4})
        self.assertEqual(out["10"]["inputs"]["image"], "pom-chat/a.png")
        self.assertEqual(out["7"]["inputs"]["denoise"], 0.4)

    def test_the_base_image_is_saved_by_its_hash(self):
        import base64
        import hashlib
        with tempfile.TemporaryDirectory() as inputs:
            data = "data:image/png;base64," + base64.b64encode(self.PNG).decode()
            name = pom_workflows.save_base_image(data, inputs)
            self.assertEqual(name, f"pom-chat/{hashlib.sha256(self.PNG).hexdigest()}.png")
            self.assertTrue(os.path.isfile(os.path.join(inputs, name)))
            with self.assertRaises(ValueError):
                pom_workflows.save_base_image(base64.b64encode(b"GIF89a").decode(), inputs)
            with self.assertRaises(ValueError):
                pom_workflows.save_base_image("not base64!", inputs)

    def test_a_published_workflow_without_an_image_node_refuses_a_base_image(self):
        with tempfile.TemporaryDirectory() as home:
            graph = {"10": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}}
            pom_workflows.publish(home, "plain", graph, {"prompt": "10"})
            with self.assertRaises(ValueError):
                pom_workflows.resolve(home, "pub:plain", [], base_image=True)
            graph["20"] = {"class_type": "LoadImage", "inputs": {"image": ""}}
            pom_workflows.publish(home, "edit", graph, {"prompt": "10", "image": "20"})
            _, mapped = pom_workflows.resolve(home, "pub:edit", [], base_image=True)
            self.assertEqual(mapped["image"], "20")
            self.assertIn("image", pom_workflows.published(home)[0]["inputs"])


GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "model": ["4", 0]}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat"}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x", "upscale": False}},
    "20": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
}
PARAMS = [
    {"key": "prompt", "label": "Prompt", "node": "6", "input": "text", "type": "text"},
    {"key": "steps", "label": "Steps", "node": "3", "input": "steps", "type": "integer", "min": 1, "max": 50, "default": 20},
    {"key": "cfg", "label": "CFG", "node": "3", "input": "cfg", "type": "number", "min": 0, "max": 20},
    {"key": "sampler", "label": "Sampler", "node": "3", "input": "sampler_name", "type": "choice", "options": ["euler", "dpmpp_2m"]},
    {"key": "upscale", "label": "Upscale", "node": "9", "input": "upscale", "type": "boolean"},
    {"key": "photo", "label": "Photo", "node": "20", "input": "image", "type": "image"},
]


class ParamTests(unittest.TestCase):
    def test_publish_with_typed_params(self):
        with tempfile.TemporaryDirectory() as home:
            workflow_id = pom_workflows.publish(home, "Typed", GRAPH, PARAMS, "Paints a picture")
            record = pom_workflows.published(home)[0]
            self.assertEqual(workflow_id, "pub:typed")
            self.assertEqual(record["description"], "Paints a picture")
            self.assertEqual([p["key"] for p in pom_workflows.params_of(record)],
                             ["prompt", "steps", "cfg", "sampler", "upscale", "photo"])

    def test_publish_refuses_bad_params(self):
        with tempfile.TemporaryDirectory() as home:
            bad = [
                [{"key": "x", "label": "X", "node": "99", "input": "text", "type": "text"}],
                [{"key": "x", "label": "X", "node": "3", "input": "model", "type": "text"}],
                [{"key": "x", "label": "X", "node": "3", "input": "steps", "type": "weird"}],
                [{"key": "x", "label": "X", "node": "3", "input": "sampler_name", "type": "choice"}],
                [{"key": "x", "label": "X", "node": "6", "input": "text", "type": "image"}],
                [{"key": "x", "label": "X", "node": "6", "input": "text", "type": "text"}] * 2,
                [{"key": f"k{i}", "label": "X", "node": "6", "input": "text", "type": "text"} for i in range(25)],
            ]
            for params in bad:
                with self.assertRaises(ValueError, msg=params[0]):
                    pom_workflows.publish(home, "bad", GRAPH, params)

    def test_old_map_is_read_as_params(self):
        record = {"id": "pub:old", "name": "old", "inputs": ["prompt", "image"],
                  "map": {"prompt": "6", "image": "20"}, "graph": GRAPH}
        params = pom_workflows.params_of(record)
        self.assertEqual([(p["key"], p["node"], p["input"], p["type"]) for p in params],
                         [("prompt", "6", "text", "text"), ("image", "20", "image", "image")])
        out = pom_workflows.apply_params(GRAPH, params, {"prompt": "a dog"}, lambda data: "unused")
        self.assertEqual(out["6"]["inputs"]["text"], "a dog")

    def test_values_are_coerced(self):
        saved = []
        out = pom_workflows.apply_params(GRAPH, PARAMS, {
            "steps": 500, "cfg": "-3", "sampler": "dpmpp_2m", "upscale": "true", "photo": "data:..."},
            lambda data: saved.append(data) or "pom-chat/h.png")
        self.assertEqual(out["3"]["inputs"]["steps"], 50)
        self.assertEqual(out["3"]["inputs"]["cfg"], 0.0)
        self.assertEqual(out["3"]["inputs"]["sampler_name"], "dpmpp_2m")
        self.assertIs(out["9"]["inputs"]["upscale"], True)
        self.assertEqual(out["20"]["inputs"]["image"], "pom-chat/h.png")
        self.assertEqual(out["6"]["inputs"]["text"], "a cat")  # not given: the graph's value
        self.assertEqual(GRAPH["3"]["inputs"]["steps"], 20)  # the stored graph is untouched
        with self.assertRaises(ValueError):
            pom_workflows.apply_params(GRAPH, PARAMS, {"sampler": "ddim"}, lambda d: "")
        with self.assertRaises(ValueError):
            pom_workflows.apply_params(GRAPH, PARAMS, {"steps": "many"}, lambda d: "")
        with self.assertRaises(ValueError):
            pom_workflows.apply_params(GRAPH, PARAMS, {"upscale": "maybe"}, lambda d: "")

    def test_the_seed_is_random_unless_it_is_a_param(self):
        out = pom_workflows.apply_params(GRAPH, PARAMS, {}, lambda d: "")
        self.assertIsInstance(out["3"]["inputs"]["seed"], int)
        seeded = PARAMS + [{"key": "seed", "label": "Seed", "node": "3", "input": "seed", "type": "integer"}]
        self.assertEqual(pom_workflows.apply_params(GRAPH, seeded, {"seed": 7}, lambda d: "")["3"]["inputs"]["seed"], 7)


class MediaTests(unittest.TestCase):
    def test_kinds_by_magic(self):
        kind = pom_workflows.media_kind
        self.assertEqual(kind(b"\x89PNG\r\n\x1a\nxx"), ("image", "image/png"))
        self.assertEqual(kind(b"\xff\xd8\xffxx"), ("image", "image/jpeg"))
        self.assertEqual(kind(b"RIFF\0\0\0\0WEBPVP8 "), ("image", "image/webp"))
        self.assertEqual(kind(b"GIF89axx"), ("image", "image/gif"))
        self.assertEqual(kind(b"\0\0\0\x18ftypmp42"), ("video", "video/mp4"))
        self.assertEqual(kind(b"\x1aE\xdf\xa3xx"), ("video", "video/webm"))
        self.assertEqual(kind(b"RIFF\0\0\0\0WAVEfmt "), ("audio", "audio/wav"))
        self.assertEqual(kind(b"ID3\x04xx"), ("audio", "audio/mpeg"))
        self.assertEqual(kind(b"fLaCxx"), ("audio", "audio/flac"))
        self.assertEqual(kind(b"OggSxx"), ("audio", "audio/ogg"))
        self.assertIsNone(kind(b"plain text"))

    def test_outputs_are_collected(self):
        with tempfile.TemporaryDirectory() as base:
            os.makedirs(os.path.join(base, "sub"))
            files = {"a.png": b"\x89PNG\r\n\x1a\nxx", "sub/b.mp4": b"\0\0\0\x18ftypmp42",
                     "c.flac": b"fLaCxx", "d.txt": b"nope", "t.png": b"\x89PNG\r\n\x1a\nyy"}
            for name, data in files.items():
                Path(base, name).write_bytes(data)
            entry = {"outputs": {
                "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"},
                                 {"filename": "t.png", "subfolder": "", "type": "temp"}]},
                "10": {"gifs": [{"filename": "b.mp4", "subfolder": "sub", "type": "output"}]},
                "11": {"audio": [{"filename": "c.flac", "subfolder": "", "type": "output"},
                                 {"filename": "d.txt", "subfolder": "", "type": "output"},
                                 {"filename": "../../etc/passwd", "subfolder": "", "type": "output"}]},
            }}
            media = pom_workflows.collect_media(entry, base)
            self.assertEqual([(m["kind"], m["mime"]) for m in media],
                             [("image", "image/png"), ("video", "video/mp4"), ("audio", "audio/flac")])
            import base64
            self.assertEqual(base64.b64decode(media[2]["data"]), b"fLaCxx")


if __name__ == "__main__":
    unittest.main()
