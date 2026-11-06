# ComfyUI

Open **ComfyUI** in the POM sidebar. On the first visit the installation starts by itself, behind POM's installer: a progress bar and a line saying what is happening (downloading ComfyUI, the installer, the Python runtime, then PyTorch and ComfyUI's packages). It downloads several gigabytes from GitHub, PyPI and the PyTorch index, and needs nothing on the host beforehand: no Python, pip, git or curl. When it finishes ComfyUI opens edge to edge. If it fails, the error is shown with **Try again**. Later visits open ComfyUI directly.

## Which PyTorch

The plugin picks the PyTorch build from the machine: CUDA 12.8 when an NVIDIA driver is present (this includes RTX 50xx cards), ROCm when only an AMD GPU is present, the default build on macOS and the CPU build otherwise. A machine whose GPU changes reinstalls the packages on its next opening.

## Your files

Models, custom nodes, inputs, outputs and saved workflows live in the POM workspace under `.comfyui/` (`models/checkpoints`, `models/diffusion_models`, `models/text_encoders`, `models/vae`, `models/loras`, `input`, `output`, `user`). Put model files there, or use the **Download** button ComfyUI shows for a workflow's missing models: the POM nodes take that button over, so the file downloads on the POM node into the right `models/` folder (Hugging Face and Civitai links), with the progress in the corner of the screen, instead of into your browser's downloads. An update of the plugin replaces only the runtime, never these folders.

## POM nodes

The **POM** category of the node library calls the models deployed on your POM cluster through this node's own gateway, server side; the key never reaches the browser or a workflow.

| Node | What it does |
|---|---|
| POM Chat (LLM) | asks a POM model anything; the answer is a text output |
| POM Prompt Enhancer | turns a short idea into a detailed text-to-image prompt |
| POM Describe Image | asks a multimodal POM model about an image (captions, tags) |

Connect their text output to a text input, such as the prompt of **CLIP Text Encode**. The model list is read from POM when ComfyUI loads; deploy a model in POM and refresh ComfyUI to see it. Change the node's seed to ask again instead of reusing the cached answer.

## Preferences

The gear on the plugin's card in **Plugins** has two toggles; a change restarts ComfyUI.

- **Low VRAM** - starts ComfyUI with `--lowvram`: weights stay in system RAM and move to the GPU per step. Use it when POM's own models hold most of the GPU memory.
- **CPU only** - installs the CPU build of PyTorch and runs everything on the CPU. Slow; for machines without a supported GPU.

## Access

ComfyUI has no accounts of its own: anyone who can open this plugin in POM can run workflows, install custom nodes from ComfyUI's manager and read the files under `.comfyui/`. The server listens on loopback only, behind a per-launch token that only POM's proxy holds. Closing ComfyUI from the menu stops it and everything it started.
