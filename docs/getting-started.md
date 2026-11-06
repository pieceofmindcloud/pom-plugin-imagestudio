# ComfyUI

Open **ComfyUI** in the POM sidebar. On the first visit the installation starts by itself, behind POM's installer: a progress bar and a line saying what is happening (downloading ComfyUI, the installer, the Python runtime, then PyTorch and ComfyUI's packages). It downloads several gigabytes from GitHub, PyPI and the PyTorch index, and needs nothing on the host beforehand: no Python, pip, git or curl. When it finishes ComfyUI opens edge to edge. If it fails, the error is shown with **Try again**. Later visits open ComfyUI directly.

## Which PyTorch

The plugin picks the PyTorch build from the machine: CUDA 13.0 with an NVIDIA driver of the 580 series or newer, CUDA 12.8 with an older one (both include RTX 50xx cards), ROCm when only an AMD GPU is present, the default build on macOS and the CPU build otherwise. A machine whose GPU changes reinstalls the packages on its next opening.

## Your files

Models, custom nodes, inputs, outputs and saved workflows live in the POM workspace under `.comfyui/` (`models/checkpoints`, `models/diffusion_models`, `models/text_encoders`, `models/vae`, `models/loras`, `input`, `output`, `user`). Put model files there, or use the **Download** button ComfyUI shows for a workflow's missing models: the POM nodes take that button over, so the file downloads on the POM node (Hugging Face and Civitai links), with the progress in the corner of the screen, instead of into your browser's downloads. It lands in POM's own model folders (Settings > Storage), in the first one with room, as `<organization>/<repository>/<file>` (Civitai: `civitai/<version>/<file>`), so POM's Full models screen lists it under Image; the message at the end shows the full path. On a node older than this plugin the file goes to the `models/` folder of ComfyUI's workspace instead. Every image model POM's catalog finds on this machine, downloaded here or copied by hand, shows up in ComfyUI's loaders under `pom/` (Load Diffusion Model, Load CLIP, Load VAE, Load Checkpoint, LoRAs...), by the category POM read from the file, and also under its plain file name when no other file there has that name, so a template that asks for `ae.safetensors` finds it without changing the node; reopen ComfyUI to pick up a file copied while it was open. An update of the plugin replaces only the runtime, never these folders.

## POM nodes

The **POM** category of the node library calls the models deployed on your POM cluster through this node's own gateway, server side; the key never reaches the browser or a workflow.

| Node | What it does |
|---|---|
| POM Chat (LLM) | asks a POM model anything; the answer is a text output |
| POM Prompt Enhancer | turns a short idea into a detailed text-to-image prompt |
| POM Describe Image | asks a multimodal POM model about an image (captions, tags) |

Connect their text output to a text input, such as the prompt of **CLIP Text Encode**. The model list is read from POM when ComfyUI loads; deploy a model in POM and refresh ComfyUI to see it. Change the node's seed to ask again instead of reusing the cached answer.

## Models served on the POM network

An admin can deploy an image model on the network by component: the diffusion model, the text encoder and the VAE each on a machine that already holds that file and runs ComfyUI (POM's **Deploy** screen, **Image model**, or **Deploy on POM** in ComfyUI's menu on an admin node). Those machines' ComfyUI then serve their part; nobody has to keep their screen open, and a machine that restarts serves its part again by itself. Closing ComfyUI with its X on a serving machine stops what it serves there until it reopens. A serving machine gives its GPU back to POM's language models a minute after the last call; the next generation takes a few seconds more to load the part again.

On an admin node, ComfyUI's menu has **POM ▸ Deploy on POM**: it lists, per part, only the files of that kind on machines that run ComfyUI, deploys under the name you give, and shows the image models already served with an **Undeploy** button. The node checks the choices and its role again on every call, so the command does nothing on a node that is not an admin.

On any machine of the network, ComfyUI's ordinary loaders list them after the local files, as `POM ▸ <model> ▸ <file>`: **Load Diffusion Model**, **Load CLIP** and **Load VAE**. Pick them in a workflow (the Z-Image template, for one) and queue it: the sampler runs here, while each denoising step, the text encoding and the VAE decode run on the machine that serves that part. Only latents, conditionings and images cross the network (a few MB per step), through this node's POM connection; the weights stay where they are, so a machine without a GPU can use them.

Not yet supported on a network model: LoRAs and other model patches, ControlNet, text encoders given images, and Set CLIP Last Layer; each one stops the run with a clear message. If the serving machine leaves or its ComfyUI stops, the run stops with an error naming that machine.

Each generation that uses network models is recorded in POM's request log as one **Image** request (with the machines and times of each part) and counts against this machine's image budget, when the admins set one; a spent budget stops the run before any GPU work.

## Preferences

The gear on the plugin's card in **Plugins** has two toggles; a change restarts ComfyUI.

- **Low VRAM** - starts ComfyUI with `--lowvram`: weights stay in system RAM and move to the GPU per step. Use it when POM's own models hold most of the GPU memory.
- **CPU only** - installs the CPU build of PyTorch and runs everything on the CPU. Slow; for machines without a supported GPU.

## Access

ComfyUI has no accounts of its own: anyone who can open this plugin in POM can run workflows, install custom nodes from ComfyUI's manager and read the files under `.comfyui/`. The server listens on loopback only, behind a per-launch token that only POM's proxy holds. Closing ComfyUI from the menu stops it and everything it started.
