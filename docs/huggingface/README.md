---
license: other
license_name: upstream-model-licenses
license_link: https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/blob/main/LICENSES.md
language:
  - en
  - zh
base_model: black-forest-labs/flux-3-image-klein-pup
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - comfyui
  - prompt-enhancement
  - flux3
  - qwen3.5
  - heretic
---

# FLUX 3 PE GGUF · T8

**English** | [简体中文](README.zh-CN.md)

Local **Prompt Enhancement** for ComfyUI using Q4_K_M versions of **official FLUX 3 Image Klein PUP** and **Heretic PUP**. Supports text, layout constraints and up to 10 reference images. Output is a detailed prompt for a text encoder; image generation uses separate diffusion, text-encoder and VAE weights.

[ComfyUI node](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8) · [v0.2.3 release](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases/tag/v0.2.3) · [Workflows](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/tree/main/examples)

## Downloads

Choose one variant and download its model, projector and manifest together. Each Q4 model is **5.63 GB**; each F16 projector is **0.92 GB**. Sizes use decimal GB.

| Variant | Q4_K_M model | F16 projector | Manifest | Destination in node folder |
|---|---|---|---|---|
| Official PUP | [Download](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/flux3-klein-pup-Q4_K_M.gguf) | [Download](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/mmproj-flux3-klein-pup-F16.gguf) | [Download](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/manifest.json) | `models/gguf/` |
| Heretic PUP | [Download](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/heretic/flux3-klein-pup-heretic-Q4_K_M.gguf) | [Download](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/heretic/mmproj-flux3-klein-pup-heretic-F16.gguf) | [Download](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/heretic/manifest.json) | `models/gguf/heretic/` |

Keep the manifests in their respective directories. [SHA256SUMS.txt](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/SHA256SUMS.txt) covers all four GGUF files. Q4_0 and BF16 are not distributed in this repository.

With the Hugging Face CLI installed, run from the node folder to download Heretic:

```bash
hf download t8star/Flux3-PE-GGUF-Comfy --include "heretic/*" --local-dir models/gguf
```

## ComfyUI

1. Install the [node](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8#installation), version **0.2.2+** for Heretic; v0.2.3 is the model-distribution and documentation update.
2. Install **llama-cpp-python >=0.3.36, <0.4** in the same Python that runs ComfyUI. [CPU wheels](https://abetlen.github.io/llama-cpp-python/whl/cpu/llama-cpp-python/) · [CUDA 12.4 wheels](https://abetlen.github.io/llama-cpp-python/whl/cu124/llama-cpp-python/) · [CUDA 13.2 wheels](https://abetlen.github.io/llama-cpp-python/whl/cu132/llama-cpp-python/). Match your Python, OS, architecture and hardware runtime.
3. Restart ComfyUI. Select the model in **FLUX 3 PE Model (GGUF)** and choose `auto` or `llama-cpp-python`.
4. Drag a [workflow JSON](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/tree/main/examples) into the canvas. For Heretic, start with [text enhancement](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/examples/05_heretic_prompt_enhancement.json) or [PE → FLUX 3](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/examples/07_heretic_pe_to_flux3.json). Reference workflows use the two images in `examples/reference_images/`; copy them to ComfyUI's input folder with the `FLUX3_PE_` filename prefix.

The wheel backend runs in Python without a server executable. GGUF files are platform-independent; acceleration depends on the chosen wheel and compatible runtime libraries. Keep `keep_loaded=false` to release PE memory before image generation. The [legacy server runtime](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases/tag/v0.1.5) remains optional.

## Provenance and validation

| Variant | Source | Pinned revision |
|---|---|---|
| Official PUP | [Black Forest Labs](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) | `8b069751eb5dcab23829d9c5eccb470eb8b8a4ed` |
| Heretic PUP | [tuanlai on ModelScope](https://modelscope.cn/models/tuanlai/flux-3-image-klein-pup-heretic) | `80efba65e01bb2e320dea412fa657d701491b443` |

Heretic GGUF was converted from the verified **BF16** source with llama.cpp `620d2225cc436ab0b001e3200e590e3b286ef9ab`, using pinned official PUP configuration and PE protocols. Source and artifact SHA256 values are recorded in each manifest.

The v0.2.2 Heretic runtime passed **481 unit tests, 40/40 real-model tasks, seven canvas executions, four workflow save/reopen checks and PNG restoration**, including a 512×512 PE-to-FLUX3 image and six official-model regressions. [Validation record](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/blob/main/validation/heretic-v0.2.2.json) · [Actual canvas](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/blob/v0.2.2/docs/images/heretic-workflow.png). Inference code and workflows are unchanged in v0.2.3.

Heretic model tests used Windows / RTX 5090 Laptop / llama-cpp-python 0.3.36 CUDA 13.2. The official PUP backend also passed [six Linux CPU model tasks](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/actions/runs/38888558362). Heretic inference on Linux/macOS was not tested. Structural and sampled text checks do not guarantee every prompt's semantics.

## Credits and licenses

GGUF conversion and node integration: **T8star**. Official PUP: **Black Forest Labs**, Apache-2.0. Heretic checkpoint: **tuanlai**, with its upstream license declarations retained separately. See [LICENSES.md](LICENSES.md), [NOTICE](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/blob/main/NOTICE) and [conversion details](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8#heretic-pup).

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com)

[Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
