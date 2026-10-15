# FLUX 3 PE GGUF · T8

**English** | [简体中文](README.zh-CN.md)

Local prompt enhancement for ComfyUI, powered by Black Forest Labs' **FLUX 3 Image Klein PUP** and a **Q4_K_M GGUF** backend. Turn short requests, reference images and layout boxes into detailed prompts for FLUX 3 workflows.

[Models](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy) · [Runtime downloads](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases) · [Workflow preview](docs/images/workflow.png)

## Features

- Text enhancement, reference-image editing instructions and both layout modes.
- Up to **10 reference images**, with explicit image order and locked object boxes.
- `STRING` output for text encoders, configurable result caching and a maximum of one recovery attempt.
- PE memory release before image generation with `keep_loaded=false`.

## Installation

Requires **Windows x64**, an **NVIDIA GPU**, a driver supporting **CUDA 13.4**, and a recent ComfyUI with the V3 node API. The distributed runtime includes the PE sampling extension.

1. Install **FLUX 3 PE GGUF (T8)** through ComfyUI Manager when the Registry version is available, or clone into `ComfyUI/custom_nodes`:

   ```bash
   git clone https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8.git ComfyUI-Flux3-PE
   ```

2. Use your **ComfyUI Python** to install `requirements.txt`.
3. Download these files from the [model repository](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/tree/main) into the node folder's `models/gguf/`:

   | File | Size |
   |---|---:|
   | `flux3-klein-pup-Q4_K_M.gguf` | 5.63 GB |
   | `mmproj-flux3-klein-pup-F16.gguf` | 0.92 GB |
   | `manifest.json` | Small metadata file |

4. Download `flux3-pe-runtime-win-x64-cuda13.4.zip` from [Releases](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases), extract it into the node folder, and restart ComfyUI. The executable should be at `runtime/bin/llama-server.exe`.

Model files stay outside Git and the Registry package. Enhancement runs locally; node execution does not upload prompts or reference images.

## Usage

1. Add **FLUX 3 PE Model (GGUF)** and select the Q4_K_M model.
2. Connect it to **FLUX 3 Prompt Enhancement** and enter a prompt. Connect images in reference order when needed.
3. Connect `enhanced_prompt` to a text encoder or ComfyUI's **Preview as Text**. `details_json` reports timing, caching and recovery.

Leave `keep_loaded=false` when the next step generates an image. Disable `use_cache` to force a new enhancement.

Layout input uses normalized `[y_min, x_min, y_max, x_max]` coordinates:

```json
[{"id":"balloon","desc":"a red balloon","bbox":[0.1,0.1,0.6,0.4]}]
```

For reference-image layouts, use `from="ref_image_0"`, `src_bbox` and `tgt_bbox`. New objects use `from="target"` and `src_bbox=null`; removed objects use `tgt_bbox=null`.

## Example workflows

Drag a JSON file from [examples](examples) into ComfyUI:

- [Text enhancement](examples/01_prompt_enhancement.json)
- [Layout enhancement](examples/02_layout_enhancement.json)
- [PE → FLUX 3 image generation](examples/03_pe_to_flux3.json)
- [Two-reference editing instructions](examples/04_reference_editing.json)

To copy all four workflows and their reference images into your installation, run from the node folder:

```powershell
& .\scripts\install_comfy.ps1 -ComfyUI 'C:\path\to\ComfyUI'
```

The image-generation example additionally requires FLUX 3 diffusion weights, its Qwen3-VL text encoder and FLUX.2 VAE. These are separate downloads.

## Validation and development

The PE core passed 40 structural/text-preservation cases, seven real canvas executions, workflow save/reopen checks and PNG workflow restoration. One Chinese editing case used the bounded recovery path. These checks do not guarantee semantic correctness for every prompt.

[Development, runtime build and benchmark notes](docs/DEVELOPMENT.md)

## T8star links

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com)

[Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)

## Credits and license

Node integration: **T8star**. PUP weights and reference code: [Black Forest Labs](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup), Apache-2.0. Runtime: [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT, with the included sampling patch. CUDA libraries retain NVIDIA's license terms. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [licenses](licenses).
