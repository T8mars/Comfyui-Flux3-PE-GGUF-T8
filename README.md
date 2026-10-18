# FLUX 3 PE GGUF · T8

**English** | [简体中文](README.zh-CN.md)

Local prompt enhancement for ComfyUI, powered by Black Forest Labs' **FLUX 3 Image Klein PUP** and a **Q4_K_M GGUF** backend. Turn short requests, reference images and layout boxes into detailed prompts for FLUX 3 workflows.

[Models](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy) · [Runtime downloads](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases) · [Workflow preview](docs/images/workflow.png)

## Features

- Text enhancement, reference-image editing instructions and both layout modes.
- Up to **10 reference images**, with explicit image order and locked object boxes.
- `STRING` output for text encoders, configurable result caching and a maximum of one recovery attempt.
- Direct inference with an installed **llama-cpp-python wheel**, including reference images; no separate server executable is needed for this backend.
- PE memory release before image generation with `keep_loaded=false`.

## Installation

Requires Python 3.10+ and a recent ComfyUI with the V3 node API. The Python-wheel backend uses the installed wheel and its platform-specific dependencies. The optional prebuilt server runtime targets Windows x64 / NVIDIA CUDA 13.4.

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

4. Install a compatible **llama-cpp-python >=0.3.36, <0.4** wheel into the **same Python that runs ComfyUI**, then restart ComfyUI. An existing compatible installation can be reused. For example, with CUDA 13.2 runtime libraries and a compatible driver:

   ```bash
   python -m pip install --upgrade --only-binary=llama-cpp-python llama-cpp-python==0.3.36 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu132
   ```

   Select the wheel matching your system below. The CPU index can be used without CUDA. GPU wheels require compatible CUDA libraries and drivers.

   On Windows, set `CUDA_PATH` to the compatible CUDA installation if its runtime libraries are not otherwise available. Version **0.2.1** loads the wheel's DLL dependencies directly; no external preload script is required.

   To use the original **server backend**, download `flux3-pe-runtime-win-x64-cuda13.4.zip` from [v0.1.5 runtime downloads](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases/tag/v0.1.5), extract it into the node folder, and select `llama-server`. Its executable should be at `runtime/bin/llama-server.exe`.

Model files stay outside Git and the Registry package. Enhancement runs locally; node execution does not upload prompts or reference images.

### llama.cpp and Python wheels

- [llama.cpp official binaries](https://github.com/ggml-org/llama.cpp/releases): standalone programs, including `llama-server` (`llama-server.exe` on Windows).
- [llama-cpp-python GitHub releases](https://github.com/abetlen/llama-cpp-python/releases) and [installation guide](https://github.com/abetlen/llama-cpp-python#installation): Python bindings that can run inference without a separate server executable.
- Official wheel files: [CPU](https://abetlen.github.io/llama-cpp-python/whl/cpu/llama-cpp-python/) · [CUDA 12.4](https://abetlen.github.io/llama-cpp-python/whl/cu124/llama-cpp-python/) · [CUDA 13.2](https://abetlen.github.io/llama-cpp-python/whl/cu132/llama-cpp-python/). Select a wheel matching your Python version, OS, architecture and CUDA requirements; see the installation guide for other builds.

**Backend selection:** `auto` prefers a compatible installed Python wheel and otherwise uses the packaged server if available. Select `llama-cpp-python` to require in-process inference, or `llama-server` for the original server backend. The wheel backend uses MTMD for reference images and a fresh native sampler to apply penalties only to generated tokens; it needs neither `llama-server.exe` nor `flux3-pe-runtime.json`. Python-wheel availability and acceleration depend on your OS and hardware; local GPU model validation used Windows x64 / CUDA 13.2, not Linux/macOS GPUs.

## Usage

1. Add **FLUX 3 PE Model (GGUF)**, select the Q4_K_M model, and leave `backend=auto` or select `llama-cpp-python`.
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

To copy the example workflows and their reference images into your installation, run from the node folder:

```powershell
& .\scripts\install_comfy.ps1 -ComfyUI 'C:\path\to\ComfyUI'
```

The image-generation example additionally requires FLUX 3 diffusion weights, its Qwen3-VL text encoder and FLUX.2 VAE. These are separate downloads.

### Heretic PUP

Version 0.2.2 supports the pinned [tuanlai/flux-3-image-klein-pup-heretic](https://modelscope.cn/models/tuanlai/flux-3-image-klein-pup-heretic) checkpoint. It uses the same four PE protocols, keeps separate model artifacts and reports its actual source in the details output.

Place `flux3-klein-pup-heretic-Q4_K_M.gguf`, `mmproj-flux3-klein-pup-heretic-F16.gguf` and their `manifest.json` together in `models/gguf/heretic/` or `ComfyUI/models/flux3_pe/heretic/`. Keep the official model's manifest separate. Select `heretic/flux3-klein-pup-heretic-Q4_K_M.gguf` in the loader.

- [Heretic text enhancement](examples/05_heretic_prompt_enhancement.json)
- [Heretic layout enhancement](examples/06_heretic_layout_enhancement.json)
- [Heretic PE → FLUX 3 image generation](examples/07_heretic_pe_to_flux3.json)
- [Heretic two-reference editing](examples/08_heretic_reference_editing.json)

The source repository provides BF16 and INT8 safetensors. For GGUF conversion, use its **BF16** checkpoint as `model.safetensors`, together with the pinned official PUP configuration, tokenizer and image-processing files. Prepare the llama.cpp conversion tools using the [development guide](docs/DEVELOPMENT.md), then run this from a source checkout. The converter verifies the source SHA256:

```bash
python scripts/quantize.py --variant heretic --source "path/to/heretic-source" --output models/gguf/heretic --quants Q4_K_M
```

Pinned source-file revision: `80efba65e01bb2e320dea412fa657d701491b443`. Protocol and configuration source: official PUP `8b069751eb5dcab23829d9c5eccb470eb8b8a4ed`. See the source repository for the Heretic weights' license information.

Heretic Q4_K_M passed **40/40 real-model tasks, seven canvas executions, four workflow save/reopen checks and PNG workflow restoration**, alongside 481 unit tests and six official-model regressions. The real-model checks used Windows / RTX 5090 Laptop / the Python-wheel backend. [Validation record](docs/heretic-validation.json) · [Actual canvas preview](docs/images/heretic-workflow.png)

## Validation and development

The 0.2.1 Python-wheel backend passed **470 unit tests**, **40 real model structural/text-preservation cases in a clean Windows environment**, and **six real Linux CPU tasks** covering all four PE modes, Chinese text and ten references. Seven real canvas executions, workflow save/reopen checks and PNG workflow restoration also passed. No external native-library preload script was used for these model or canvas checks. One Windows Chinese editing case used the bounded recovery path. These checks do not guarantee semantic correctness for every prompt.

[Development, runtime build and benchmark notes](docs/DEVELOPMENT.md)

## T8star links

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com)

[Online AI apps](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)

## Credits and license

Node integration: **T8star**. PUP weights and reference code: [Black Forest Labs](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup), Apache-2.0. Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT; the optional server runtime includes the sampling patch. CUDA libraries retain NVIDIA's license terms. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [licenses](licenses).
