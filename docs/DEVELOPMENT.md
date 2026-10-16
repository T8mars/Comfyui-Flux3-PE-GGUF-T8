# Development

The production backend uses the official PUP revision `8b069751eb5dcab23829d9c5eccb470eb8b8a4ed`, converted with llama.cpp commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`. Model checksums and tensor metadata are published with the [GGUF artifacts](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/tree/main).

## Backend integration

The node currently starts an owned `llama-server` process. Its supplied and validated prebuilt runtime targets Windows x64 / NVIDIA CUDA 13.4. GGUF weights are platform-independent, and upstream [llama.cpp binaries](https://github.com/ggml-org/llama.cpp/releases) and [llama-cpp-python wheels](https://github.com/abetlen/llama-cpp-python#installation) offer other deployment options.

Those upstream packages are not currently drop-in replacements for this node's runtime. `ModelSpec.load` requires a matching `flux3-pe-runtime.json`, verifies the executable and patched Windows DLLs, and startup checks for the `penalties_generated_only` sampling extension. A Python-wheel backend would also need compatible PUP/Qwen3.5 multimodal inference and equivalent sampling behavior. Direct Python-wheel and Linux/macOS integration have not yet been implemented or validated.

## Test

Use Python 3.10+ with PyTorch, NumPy, pytest and the project requirements. Windows installer tests additionally require PowerShell 7. Run from the repository root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python scripts/run_checks.py
```

Real model validation additionally requires the GGUF files, the patched runtime and an NVIDIA GPU:

```powershell
python scripts/evaluate.py --models Q4_K_M --output artifacts/evaluation-q4.json
python scripts/lifecycle_smoke.py
```

The evaluation covers text, reference editing, text layout and reference layout, including ten references. It checks output structure, locked coordinates and sampled literal text. The known Chinese roof-edit recovery can describe the original brown roof color inaccurately while requesting the correct red target.

## Rebuild artifacts

1. `scripts/download_model.py` downloads the pinned source weights and verifies SHA256.
2. `scripts/build_runtime.ps1` builds the PE sampling extension using Visual Studio 2022 and CMake, with the matching official CUDA 13.4 backend.
3. `scripts/quantize.py --quants Q4_K_M` converts the original weights, creates the F16 vision projector and quantizes the text backbone.
4. `scripts/verify_artifacts.py` verifies the local artifacts.

Conversion requires the upstream llama.cpp conversion dependencies, Transformers, safetensors and Hugging Face Hub. Native sampling regression source is in `tests/native/`; the change is in `patches/llama-generated-penalties.patch`. Source preparation scripts are explicit developer commands, and do not run during node import.

## Benchmark scope

An eight-case comparison on an RTX 5090 Laptop 24 GB used the same llama.cpp backend:

| Metric | BF16 | Q4_K_M |
|---|---:|---:|
| Median decode speed | 38.8 token/s | 85.9 token/s |
| Median generation time | 7.02 s | 3.30 s |
| Whole-GPU memory peak | 17,495 MiB | 7,071 MiB |

Startup is excluded from generation time. Natural output lengths differ; GPU peaks include the host context. These results describe those eight cases and that hardware. Q4_0 failed a multi-reference layout case and is excluded from the production model repository.

## Registry publication

The package ID is `flux3-pe-gguf-t8`, and the publisher is `t8star`. `.github/workflows/publish.yml` uses the [official publishing action](https://github.com/Comfy-Org/publish-node-action) with the repository secret `REGISTRY_ACCESS_TOKEN`. Increment `pyproject.toml` and `flux3_pe/__init__.py` together for a new version; Registry versions are immutable.

The Registry archive excludes development scripts and tests through `.comfyignore`. Model files are hosted on Hugging Face, and the patched Windows runtime is a GitHub Release asset. Registry submission, security review and Manager indexing are separate stages; see the [official publishing guide](https://docs.comfy.org/registry/publishing).
