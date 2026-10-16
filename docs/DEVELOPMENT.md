# Development

The production backend uses the official PUP revision `8b069751eb5dcab23829d9c5eccb470eb8b8a4ed`, converted with llama.cpp commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`. Model checksums and tensor metadata are published with the [GGUF artifacts](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/tree/main).

## Backend integration

Version 0.2.0 adds in-process inference using an installed [llama-cpp-python wheel](https://github.com/abetlen/llama-cpp-python#installation) (>=0.3.36, <0.4). `auto` prefers that backend; `llama-cpp-python` and `llama-server` select explicitly. Wheel inference does not require a server executable or runtime manifest. The original, owned-server backend retains its Windows x64 / NVIDIA CUDA 13.4 runtime and integrity checks.

The wheel backend renders the model's chat template with thinking disabled, evaluates multimodal chunks through MTMD while preserving its returned positions, and samples with a fresh native penalty/greedy chain. Only sampled output tokens enter penalty history, so this backend does not need the server sampling patch. Token decoding uses the C API's returned byte count rather than the wheel helper's padded buffer. Temporary native objects are freed on errors and cancellation; `keep_loaded=false` also closes the model and projector.

Version 0.2.1 fixes loading the CUDA wheel on clean Windows installations. The node loads the installed wheel's DLLs in dependency order using Windows' secure default search behavior and keeps the necessary directory handles alive; compatible CUDA libraries must still be installed. It also frees a partially initialized projector before its model, including the path where MTMD has not registered a cleanup callback. Unsupported wheel versions >=0.4 are rejected before native loading.

`python scripts/wheel_native_check.py` checks the installed wheel's MTMD interfaces and generated-only penalties without model downloads. CI runs this on Windows and Linux with the official CPU wheel. Real GPU/model validation uses the official 0.3.36 CUDA 13.2 Windows wheel and compatible CUDA libraries. Linux/macOS GPU inference has not been locally validated.

Manually dispatch the **Tests** workflow to additionally download the pinned production GGUF files and run six real Linux CPU tasks covering all four PE modes, Chinese text and ten references. `scripts/download_gguf.py` checks the pinned manifest, model and projector SHA256 values before exposing completed files.

The six Linux CPU tasks passed in [the real-model CI run](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/actions/runs/38888558362). Its nine runtime source files match the 0.2.1 core exactly. That run exposed a GPU-monitoring assumption in the evaluation utility; CPU evaluation now skips NVIDIA sampling, and missing drivers or unsupported metrics no longer raise in its monitoring thread. This utility change is separately tested on Windows and Linux without changing the tested PE runtime or corpus.

## Test

Use Python 3.10+ with PyTorch, NumPy, pytest and the project requirements. Windows installer tests additionally require PowerShell 7. Run from the repository root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python scripts/run_checks.py
```

Real model validation additionally requires the GGUF files, a compatible wheel (or the optional patched server runtime) and an NVIDIA GPU:

```powershell
python scripts/evaluate.py --backend llama-cpp-python --models Q4_K_M --output artifacts/evaluation-q4.json
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

The original server backend's eight-case comparison on an RTX 5090 Laptop 24 GB used the same llama.cpp backend for BF16 and Q4_K_M:

| Metric | BF16 | Q4_K_M |
|---|---:|---:|
| Median decode speed | 38.8 token/s | 85.9 token/s |
| Median generation time | 7.02 s | 3.30 s |
| Whole-GPU memory peak | 17,495 MiB | 7,071 MiB |

Startup is excluded from generation time. Natural output lengths differ; GPU peaks include the host context. These results describe those eight cases and that hardware. Q4_0 failed a multi-reference layout case and is excluded from the production model repository.

## Registry publication

The package ID is `flux3-pe-gguf-t8`, and the publisher is `t8star`. `.github/workflows/publish.yml` uses the [official publishing action](https://github.com/Comfy-Org/publish-node-action) with the repository secret `REGISTRY_ACCESS_TOKEN`. Increment `pyproject.toml` and `flux3_pe/__init__.py` together for a new version; Registry versions are immutable.

The Registry archive excludes development scripts and tests through `.comfyignore`. Model files are hosted on Hugging Face, and the patched Windows runtime is a GitHub Release asset. Registry submission, security review and Manager indexing are separate stages; see the [official publishing guide](https://docs.comfy.org/registry/publishing).
