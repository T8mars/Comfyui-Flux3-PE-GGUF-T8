# FLUX 3 PE GGUF · T8

[English](README.md) | **简体中文**

用于 ComfyUI 本地**提示词增强（PE）**的 Q4_K_M 模型，包含**官方 FLUX 3 Image Klein PUP** 与 **Heretic PUP**。支持文本、布局约束和最多 10 张参考图，输出可连接文本编码器的详细提示词；完整出图还需单独准备出图模型、文本编码器和 VAE。

[ComfyUI 节点](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8) · [v0.2.3 新版](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases/tag/v0.2.3) · [工作流](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/tree/main/examples)

## 下载

选择一个版本，将主模型、视觉文件和清单一起下载。每个 Q4 主模型约 **5.63 GB**，每个 F16 视觉文件约 **0.92 GB**，按十进制 GB 计。

| 版本 | Q4_K_M 主模型 | F16 视觉文件 | 清单 | 节点目录内的存放位置 |
|---|---|---|---|---|
| 官方 PUP | [下载](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/flux3-klein-pup-Q4_K_M.gguf) | [下载](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/mmproj-flux3-klein-pup-F16.gguf) | [下载](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/manifest.json) | `models/gguf/` |
| Heretic PUP | [下载](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/heretic/flux3-klein-pup-heretic-Q4_K_M.gguf) | [下载](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/heretic/mmproj-flux3-klein-pup-heretic-F16.gguf) | [下载](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/heretic/manifest.json) | `models/gguf/heretic/` |

两种模型的清单分目录保存。[SHA256SUMS.txt](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/SHA256SUMS.txt) 包含四个 GGUF 文件的校验值。本仓库不分发 Q4_0 和 BF16。

安装 Hugging Face CLI 后，可在节点目录执行以下命令下载 Heretic：

```bash
hf download t8star/Flux3-PE-GGUF-Comfy --include "heretic/*" --local-dir models/gguf
```

## ComfyUI 使用

1. 按[中文安装说明](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/blob/main/README.zh-CN.md#安装)安装节点。Heretic 需要 **0.2.2 及以上版本**；v0.2.3 完善了模型分发与文档。
2. 在运行 ComfyUI 的同一个 Python 中安装 **llama-cpp-python >=0.3.36、<0.4**。[CPU 轮子](https://abetlen.github.io/llama-cpp-python/whl/cpu/llama-cpp-python/) · [CUDA 12.4 轮子](https://abetlen.github.io/llama-cpp-python/whl/cu124/llama-cpp-python/) · [CUDA 13.2 轮子](https://abetlen.github.io/llama-cpp-python/whl/cu132/llama-cpp-python/)。请匹配 Python、系统、架构和硬件运行库。
3. 重启 ComfyUI，在 **FLUX 3 PE Model (GGUF)** 中选择模型，后端选择 `auto` 或 `llama-cpp-python`。
4. 将[工作流 JSON](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/tree/main/examples) 拖入画布。Heretic 可先用[纯文本增强](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/examples/05_heretic_prompt_enhancement.json)或 [PE → FLUX 3 完整出图](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/resolve/main/examples/07_heretic_pe_to_flux3.json)。参考图工作流使用 `examples/reference_images/` 内的两张图片，复制到 ComfyUI 的 input 目录时，为文件名加上 `FLUX3_PE_` 前缀。

轮子后端在 Python 内推理，无需独立服务程序。GGUF 文件可跨平台，实际加速能力取决于轮子与兼容运行库。后续需要出图时保持 `keep_loaded=false`，释放 PE 显存。[旧版服务运行包](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases/tag/v0.1.5)仍可选用。

## 来源与验证

| 版本 | 来源 | 固定版本 |
|---|---|---|
| 官方 PUP | [Black Forest Labs](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) | `8b069751eb5dcab23829d9c5eccb470eb8b8a4ed` |
| Heretic PUP | [ModelScope / tuanlai](https://modelscope.cn/models/tuanlai/flux-3-image-klein-pup-heretic) | `80efba65e01bb2e320dea412fa657d701491b443` |

Heretic 从校验通过的 **BF16** 源权重转换，使用 llama.cpp `620d2225cc436ab0b001e3200e590e3b286ef9ab`，配合官方固定版本的配置与 PE 协议。两套清单分别记录来源及文件 SHA256。

v0.2.2 Heretic 已通过 **481 项单元测试、40/40 条真实模型任务、7 次画布运行、四套工作流保存重开及 PNG 恢复**，包含 512×512 完整出图及官方模型 6 条回归。[验证记录](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/blob/main/validation/heretic-v0.2.2.json) · [实际画布](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/blob/v0.2.2/docs/images/heretic-workflow.png)。v0.2.3 的推理代码及工作流保持一致。

Heretic 实测使用 Windows / RTX 5090 Laptop / llama-cpp-python 0.3.36 CUDA 13.2。官方 PUP 后端还通过了 [Linux CPU 的 6 条真实模型任务](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/actions/runs/38888558362)；Heretic 未在 Linux/macOS 做推理实测。结构与受检文字检查不代表任意提示词的语义均完全正确。

## 致谢与许可

GGUF 转换及节点集成：**T8star**。官方 PUP：**Black Forest Labs**，Apache-2.0。Heretic 权重：**tuanlai**，单独保留其上游许可声明，见 [LICENSES.md](LICENSES.md)、[NOTICE](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/blob/main/NOTICE) 及[转换说明](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/blob/main/README.zh-CN.md#heretic-pup)。

## T8star

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com)

[在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)
