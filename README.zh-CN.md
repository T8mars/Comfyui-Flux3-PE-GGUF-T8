# FLUX 3 PE GGUF · T8

[English](README.md) | **简体中文**

基于 Black Forest Labs **FLUX 3 Image Klein PUP** 与 **Q4_K_M GGUF** 后端的 ComfyUI 本地提示词增强节点。将简短需求、参考图和布局框转换为 FLUX 3 工作流使用的详细提示词。

[模型下载](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy) · [运行库下载](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases) · [工作流预览](docs/images/workflow.png)

## 功能

- 支持纯文本增强、参考图编辑指令，以及两种布局增强模式。
- 最多 **10 张参考图**，支持明确的图片顺序与对象框锁定。
- 输出可连接文本编码器的 `STRING`，支持结果缓存和最多一次恢复。
- 直接调用已安装的 **llama-cpp-python 轮子**，支持参考图；此后端无需独立服务程序。
- `keep_loaded=false` 时，在后续出图前释放 PE 显存。

## 安装

需要 Python 3.10+ 和具备 V3 节点 API 的近期 ComfyUI。Python 轮子后端使用已安装的轮子及其对应平台依赖；可选的预编译服务运行包适用于 Windows x64 / NVIDIA CUDA 13.4。

1. Registry 版本可用后，可在 ComfyUI 节点管理器搜索 **FLUX 3 PE GGUF (T8)**；也可进入 `ComfyUI/custom_nodes` 手动安装：

   ```bash
   git clone https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8.git ComfyUI-Flux3-PE
   ```

2. 使用 **ComfyUI 自带的 Python** 安装 `requirements.txt`。
3. 从[模型仓库](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy/tree/main)下载以下文件，放入节点目录的 `models/gguf/`：

   | 文件 | 大小 |
   |---|---:|
   | `flux3-klein-pup-Q4_K_M.gguf` | 5.63 GB |
   | `mmproj-flux3-klein-pup-F16.gguf` | 0.92 GB |
   | `manifest.json` | 少量元数据 |

4. 在**运行 ComfyUI 的同一个 Python** 中安装兼容的 **llama-cpp-python >=0.3.36、<0.4** 轮子，然后重启 ComfyUI；已有兼容版本可直接复用。以已具备 CUDA 13.2 运行库和兼容驱动的环境为例：

   ```bash
   python -m pip install --upgrade --only-binary=llama-cpp-python llama-cpp-python==0.3.36 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu132
   ```

   请从下方选择匹配本机环境的轮子；CPU 轮子无需 CUDA，GPU 轮子需要兼容的 CUDA 库和驱动。

   Windows 下如无法找到 CUDA 运行库，请将 `CUDA_PATH` 指向兼容的 CUDA 安装目录。**0.2.1** 已在节点内处理轮子 DLL 的依赖加载，无需额外预加载脚本。

   如需使用原有**服务后端**，从 [v0.1.5 运行库下载页](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases/tag/v0.1.5) 下载 `flux3-pe-runtime-win-x64-cuda13.4.zip`，解压到节点目录，选择 `llama-server`。其程序应位于 `runtime/bin/llama-server.exe`。

模型文件不随 Git 或 Registry 节点包分发。增强过程在本机运行，节点执行不会上传提示词或参考图。

### llama.cpp 与 Python 轮子

- [llama.cpp 官方预编译包](https://github.com/ggml-org/llama.cpp/releases)：包含 `llama-server` 等独立程序，Windows 下程序名为 `llama-server.exe`。
- [llama-cpp-python GitHub 发布页](https://github.com/abetlen/llama-cpp-python/releases)及[安装说明](https://github.com/abetlen/llama-cpp-python#installation)：通过 Python 调用 llama.cpp，可在不启动独立服务程序的情况下推理。
- 官方轮子文件：[CPU](https://abetlen.github.io/llama-cpp-python/whl/cpu/llama-cpp-python/) · [CUDA 12.4](https://abetlen.github.io/llama-cpp-python/whl/cu124/llama-cpp-python/) · [CUDA 13.2](https://abetlen.github.io/llama-cpp-python/whl/cu132/llama-cpp-python/)。请匹配 Python 版本、操作系统、架构及 CUDA 要求；其他构建见官方安装说明。

**后端选择：**`auto` 优先使用已安装的兼容 Python 轮子，没有兼容轮子时尝试配套服务运行库。选择 `llama-cpp-python` 可强制在 Python 内推理；`llama-server` 使用原服务后端。轮子后端通过 MTMD 处理参考图，每次生成创建新的原生采样器，仅对生成 token 施加惩罚，无需 `llama-server.exe` 或 `flux3-pe-runtime.json`。可用平台和加速能力取决于轮子及本机环境；本地 GPU 模型实测环境为 Windows x64 / CUDA 13.2，未在 Linux/macOS GPU 上实测。

## 使用

1. 添加 **FLUX 3 PE Model (GGUF)**，选择 Q4_K_M 模型，`backend` 保持 `auto` 或选择 `llama-cpp-python`。
2. 连接 **FLUX 3 Prompt Enhancement** 并填写需求；使用参考图时，按顺序连接图片。
3. 将 `enhanced_prompt` 连接文本编码器，或使用 ComfyUI 内置 **Preview as Text** 查看结果。`details_json` 包含耗时、缓存和恢复信息。

后续需要出图时保持 `keep_loaded=false`；需要重新增强时关闭 `use_cache`。

布局框采用归一化的 `[y_min, x_min, y_max, x_max]` 坐标：

```json
[{"id":"balloon","desc":"a red balloon","bbox":[0.1,0.1,0.6,0.4]}]
```

参考图布局使用 `from="ref_image_0"`、`src_bbox` 和 `tgt_bbox`。新对象使用 `from="target"`、`src_bbox=null`；删除对象使用 `tgt_bbox=null`。

## 示例工作流

将 [examples](examples) 中的 JSON 拖入 ComfyUI：

- [纯文本增强](examples/01_prompt_enhancement.json)
- [布局增强](examples/02_layout_enhancement.json)
- [PE → FLUX 3 完整出图](examples/03_pe_to_flux3.json)
- [双参考图编辑指令](examples/04_reference_editing.json)

在节点目录执行以下命令，可将四套画布和参考图复制到对应 ComfyUI：

```powershell
& .\scripts\install_comfy.ps1 -ComfyUI 'C:\你的路径\ComfyUI'
```

完整出图示例还需另行准备 FLUX 3 出图模型、对应的 Qwen3-VL 文本编码器和 FLUX.2 VAE。

## 验证与开发

0.2.1 Python 轮子后端已通过 **470 项单元测试**、**全新 Windows 环境的 40 条真实模型结构与受检文字回归**，以及 **Linux CPU 的 6 条真实模型任务**，覆盖四种 PE 模式、中文和十张参考图。7 次真实画布运行、工作流保存重开及 PNG 工作流恢复也已通过。上述模型和画布测试均未使用外部原生库预加载脚本。其中一条 Windows 中文编辑使用了一次恢复。上述检查不代表任意提示词的语义均完全正确。

[开发、运行库构建及性能说明](docs/DEVELOPMENT.md)

## T8star 链接

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com)

[在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)

## 致谢与许可

节点集成：**T8star**。PUP 权重及参考代码来自 [Black Forest Labs](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)，采用 Apache-2.0；推理基于 MIT 许可的 [llama.cpp](https://github.com/ggml-org/llama.cpp)，可选的服务运行库包含本仓库的采样补丁。CUDA 库遵循 NVIDIA 对应许可。详见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [licenses](licenses)。
