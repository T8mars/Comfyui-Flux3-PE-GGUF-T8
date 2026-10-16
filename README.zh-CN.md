# FLUX 3 PE GGUF · T8

[English](README.md) | **简体中文**

基于 Black Forest Labs **FLUX 3 Image Klein PUP** 与 **Q4_K_M GGUF** 后端的 ComfyUI 本地提示词增强节点。将简短需求、参考图和布局框转换为 FLUX 3 工作流使用的详细提示词。

[模型下载](https://huggingface.co/t8star/Flux3-PE-GGUF-Comfy) · [运行库下载](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases) · [工作流预览](docs/images/workflow.png)

## 功能

- 支持纯文本增强、参考图编辑指令，以及两种布局增强模式。
- 最多 **10 张参考图**，支持明确的图片顺序与对象框锁定。
- 输出可连接文本编码器的 `STRING`，支持结果缓存和最多一次恢复。
- `keep_loaded=false` 时，在后续出图前释放 PE 显存。

## 安装

**当前提供并实测的预编译运行包**适用于 **Windows x64 / NVIDIA CUDA 13.4**，需要兼容的 NVIDIA 驱动，以及具备 V3 节点 API 的近期 ComfyUI。GGUF 模型文件本身可跨平台使用；这里的 Windows 要求指当前节点运行库。

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

4. 从 [Releases](https://github.com/T8mars/Comfyui-Flux3-PE-GGUF-T8/releases) 下载 `flux3-pe-runtime-win-x64-cuda13.4.zip`，解压到节点目录后重启 ComfyUI。运行程序应位于 `runtime/bin/llama-server.exe`。

模型文件不随 Git 或 Registry 节点包分发。增强过程在本机运行，节点执行不会上传提示词或参考图。

### llama.cpp 与 Python 轮子

- [llama.cpp 官方预编译包](https://github.com/ggml-org/llama.cpp/releases)：包含 `llama-server` 等独立程序，Windows 下程序名为 `llama-server.exe`。
- [llama-cpp-python GitHub 发布页](https://github.com/abetlen/llama-cpp-python/releases)及[安装说明](https://github.com/abetlen/llama-cpp-python#installation)：通过 Python 调用 llama.cpp，可在不启动独立服务程序的情况下推理。
- 官方轮子文件：[CPU](https://abetlen.github.io/llama-cpp-python/whl/cpu/llama-cpp-python/) · [CUDA 12.4](https://abetlen.github.io/llama-cpp-python/whl/cu124/llama-cpp-python/) · [CUDA 13.2](https://abetlen.github.io/llama-cpp-python/whl/cu132/llama-cpp-python/)。请匹配 Python 版本、操作系统、架构及 CUDA 要求；其他构建见官方安装说明。

**当前接入状态：**本节点启动独立的 llama.cpp 服务，并校验配套运行库清单。该运行库包含“仅对生成 token 施加惩罚”的 PE 采样扩展。上游通用程序包及 Python 轮子暂不能直接替换本节点的配套运行库；直接使用 Python 轮子的后端，以及 Linux/macOS 接入尚未实现或测试。

## 使用

1. 添加 **FLUX 3 PE Model (GGUF)**，选择 Q4_K_M 模型。
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

PE 核心经 40 条结构与受检文字回归、7 次真实画布运行、工作流保存重开及 PNG 工作流恢复验证。其中一条中文编辑使用了一次恢复。上述检查不代表任意提示词的语义均完全正确。

[开发、运行库构建及性能说明](docs/DEVELOPMENT.md)

## T8star 链接

[B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com)

[在线 AI 应用](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)

## 致谢与许可

节点集成：**T8star**。PUP 权重及参考代码来自 [Black Forest Labs](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)，采用 Apache-2.0；运行库基于 MIT 许可的 [llama.cpp](https://github.com/ggml-org/llama.cpp)，包含本仓库的采样补丁。CUDA 库遵循 NVIDIA 对应许可。详见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和 [licenses](licenses)。
