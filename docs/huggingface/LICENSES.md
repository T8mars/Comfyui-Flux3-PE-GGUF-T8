# Model license declarations / 模型许可声明

The ComfyUI node source uses Apache-2.0. GGUF conversion does not relicense upstream weights.

| Artifact | Upstream declaration |
|---|---|
| Official FLUX 3 Image Klein PUP | [Black Forest Labs](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup), Apache-2.0; the repository's `LICENSE` and `licenses/BFL-Apache-2.0.txt` retain this license. |
| Heretic PUP checkpoint | [tuanlai model card](https://modelscope.cn/models/tuanlai/flux-3-image-klein-pup-heretic), README revision `1a2af6a21559e319675875fefd6d41b9c9778d9d`, declares `license: other` and `license_name: flux-community-license-v1.0`. Its prose also describes the prompt upsampler as Apache-2.0. These upstream declarations are retained without resolving their ambiguity or assigning a new license. Consult the source for applicable terms. |
| Separately installed inference libraries | llama.cpp and llama-cpp-python, MIT. |

The Hub model card uses `license: other` for this collection because its two variants carry different upstream declarations. The official PUP artifacts retain Apache-2.0.

节点源码采用 Apache-2.0。官方 PUP 权重保留 Apache-2.0；Heretic 源模型卡的元数据声明 `other / flux-community-license-v1.0`，正文同时称提示词增强模型采用 Apache-2.0。本仓库如实保留该差异，不为上游权重重新指定许可；Heretic 的适用条款请查阅源仓库。GGUF 转换不会改变权重的上游许可。
