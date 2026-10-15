import json
from pathlib import Path

import numpy as np
from PIL import Image
import folder_paths
import comfy.model_management
from comfy_api.latest import ComfyExtension, io

from .flux3_pe.engine import ModelSpec, enhance
from .flux3_pe.protocol import MAX_REFERENCES

ROOT = Path(__file__).resolve().parent
MODEL_TYPE = io.Custom("FLUX3_PE_MODEL")
folder_paths.add_model_folder_path("flux3_pe", str(ROOT / "models/gguf"))
folder_paths.add_model_folder_path("flux3_pe", str(Path(folder_paths.models_dir) / "flux3_pe"))


class Flux3PELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        names = [name for name in folder_paths.get_filename_list("flux3_pe") if name.endswith(".gguf") and not Path(name).name.startswith("mmproj-") and ".partial." not in name]
        names.sort(key=lambda name: ("Q4_K_M" not in name, name))
        return io.Schema(
            node_id="Flux3PELoader",
            display_name="FLUX 3 PE Model (GGUF)",
            category="FLUX3/Prompt Enhancement",
            inputs=[
                io.Combo.Input("model_name", options=names),
                io.Int.Input("context_size", default=8192, min=4096, max=32768, step=1024, advanced=True),
            ],
            outputs=[MODEL_TYPE.Output(display_name="pe_model")],
        )

    @classmethod
    def execute(cls, model_name, context_size=8192):
        path = folder_paths.get_full_path_or_raise("flux3_pe", model_name)
        return io.NodeOutput(ModelSpec.load(path, context=context_size))

    @classmethod
    def fingerprint_inputs(cls, model_name, context_size=8192):
        path = folder_paths.get_full_path_or_raise("flux3_pe", model_name)
        return ModelSpec.load(path, context=context_size).revision


class Flux3PromptEnhance(io.ComfyNode):
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # Let enhance() own caching so cache settings, metadata and model release
        # still take effect when the same workflow is queued repeatedly.
        return float("nan")

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Flux3PromptEnhance",
            display_name="FLUX 3 Prompt Enhancement",
            category="FLUX3/Prompt Enhancement",
            description="Expand a request with the official FLUX 3 PUP. Reference images select editing mode; boxes select layout mode. Output connects to a text encoder.",
            inputs=[
                MODEL_TYPE.Input("pe_model"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False, default="A man walking with his dog, afternoon sun"),
                io.String.Input("boxes_json", multiline=True, default="", optional=True, tooltip="Optional normalized boxes: [{\"id\":\"obj_1\",\"desc\":\"a red balloon\",\"bbox\":[0.1,0.1,0.5,0.4]}]"),
                io.Autogrow.Input("images", template=io.Autogrow.TemplateNames(io.Image.Input("image"), names=[f"image_{i}" for i in range(1, 11)], min=0)),
                io.Boolean.Input("keep_loaded", default=False, advanced=True, tooltip="Keep PE in GPU memory for repeated enhancement. Disable before running an image model on a small GPU."),
                io.Boolean.Input("use_cache", default=True, advanced=True),
                io.Int.Input("max_tokens", default=3584, min=1, max=3584, advanced=True),
                io.Int.Input("canvas_width", default=0, min=0, max=16384, advanced=True),
                io.Int.Input("canvas_height", default=0, min=0, max=16384, advanced=True),
                io.Boolean.Input("retry_invalid_output", default=True, optional=True, advanced=True, tooltip="Retry once if the model loops or returns invalid layout bindings. The details output reports recovery."),
            ],
            outputs=[io.String.Output(display_name="enhanced_prompt"), io.String.Output(display_name="details_json")],
        )

    @classmethod
    def execute(cls, pe_model, prompt, boxes_json="", images=None, keep_loaded=False, use_cache=True, max_tokens=3584, canvas_width=0, canvas_height=0, retry_invalid_output=True):
        slots = []
        count = 0
        for name in sorted(images or {}, key=lambda name: int(name.rsplit("_", 1)[-1])):
            tensor = images[name]
            if tensor is None:
                continue
            shape = getattr(tensor, "shape", ())
            if len(shape) != 4 or any(size <= 0 for size in shape[:3]) or shape[-1] not in (1, 2, 3, 4):
                raise ValueError(f"{name} must be a non-empty BHWC image tensor with 1 to 4 channels")
            count += shape[0]
            slots.append(tensor)
        if count > MAX_REFERENCES:
            raise ValueError(f"PUP accepts at most {MAX_REFERENCES} reference images")

        def interrupt():
            comfy.model_management.throw_exception_if_processing_interrupted()

        interrupt()
        references = []
        for tensor in slots:
            for frame in tensor:
                interrupt()
                pixels = frame.detach().float().cpu().numpy()
                if not np.isfinite(pixels).all():
                    raise ValueError("Reference image contains non-finite pixels")
                pixels = np.clip(pixels * 255.0, 0, 255).round().astype(np.uint8)
                if pixels.shape[-1] == 1:
                    pixels = pixels[..., 0]
                references.append(Image.fromarray(pixels))
        canvas = (canvas_width, canvas_height) if canvas_width or canvas_height else None

        def before_load():
            # External CUDA allocations are not visible to Comfy's model manager.
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()

        caption, details = enhance(pe_model, prompt, references, boxes_json, canvas, max_tokens, keep_loaded, use_cache, interrupt, recover_invalid=retry_invalid_output, before_load=before_load)
        return io.NodeOutput(caption, json.dumps(details, ensure_ascii=False, indent=2))


class Flux3PEExtension(ComfyExtension):
    async def get_node_list(self):
        return [Flux3PELoader, Flux3PromptEnhance]
