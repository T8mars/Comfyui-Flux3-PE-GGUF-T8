from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux3_pe.engine import ModelSpec, enhance, release_model


def same_file(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def output_metadata_path(output: Path, inputs: list[Path]) -> Path:
    candidate = output.with_suffix(".json")
    index = 1
    while any(same_file(candidate, protected) for protected in [output, *inputs]):
        suffix = ".metadata.json" if index == 1 else f".metadata-{index}.json"
        candidate = output.with_name(output.name + suffix)
        index += 1
    return candidate


def main():
    parser = argparse.ArgumentParser(description="Local FLUX 3 prompt enhancement")
    parser.add_argument("prompt")
    parser.add_argument("--model", type=Path, default=ROOT / "models/gguf/flux3-klein-pup-Q4_K_M.gguf")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--boxes", type=Path)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true", help="Use only the original decoding policy; disable recovery")
    args = parser.parse_args()
    inputs = [args.model, *args.image, *([args.boxes] if args.boxes else [])]
    metadata_path = None
    if args.output:
        if any(same_file(args.output, path) for path in inputs):
            parser.error("--output must not overwrite a model, reference image or boxes input")
        metadata_path = output_metadata_path(args.output, inputs)
    images = []
    for path in args.image:
        with Image.open(path) as image:
            images.append(image.copy())
    boxes = args.boxes.read_text(encoding="utf-8") if args.boxes else None
    spec = ModelSpec.load(args.model, context=args.context)
    inputs.append(Path(getattr(spec, "model", args.model)).parent / "manifest.json")
    for name in ("mmproj", "server"):
        path = getattr(spec, name, None)
        if path is not None:
            inputs.append(Path(path))
    server = getattr(spec, "server", None)
    if server is not None:
        runtime_manifest = Path(server).parent / "flux3-pe-runtime.json"
        inputs.append(runtime_manifest)
        if runtime_manifest.is_file():
            runtime = json.loads(runtime_manifest.read_text(encoding="utf-8"))
            inputs.extend(runtime_manifest.parent / name for name in runtime.get("core_sha256", {}))
    if args.output:
        if any(same_file(args.output, path) for path in inputs):
            parser.error("--output must not overwrite a model, projector, runtime, manifest or reference input")
        metadata_path = output_metadata_path(args.output, inputs)
    try:
        caption, metadata = enhance(spec, args.prompt, images, boxes, max_tokens=args.max_tokens, use_cache=False, recover_invalid=not args.strict)
    finally:
        release_model()
    print(caption)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(caption + "\n", encoding="utf-8")
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
