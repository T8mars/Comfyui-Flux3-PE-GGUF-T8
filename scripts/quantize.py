"""Convert the original PUP and create traceable GGUF artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/llama.cpp/gguf-py"))
from gguf import GGUFReader, LlamaFileType
from flux3_pe.engine import sha256_file
from flux3_pe.protocol import SOURCE_MODEL, SOURCE_REVISION


def metadata(path, expected_type=None):
    reader = GGUFReader(path)
    if expected_type is not None:
        file_type = reader.fields.get("general.file_type")
        if file_type is None or file_type.contents() != LlamaFileType["MOSTLY_" + expected_type]:
            raise RuntimeError(f"GGUF file type does not match requested quantization {expected_type}: {path}")
    arch = reader.fields["general.architecture"].contents()
    if arch != "qwen35":
        raise RuntimeError(f"Unexpected GGUF architecture: {arch}")
    if reader.fields["qwen35.block_count"].contents() != 32:
        raise RuntimeError("Converted PUP must have exactly 32 layers")
    names = {tensor.name for tensor in reader.tensors}
    for name in ("token_embd.weight", "output.weight", "blk.31.ffn_down.weight"):
        if name not in names:
            raise RuntimeError(f"Required converted tensor missing: {name}")
    embedding = next(tensor for tensor in reader.tensors if tensor.name == "token_embd.weight")
    if sorted(int(x) for x in embedding.shape) != [4096, 248320]:
        raise RuntimeError("Converted tokenizer/embedding dimensions are incorrect")
    quant_types = {}
    for tensor in reader.tensors:
        kind = tensor.tensor_type.name
        quant_types[kind] = quant_types.get(kind, 0) + 1
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "architecture": arch, "layers": 32, "vocab_size": 248320, "tensor_count": len(names), "tensor_types": quant_types}


def run(args):
    print("RUN", " ".join(str(value) for value in args), flush=True)
    subprocess.run([str(value) for value in args], check=True, cwd=ROOT)


def mmproj_metadata(path):
    reader = GGUFReader(path)
    file_type = reader.fields.get("general.file_type")
    if file_type is None or file_type.contents() != LlamaFileType.MOSTLY_F16:
        raise RuntimeError(f"Vision GGUF file type must be F16: {path}")
    if reader.fields["general.architecture"].contents() != "clip":
        raise RuntimeError("Unexpected multimodal GGUF architecture")
    names = {tensor.name for tensor in reader.tensors}
    if not {"v.patch_embd.weight", "v.patch_embd.weight.1"}.issubset(names):
        raise RuntimeError("The temporal vision patch weight was not split as expected")
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "type": "F16", "source_revision": SOURCE_REVISION, "tensor_count": len(reader.tensors), "source_tensor_count": len(reader.tensors) - 1, "layout_note": "One 5D temporal patch weight becomes two 4D GGUF tensors", "projector_type": reader.fields["clip.projector_type"].contents()}


def save_manifest(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def promote_verified(temporary, target, reader):
    """Validate converter output while it is still clearly marked partial."""
    entry = reader(temporary)
    temporary.replace(target)
    entry["file"] = target.name
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "models/original")
    parser.add_argument("--output", type=Path, default=ROOT / "models/gguf")
    parser.add_argument("--quants", nargs="+", choices=["Q4_K_M", "Q4_0", "Q4_K_S", "Q5_K_M", "Q8_0"], default=["Q4_K_M"])
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = source / "model.safetensors"
    source_checksum = sha256_file(original)
    if source_checksum != "f5062a1fba3e63afe5c88349a311903fec06c959b362f1247127e63d5c860ed4":
        raise RuntimeError("Original PUP weights do not match the pinned source revision")
    with original.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(length))
    if any(name.startswith(("mtp.", "model.mtp.")) for name in header):
        raise RuntimeError("Source weights changed: expected PUP without MTP tensors")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "format_version": 1,
        "source": {"model_id": SOURCE_MODEL, "revision": SOURCE_REVISION, "sha256": source_checksum, "tensor_count": len(header) - int("__metadata__" in header)},
        "converter_commit": "620d2225cc436ab0b001e3200e590e3b286ef9ab",
        "mtp": False,
        "models": {},
    }
    source_info = manifest.get("source", {})
    if any(source_info.get(key) != expected for key, expected in (("model_id", SOURCE_MODEL), ("revision", SOURCE_REVISION), ("sha256", source_checksum))):
        raise RuntimeError("Existing GGUF manifest describes a different source revision or checksum")
    converter = ROOT / "vendor/llama.cpp/convert_hf_to_gguf.py"
    quantizer = ROOT / "runtime/bin/llama-quantize.exe"
    bf16 = output / "flux3-klein-pup-BF16.gguf"
    mmproj = output / "mmproj-flux3-klein-pup-F16.gguf"
    if not bf16.exists():
        if shutil.disk_usage(output).free < 25 * 1024**3:
            raise RuntimeError("At least 25 GiB free is needed for BF16 GGUF and Q4 output")
        temporary = output / "flux3-klein-pup-BF16.partial.gguf"
        run([sys.executable, converter, source, "--no-mtp", "--outtype", "bf16", "--outfile", temporary])
        bf16_entry = promote_verified(temporary, bf16, lambda path: metadata(path, "BF16"))
    else:
        bf16_entry = metadata(bf16, "BF16")
    if not mmproj.exists():
        # The converter adds mmproj- itself if absent from the file name.
        temporary = output / "mmproj-flux3-klein-pup-F16.partial.gguf"
        run([sys.executable, converter, source, "--mmproj", "--outtype", "f16", "--outfile", temporary])
        projector_entry = promote_verified(temporary, mmproj, mmproj_metadata)
    else:
        projector_entry = mmproj_metadata(mmproj)
    manifest["models"]["BF16"] = bf16_entry
    manifest["mmproj"] = projector_entry
    if manifest["models"]["BF16"]["tensor_count"] + manifest["mmproj"]["source_tensor_count"] != manifest["source"]["tensor_count"]:
        raise RuntimeError("Converted text + visual tensor counts do not cover the source checkpoint")
    for quant in args.quants:
        path = output / f"flux3-klein-pup-{quant}.gguf"
        if not path.exists():
            temporary = output / f"flux3-klein-pup-{quant}.partial.gguf"
            run([quantizer, bf16, temporary, quant, "8"])
            entry = promote_verified(temporary, path, lambda path: metadata(path, quant))
        else:
            entry = metadata(path, quant)
        manifest["models"][quant] = entry
        save_manifest(manifest_path, manifest)
    save_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
