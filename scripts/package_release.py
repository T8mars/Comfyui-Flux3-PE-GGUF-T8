import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux3_pe.engine import sha256_file
from flux3_pe import __version__


def main():
    parser = argparse.ArgumentParser(description="Package the tested node, runtime, and Q4 artifacts")
    parser.add_argument("--quant", default="Q4_K_M", choices=["Q4_K_M"])
    parser.add_argument("--zip", action="store_true")
    parser.add_argument("--zip-directory", type=Path, help="Write the archive on a different volume when staging space is limited")
    parser.add_argument("--link-assets", action="store_true", help="Stage immutable model/runtime files with same-volume hard links to reduce disk use")
    args = parser.parse_args()
    zip_path = (args.zip_directory or ROOT / "release").resolve() / f"ComfyUI-Flux3-PE-{__version__}.zip"
    if args.zip and zip_path.exists():
        raise RuntimeError(f"Release archive already exists: {zip_path}")
    subprocess.run([sys.executable, str(ROOT / "scripts/validation_summary.py")], check=True)
    target = ROOT / "release" / __version__ / "ComfyUI-Flux3-PE"
    if target.exists():
        raise RuntimeError(f"Release directory already exists: {target}")
    target.mkdir(parents=True)
    for name in ("__init__.py", "nodes.py", "requirements.txt", "pyproject.toml", "LICENSE", "NOTICE", "README.md"):
        shutil.copy2(ROOT / name, target / name)
    for name in ("flux3_pe", "licenses", "examples", "patches", "scripts", "tests"):
        shutil.copytree(ROOT / name, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    validation = target / "validation"
    validation.mkdir()
    for name in ("comparison.json", "evaluation-q4.json", "evaluation-q4-original.json", "evaluation-q4-fast.json", "recovery-regression.json", "lifecycle-smoke.json", "comfy-smoke.json", "comfy-e2e-fresh.json", "tokenizer-parity.json", "vision-token-parity.json", "validation-summary.json", "transformers-reference.json"):
        source = ROOT / "artifacts" / name
        if source.is_file():
            shutil.copy2(source, validation / name)
    for name in ("review20", "audit20-v2", "audit20-v3", "audit20-v4", "audit20-v5", "canvas-qa"):
        if (ROOT / "artifacts" / name).exists():
            shutil.copytree(ROOT / "artifacts" / name, validation / name, ignore=shutil.ignore_patterns("__pycache__", "*.bak", "install-*", "installer", "*fixture*", "*.py", "*.cjs", "*.ps1"))
    shutil.copy2(ROOT / "artifacts/unit-tests.xml", validation / "unit-tests.xml")
    shutil.copy2(ROOT / "artifacts/test-identity.json", validation / "test-identity.json")
    shutil.copy2(ROOT / "logs/pup-native-test.log", validation / "native-sampling.log")
    def copy_asset(source, destination):
        if args.link_assets:
            os.link(source, destination)
        else:
            shutil.copy2(source, destination)
        return destination

    shutil.copytree(ROOT / "runtime/bin", target / "runtime/bin", copy_function=copy_asset)
    model_dir = target / "models/gguf"
    model_dir.mkdir(parents=True)
    manifest = json.loads((ROOT / "models/gguf/manifest.json").read_text(encoding="utf-8"))
    entries = [manifest["models"][args.quant], manifest["mmproj"]]
    for entry in entries:
        source = ROOT / "models/gguf" / entry["file"]
        if sha256_file(source) != entry["sha256"]:
            raise RuntimeError(f"Source artifact checksum failed: {source}")
        destination = model_dir / entry["file"]
        copy_asset(source, destination)
        if sha256_file(destination) != entry["sha256"]:
            raise RuntimeError(f"Packaged artifact checksum failed: {destination}")
    manifest["models"] = {args.quant: manifest["models"][args.quant]}
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (target / "INSTALL.txt").write_text("解压后，把整个 ComfyUI-Flux3-PE 文件夹放入 ComfyUI/custom_nodes。\n在该文件夹执行 PowerShell：& .\\scripts\\install_comfy.ps1 -ComfyUI '你的ComfyUI目录'\n脚本保存四套画布工作流和两张参考图；已有不同文件先备份。自定义目录可用 -UserDirectory / -InputDirectory。\n然后重启 ComfyUI，在工作流面板 FLUX3_PE 中打开。\n运行后端适用于 Windows x64 / NVIDIA，驱动需支持 CUDA 13.4。完整出图需要另备 FLUX3 出图模型、文本编码器及 VAE。\n", encoding="utf-8")
    for entry in entries:
        print("PACKAGED", entry["file"], entry["bytes"], flush=True)
    if args.zip:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        # A formal release name must never expose an incomplete archive.
        # Keep uniquely named failed bytes for diagnosis without occupying it.
        with tempfile.NamedTemporaryFile(prefix=zip_path.stem + "-", suffix=".partial.zip", dir=zip_path.parent, delete=False) as temporary_file:
            temporary = Path(temporary_file.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in target.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(target.parent))
        if os.name == "nt":
            # Windows rename is exclusive even if another publisher raced
            # the initial existence check.
            temporary.rename(zip_path)
        else:
            # POSIX rename replaces its destination; link preserves the same
            # create-exclusive publication contract on this same filesystem.
            os.link(temporary, zip_path)
            temporary.unlink()
        print("ZIP", zip_path, zip_path.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
