"""Verify every packaged file and ZIP member, then write a SHA256 receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux3_pe import __version__
from flux3_pe.engine import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-directory", type=Path, help="Directory containing the separately stored release archive")
    args = parser.parse_args([] if argv is None else argv)
    target = ROOT / "release" / __version__ / "ComfyUI-Flux3-PE"
    archive_path = (args.zip_directory or ROOT / "release").resolve() / f"ComfyUI-Flux3-PE-{__version__}.zip"
    expected = {"ComfyUI-Flux3-PE/" + path.relative_to(target).as_posix(): sha256_file(path) for path in target.rglob("*") if path.is_file() and "__pycache__" not in path.parts and "logs" not in path.relative_to(target).parts}
    audit = json.loads((target / "validation/validation-summary.json").read_text(encoding="utf-8"))
    audited = {}
    for field in ("source_files_sha256", "workflow_files_sha256"):
        hashes = audit.get(field)
        if not isinstance(hashes, dict) or not hashes:
            raise RuntimeError(f"Release audit is missing its hash binding: {field}")
        audited[field] = {"ComfyUI-Flux3-PE/" + name.replace("\\", "/") for name in hashes}
        if len(audited[field]) != len(hashes):
            raise RuntimeError(f"Release audit has duplicate normalized paths: {field}")
        for name, checksum in hashes.items():
            member = "ComfyUI-Flux3-PE/" + name.replace("\\", "/")
            if expected.get(member) != checksum:
                raise RuntimeError(f"Release file differs from its tested audit: {name}")
    workflows = {name for name in expected if name.startswith("ComfyUI-Flux3-PE/examples/")}
    sources = {name for name in expected if name.split("/")[1] not in {"models", "runtime", "validation", "examples"} and name != "ComfyUI-Flux3-PE/INSTALL.txt"}
    if audited["source_files_sha256"] != sources or audited["workflow_files_sha256"] != workflows:
        raise RuntimeError("Release audit hash binding does not cover every shipped source and workflow")
    verified = []
    with zipfile.ZipFile(archive_path) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise RuntimeError("ZIP member list differs from the complete release directory")
        for info in archive.infolist():
            if info.is_dir():
                continue
            checksum = hashlib.sha256()
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
            if checksum.hexdigest() != expected[info.filename]:
                raise RuntimeError(f"ZIP member checksum failed: {info.filename}")
            verified.append(info.filename)
    checksum = sha256_file(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(checksum + "  " + archive_path.name + "\n", encoding="ascii")
    report = {"version": __version__, "zip": str(archive_path), "bytes": archive_path.stat().st_size, "sha256": checksum, "crc_all_entries_passed": True, "entries": len(verified), "sha256_verified_members": verified}
    (ROOT / "artifacts" / f"release-integrity-{__version__}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("version", "bytes", "sha256", "entries", "crc_all_entries_passed")}, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
