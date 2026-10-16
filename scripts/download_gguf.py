"""Download the pinned production GGUF files and verify size and SHA256."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import tempfile

import requests

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "t8star/Flux3-PE-GGUF-Comfy"
REVISION = "b15cf2a0ada6c29949a23a1c16cbc8b4946e755d"
MANIFEST_SHA256 = "96d30b4fbe5b3987f99637bf735f0a90f6fd6bb54df95443fd9a511d69b6b28f"


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(session, destination, entry):
    name = entry["file"]
    if Path(name).name != name or any(character in name for character in ("/", "\\", ":")):
        raise ValueError("Artifact filename must stay inside the selected directory")
    target = destination / name
    if target.exists():
        if target.stat().st_size == entry["bytes"] and checksum(target) == entry["sha256"]:
            print(f"Verified existing {name}", flush=True)
            return
        raise ValueError(f"Existing artifact failed verification: {target}; move it aside before retrying")
    descriptor, temporary_name = tempfile.mkstemp(prefix=name + ".partial.", dir=destination)
    temporary = Path(temporary_name)
    try:
        digest, size = hashlib.sha256(), 0
        with os.fdopen(descriptor, "wb") as output:
            with session.get(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}", stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(8 * 1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Downloaded artifact failed size/SHA256 verification: {name}")
        # Keep an existing artifact if another downloader finished first.
        if target.exists():
            if target.stat().st_size != entry["bytes"] or checksum(target) != entry["sha256"]:
                raise ValueError(f"A different artifact appeared while downloading: {target}")
        else:
            temporary.replace(target)
        print(f"Downloaded and verified {name}: {size} bytes", flush=True)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "models/gguf")
    args = parser.parse_args()
    destination = args.directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        response = session.get(f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/manifest.json", timeout=30)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != MANIFEST_SHA256:
            raise ValueError("The pinned model manifest failed SHA256 verification")
        manifest = response.json()
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if any(existing.get(key) != manifest[key] for key in ("source", "mmproj")) or existing.get("models", {}).get("Q4_K_M") != manifest["models"]["Q4_K_M"]:
                raise ValueError("Existing model manifest differs; select another download directory")
        for entry in (manifest["models"]["Q4_K_M"], manifest["mmproj"]):
            download(session, destination, entry)
        if not manifest_path.exists():
            manifest_path.write_bytes(response.content)


if __name__ == "__main__":
    main()
