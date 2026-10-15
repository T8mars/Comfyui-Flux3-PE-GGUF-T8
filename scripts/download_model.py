"""Resume the pinned PUP download with verified HTTP ranges."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
from pathlib import Path
import threading
import time

import requests

REPO = "black-forest-labs/flux-3-image-klein-pup"
REVISION = "8b069751eb5dcab23829d9c5eccb470eb8b8a4ed"
SIZE = 18819722392
SHA256 = "f5062a1fba3e63afe5c88349a311903fec06c959b362f1247127e63d5c860ed4"
CHUNK = 32 * 1024 * 1024


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def validate_auxiliary(data: bytes, name: str):
    text = data.decode("utf-8")
    if not text.strip() or (name.endswith(".json") and not isinstance(json.loads(text), dict)):
        raise ValueError(f"Invalid model metadata: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "models/original")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    files = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "system_prompts.json", "README.md"]
    for name in files:
        path = destination / name
        if path.is_file():
            try:
                validate_auxiliary(path.read_bytes(), name)
                continue
            except (ValueError, OSError):
                pass
        response = requests.get(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", timeout=(20, 90))
        response.raise_for_status()
        validate_auxiliary(response.content, name)
        temporary = path.with_name(path.name + ".download")
        temporary.write_bytes(response.content)
        os.replace(temporary, path)
    target = destination / "model.safetensors"
    if target.exists():
        if target.stat().st_size == SIZE and checksum(target) == SHA256:
            print("Original weights already verified", flush=True)
            return
        raise RuntimeError("Existing model.safetensors failed integrity validation")
    partial = destination / "model.safetensors.download"
    journal = destination / "model.safetensors.ranges.json"
    done = json.loads(journal.read_text()) if journal.exists() and partial.exists() else {}
    count = math.ceil(SIZE / CHUNK)
    if not isinstance(done, dict) or any(
        not isinstance(key, str) or not key.isascii() or not key.isdigit()
        or str(int(key)) != key or not 0 <= int(key) < count
        or not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for key, value in done.items()
    ):
        raise ValueError("Invalid range journal; keys must be supported chunk indices with SHA256 values")
    if not partial.exists():
        with partial.open("wb") as stream:
            stream.truncate(SIZE)
    elif partial.stat().st_size != SIZE:
        # This is our resumable temporary, never an existing final checkpoint.
        # Keep valid ranges, remove stale trailing bytes or extend a short file.
        with partial.open("r+b") as stream:
            stream.truncate(SIZE)
    lock = threading.Lock()
    started = time.monotonic()
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/model.safetensors"

    def download(index):
        start, end = index * CHUNK, min(SIZE, (index + 1) * CHUNK) - 1
        if str(index) in done:
            with partial.open("rb") as stream:
                stream.seek(start)
                if hashlib.sha256(stream.read(end - start + 1)).hexdigest() == done[str(index)]:
                    return
        for attempt in range(8):
            try:
                digest = hashlib.sha256()
                written = 0
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(20, 90)) as response:
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
                        raise RuntimeError(f"Invalid range response: {response.status_code}, {response.headers.get('Content-Range')}")
                    with partial.open("r+b") as output:
                        output.seek(start)
                        for data in response.iter_content(1024 * 1024):
                            if written + len(data) > end - start + 1:
                                raise RuntimeError("Range response exceeded requested length")
                            output.write(data)
                            digest.update(data)
                            written += len(data)
                        output.flush()
                if written != end - start + 1:
                    raise RuntimeError("Incomplete range response")
                with lock:
                    done[str(index)] = digest.hexdigest()
                    temp = journal.with_suffix(".tmp")
                    temp.write_text(json.dumps(done, sort_keys=True), encoding="utf-8")
                    os.replace(temp, journal)
                    completed = sum(min(CHUNK, SIZE - int(key) * CHUNK) for key in done)
                    print(f"{len(done)}/{count} chunks; {completed / 1e9:.2f}/{SIZE / 1e9:.2f} GB; elapsed {time.monotonic() - started:.0f}s", flush=True)
                return
            except (requests.RequestException, OSError, RuntimeError) as error:
                if attempt == 7:
                    raise
                print(f"Retry chunk {index}: {type(error).__name__}", flush=True)
                time.sleep(min(2 ** attempt, 20))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(download, range(count)))
    print("Verifying original SHA256", flush=True)
    if checksum(partial) != SHA256:
        raise RuntimeError("Final model SHA256 mismatch; model has NOT been installed")
    os.replace(partial, target)
    print(f"Verified: {target}", flush=True)


if __name__ == "__main__":
    main()
