"""Pinned download/quantization failure protection with tiny local byte fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json", "system_prompts.json", "README.md"]


def script(name):
    spec = importlib.util.spec_from_file_location("audit_build_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def download(tmp_path, monkeypatch):
    module = script("download_model")
    data = b"abcdefghij"
    monkeypatch.setattr(module, "SIZE", len(data))
    monkeypatch.setattr(module, "CHUNK", 4)
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    for name in CONFIG_FILES:
        (tmp_path / name).write_text("{}" if name.endswith(".json") else "fixture")
    requests = []
    def get(url, *, headers, **kwargs):
        value = headers["Range"]
        requests.append(value)
        start, end = map(int, value.removeprefix("bytes=").split("-"))
        body = data[start:end + 1]
        class Response:
            status_code = 206
            headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def iter_content(self, length): yield body
        return Response()
    monkeypatch.setattr(module.requests, "get", get)
    monkeypatch.setattr(sys, "argv", ["download_model.py", "--destination", str(tmp_path), "--workers", "1"])
    return module, tmp_path, data, requests


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_count_rejected_before_partial_or_network_effects(download, monkeypatch, workers):
    module, destination, _, calls = download
    monkeypatch.setattr(sys, "argv", ["download_model.py", "--destination", str(destination), "--workers", str(workers)])
    with pytest.raises((SystemExit, ValueError)):
        module.main()
    assert not (destination / "model.safetensors.download").exists()
    assert calls == []


def test_verified_resume_skips_good_chunk_and_installs_exact_final_range(download):
    module, destination, data, calls = download
    partial = destination / "model.safetensors.download"
    partial.write_bytes(data[:4] + b"\0" * 6)
    (destination / "model.safetensors.ranges.json").write_text(json.dumps({"0": hashlib.sha256(data[:4]).hexdigest()}))
    module.main()
    assert (destination / "model.safetensors").read_bytes() == data
    assert calls == ["bytes=4-7", "bytes=8-9"]


def test_resume_repairs_trailing_partial_bytes_without_discarding_verified_ranges(download):
    module, destination, data, calls = download
    (destination / "model.safetensors.download").write_bytes(data + b"stale trailing data")
    done = {str(index): hashlib.sha256(data[index * 4:(index + 1) * 4]).hexdigest() for index in range(3)}
    (destination / "model.safetensors.ranges.json").write_text(json.dumps(done))
    module.main()
    assert (destination / "model.safetensors").read_bytes() == data and calls == []


@pytest.mark.parametrize("journal", [[], {"999": "0" * 64}, {"invalid": "0" * 64}, {"0": 42}])
def test_invalid_resume_journal_rejected_before_requests(download, journal):
    module, destination, data, calls = download
    (destination / "model.safetensors.download").write_bytes(b"\0" * len(data))
    (destination / "model.safetensors.ranges.json").write_text(json.dumps(journal))
    with pytest.raises((ValueError, RuntimeError)):
        module.main()
    assert calls == []
    assert not (destination / "model.safetensors").exists()


def test_existing_wrong_final_file_is_preserved_without_partial_install(download):
    module, destination, _, calls = download
    (destination / "model.safetensors").write_bytes(b"user file")
    with pytest.raises(RuntimeError, match="Existing"):
        module.main()
    assert (destination / "model.safetensors").read_bytes() == b"user file"
    assert calls == [] and not (destination / "model.safetensors.download").exists()


@pytest.mark.parametrize("failure", ["status", "header", "short", "long"])
def test_bad_http_ranges_never_install_a_final_checkpoint(download, monkeypatch, failure):
    module, destination, _, calls = download
    original_get = module.requests.get
    def bad_get(*args, **kwargs):
        response = original_get(*args, **kwargs)
        if failure == "status":
            response.status_code = 200
        elif failure == "header":
            response.headers = {"Content-Range": "bytes 1-3/10"}
        elif failure == "short":
            response.iter_content = lambda length: iter([b"x"])
        else:
            response.iter_content = lambda length: iter([b"x" * 100])
        return response
    monkeypatch.setattr(module.requests, "get", bad_get)
    with pytest.raises(RuntimeError, match="range|Range"):
        module.main()
    assert calls and not (destination / "model.safetensors").exists()


def test_corrupted_resume_chunk_is_refetched_instead_of_trusting_journal(download):
    module, destination, data, calls = download
    (destination / "model.safetensors.download").write_bytes(b"0" * len(data))
    (destination / "model.safetensors.ranges.json").write_text(json.dumps({"0": hashlib.sha256(data[:4]).hexdigest()}))
    module.main()
    assert (destination / "model.safetensors").read_bytes() == data
    assert calls == ["bytes=0-3", "bytes=4-7", "bytes=8-9"]


@pytest.mark.parametrize("stage", ["BF16", "mmproj", "Q4_K_M"])
def test_invalid_converter_success_cannot_promote_a_bad_final_artifact(tmp_path, monkeypatch, stage):
    module = script("quantize")
    source, output = tmp_path / "original", tmp_path / "gguf"
    source.mkdir(); output.mkdir()
    header = json.dumps({"tensor": {}}).encode()
    original = source / "model.safetensors"
    original.write_bytes(struct.pack("<Q", len(header)) + header)
    real_hash = module.sha256_file
    monkeypatch.setattr(module, "sha256_file", lambda path: "f5062a1fba3e63afe5c88349a311903fec06c959b362f1247127e63d5c860ed4" if path == original else real_hash(path))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 1024**3))
    bf16 = output / "flux3-klein-pup-BF16.gguf"
    projector = output / "mmproj-flux3-klein-pup-F16.gguf"
    target = bf16 if stage == "BF16" else projector if stage == "mmproj" else output / "flux3-klein-pup-Q4_K_M.gguf"
    for path in (bf16, projector):
        if path != target:
            path.write_bytes(b"valid fixture")
    def metadata(path, expected_type=None):
        if path.read_bytes() == b"bad converter output":
            raise RuntimeError("invalid converted GGUF")
        return {"file": path.name, "tensor_count": 1}
    def mmproj_metadata(path):
        if path.read_bytes() == b"bad converter output":
            raise RuntimeError("invalid converted GGUF")
        return {"file": path.name, "source_tensor_count": 0}
    def run(args):
        temporary = Path(args[args.index("--outfile") + 1]) if "--outfile" in args else Path(args[2])
        temporary.write_bytes(b"bad converter output")
    monkeypatch.setattr(module, "metadata", metadata)
    monkeypatch.setattr(module, "mmproj_metadata", mmproj_metadata)
    monkeypatch.setattr(module, "run", run)
    monkeypatch.setattr(sys, "argv", ["quantize.py", "--source", str(source), "--output", str(output)])
    with pytest.raises(RuntimeError, match="invalid converted GGUF"):
        module.main()
    assert not target.exists(), "Invalid converter output was promoted to a final GGUF filename"
    assert list(output.glob("*.partial.gguf")), "Keep failed output clearly labeled partial for diagnosis"
