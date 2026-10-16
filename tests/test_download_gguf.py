import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def downloader():
    path = Path(__file__).resolve().parents[1] / "scripts/download_gguf.py"
    spec = importlib.util.spec_from_file_location("download_gguf_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    def __init__(self, data): self.data = data
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_content(self, size): yield self.data


def entry(payload):
    return {"file": "pup.gguf", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_valid_download_is_committed_only_after_size_and_hash_verification(downloader, tmp_path):
    payload = b"verified model"
    downloader.download(SimpleNamespace(get=lambda *a, **k: Response(payload)), tmp_path, entry(payload))
    assert (tmp_path / "pup.gguf").read_bytes() == payload
    assert not list(tmp_path.glob("*.partial.*"))


@pytest.mark.parametrize("bad_payload", [b"truncated", b"wrong same size"])
def test_invalid_download_never_exposes_a_model_or_leaves_partials(downloader, tmp_path, bad_payload):
    with pytest.raises(ValueError, match="verification"):
        downloader.download(SimpleNamespace(get=lambda *a, **k: Response(bad_payload)), tmp_path, entry(b"correct content"))
    assert not list(tmp_path.iterdir())


def test_existing_unverified_artifact_is_never_overwritten(downloader, tmp_path):
    target = tmp_path / "pup.gguf"
    target.write_bytes(b"user file")
    session = SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(AssertionError("network accessed")))
    with pytest.raises(ValueError, match="Existing artifact"):
        downloader.download(session, tmp_path, entry(b"verified model"))
    assert target.read_bytes() == b"user file"


@pytest.mark.parametrize("name", ["../escape", "..\\escape", "C:escape"])
def test_manifest_cannot_escape_the_destination(downloader, tmp_path, name):
    with pytest.raises(ValueError, match="filename"):
        downloader.download(None, tmp_path, {**entry(b"model"), "file": name})
