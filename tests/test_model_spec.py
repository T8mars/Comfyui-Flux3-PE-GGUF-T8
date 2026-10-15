import hashlib
import json
from pathlib import Path

import pytest

from flux3_pe.engine import ModelSpec
from flux3_pe.protocol import PEError, SOURCE_REVISION


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def model_files(tmp_path):
    model = tmp_path / "pup-Q4_K_M.gguf"
    projector = tmp_path / "mmproj-pup.gguf"
    server = tmp_path / "runtime/llama-server.exe"
    server.parent.mkdir()
    model.write_bytes(b"model fixture")
    projector.write_bytes(b"vision fixture")
    server.write_bytes(b"server fixture")
    libraries = {}
    for name in ("llama-common.dll", "llama-server-impl.dll"):
        payload = (name + " fixture").encode()
        (server.parent / name).write_bytes(payload)
        libraries[name] = digest(payload)
    manifest = {
        "source": {"revision": SOURCE_REVISION},
        "models": {"Q4_K_M": {"file": model.name, "bytes": model.stat().st_size, "sha256": digest(model.read_bytes())}},
        "mmproj": {"file": projector.name, "bytes": projector.stat().st_size, "sha256": digest(projector.read_bytes()), "source_revision": SOURCE_REVISION},
    }
    runtime = {"penalties_generated_only": True, "server_sha256": digest(server.read_bytes()), "core_sha256": libraries}
    model_manifest = tmp_path / "manifest.json"
    runtime_manifest = server.parent / "flux3-pe-runtime.json"
    def save():
        model_manifest.write_text(json.dumps(manifest), encoding="utf-8")
        runtime_manifest.write_text(json.dumps(runtime), encoding="utf-8")
    save()
    return {"model": model, "projector": projector, "server": server, "manifest": manifest, "runtime": runtime, "model_manifest": model_manifest, "runtime_manifest": runtime_manifest, "save": save}


def load(files, **kwargs):
    return ModelSpec.load(files["model"], server=files["server"], **kwargs)


def test_small_fixture_can_load_without_reading_real_model_artifacts(model_files):
    spec = load(model_files, context=4096)
    assert spec.model == model_files["model"] and spec.context == 4096
    assert spec.mmproj == model_files["projector"]


@pytest.mark.parametrize("artifact", ["model_manifest", "runtime_manifest"])
def test_missing_manifest_fails_explicitly(model_files, artifact):
    model_files[artifact].unlink()
    with pytest.raises(FileNotFoundError):
        load(model_files)


@pytest.mark.parametrize("artifact", ["model", "projector"])
def test_missing_or_truncated_weights_are_rejected(model_files, artifact):
    model_files[artifact].write_bytes(b"short")
    with pytest.raises(PEError, match="missing or incomplete"):
        load(model_files)


def test_source_revision_mismatch_is_rejected(model_files):
    model_files["manifest"]["source"]["revision"] = "another revision"
    model_files["save"]()
    with pytest.raises(PEError, match="revision"):
        load(model_files)


def test_projector_source_revision_mismatch_is_rejected(model_files):
    model_files["manifest"]["mmproj"]["source_revision"] = "another revision"
    model_files["save"]()
    with pytest.raises(PEError, match="revision"):
        load(model_files)


def test_projector_cannot_escape_model_directory(model_files):
    model_files["manifest"]["mmproj"]["file"] = "../outside.gguf"
    model_files["save"]()
    with pytest.raises(PEError, match="beside"):
        load(model_files)


def test_stub_and_patched_shared_libraries_are_both_bound(model_files):
    load(model_files)
    model_files["server"].write_bytes(b"different stub")
    with pytest.raises(PEError, match="checksum"):
        load(model_files)


@pytest.mark.parametrize("library", ["llama-common.dll", "llama-server-impl.dll"])
def test_changed_patched_library_is_rejected_even_after_cached_good_load(model_files, library):
    load(model_files)
    path = model_files["server"].parent / library
    path.write_bytes(b"corrupted library")
    with pytest.raises(PEError, match="checksum"):
        load(model_files)


def test_runtime_must_bind_both_patched_dlls(model_files):
    del model_files["runtime"]["core_sha256"]["llama-common.dll"]
    model_files["save"]()
    with pytest.raises(PEError, match="patched shared libraries"):
        load(model_files)


def test_runtime_library_path_cannot_escape_runtime_directory(model_files):
    model_files["runtime"]["core_sha256"]["../outside.dll"] = "irrelevant"
    model_files["save"]()
    with pytest.raises(PEError, match="Invalid or missing"):
        load(model_files)


def test_unsupported_backend_feature_is_rejected(model_files):
    model_files["runtime"]["penalties_generated_only"] = False
    model_files["save"]()
    with pytest.raises(PEError, match="generated-token-only"):
        load(model_files)


def test_runtime_identity_changes_if_bound_library_checksum_changes(model_files):
    first = load(model_files).revision
    name = "llama-common.dll"
    payload = b"new valid extension"
    (model_files["server"].parent / name).write_bytes(payload)
    model_files["runtime"]["core_sha256"][name] = digest(payload)
    model_files["save"]()
    assert load(model_files).revision != first
