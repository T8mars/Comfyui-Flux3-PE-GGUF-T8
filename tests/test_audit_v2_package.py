"""Exercise copy/hardlink release staging with tiny fabricated weight bytes."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def package(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("audit_package_fixture", ROOT / "scripts/package_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    for name in ("__init__.py", "nodes.py", "requirements.txt", "pyproject.toml", "LICENSE", "NOTICE", "README.md"):
        (tmp_path / name).write_bytes(name.encode())
    for folder in ("flux3_pe", "licenses", "examples", "patches", "scripts", "tests"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "fixture.txt").write_bytes(folder.encode())
    (tmp_path / "flux3_pe/__pycache__").mkdir()
    (tmp_path / "flux3_pe/__pycache__/generated.pyc").write_bytes(b"ignored cache")
    (tmp_path / "runtime/bin").mkdir(parents=True)
    for name in ("llama-server.exe", "patched.dll", "flux3-pe-runtime.json"):
        (tmp_path / "runtime/bin" / name).write_bytes(name.encode())
    models = tmp_path / "models/gguf"
    models.mkdir(parents=True)
    def entry(name):
        payload = (name + " tiny synthetic weights").encode()
        (models / name).write_bytes(payload)
        return {"file": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {
        "source": {"revision": "fixture-only"},
        "models": {kind: entry("pup-" + kind + ".gguf") for kind in ("BF16", "Q4_K_M", "Q4_0")},
        "mmproj": entry("mmproj-pup-F16.gguf"),
    }
    (models / "manifest.json").write_text(json.dumps(manifest))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for name in ("unit-tests.xml", "test-identity.json", "validation-summary.json", "evaluation-q4.json"):
        (artifacts / name).write_bytes(name.encode())
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/pup-native-test.log").write_text("native fixture")
    review = artifacts / "audit20-v2"
    review.mkdir()
    (review / "rounds.json").write_text("[]")
    (review / "temporary.py").write_text("not release source")
    (review / "temporary.ps1").write_text("not release source")
    for folder in ("helper-fixtures", "resume-fixture", "resume-identity-fixture"):
        (review / folder).mkdir()
        (review / folder / "mock.json").write_text("synthetic test-only fixture")
    target = tmp_path / "release" / module.__version__ / "ComfyUI-Flux3-PE"
    return module, target, manifest


def run_package(package, monkeypatch, linked):
    module, target, _ = package
    monkeypatch.setattr(sys, "argv", ["package_release.py", "--zip"] + (["--link-assets"] if linked else []))
    module.main()
    return target


@pytest.mark.parametrize("linked", [False, True])
def test_copy_and_hardlink_release_have_identical_production_contents(package, monkeypatch, linked):
    module, target, original = package
    run_package(package, monkeypatch, linked)
    manifest = json.loads((target / "models/gguf/manifest.json").read_text())
    assert set(manifest["models"]) == {"Q4_K_M"}
    assert manifest["mmproj"] == original["mmproj"]
    assert sorted(path.name for path in (target / "models/gguf").iterdir()) == ["manifest.json", "mmproj-pup-F16.gguf", "pup-Q4_K_M.gguf"]
    for relative in ("models/gguf/pup-Q4_K_M.gguf", "models/gguf/mmproj-pup-F16.gguf", "runtime/bin/patched.dll"):
        source, destination = module.ROOT / relative, target / relative
        assert destination.read_bytes() == source.read_bytes()
        assert os.path.samefile(source, destination) is linked
    assert not (target / "flux3_pe/__pycache__").exists()
    assert not (target / "validation/audit20-v2/temporary.py").exists()
    assert not (target / "validation/audit20-v2/temporary.ps1").exists()
    archive = module.ROOT / "release" / f"ComfyUI-Flux3-PE-{module.__version__}.zip"
    with zipfile.ZipFile(archive) as output:
        for member in output.namelist():
            assert output.read(member) == (target.parent / member).read_bytes()
        assert not any("BF16" in name or "Q4_0" in name for name in output.namelist())


def test_validation_package_excludes_all_synthetic_fixture_directories(package, monkeypatch):
    _, target, _ = package
    run_package(package, monkeypatch, False)
    evidence = target / "validation/audit20-v2"
    assert (evidence / "rounds.json").is_file()
    assert not any("fixture" in path.name for path in evidence.rglob("*")), "Test fixture sources/reports leaked into release evidence"


def test_bad_source_weight_cannot_be_installed_or_modify_original(package, monkeypatch):
    module, target, _ = package
    original = module.ROOT / "models/gguf/pup-Q4_K_M.gguf"
    original.write_bytes(b"deliberately corrupted fixture")
    monkeypatch.setattr(sys, "argv", ["package_release.py"])
    with pytest.raises(RuntimeError, match="Source artifact checksum failed"):
        module.main()
    assert original.read_bytes() == b"deliberately corrupted fixture"
    assert not (target / "models/gguf/pup-Q4_K_M.gguf").exists()


def test_archive_can_use_separate_directory_with_literal_path(package, tmp_path, monkeypatch):
    module, target, _ = package
    destination = tmp_path / "separate archives [qa]"
    monkeypatch.setattr(sys, "argv", ["package_release.py", "--zip", "--link-assets", "--zip-directory", str(destination)])
    module.main()
    archive = destination / f"ComfyUI-Flux3-PE-{module.__version__}.zip"
    assert archive.is_file() and target.is_dir()
    assert not (module.ROOT / "release" / archive.name).exists()
    with zipfile.ZipFile(archive) as output:
        assert output.read("ComfyUI-Flux3-PE/models/gguf/pup-Q4_K_M.gguf") == (target / "models/gguf/pup-Q4_K_M.gguf").read_bytes()


def test_existing_archive_is_preserved_before_any_staging(package, tmp_path, monkeypatch):
    module, target, _ = package
    archive = tmp_path / f"ComfyUI-Flux3-PE-{module.__version__}.zip"
    archive.write_bytes(b"previous release must remain intact")
    monkeypatch.setattr(sys, "argv", ["package_release.py", "--zip", "--zip-directory", str(tmp_path)])
    with pytest.raises(RuntimeError, match="archive already exists"):
        module.main()
    assert archive.read_bytes() == b"previous release must remain intact"
    assert not target.exists()
