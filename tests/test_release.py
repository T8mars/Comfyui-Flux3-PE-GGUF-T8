import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location("test_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def release(tmp_path, monkeypatch):
    verify = script("verify_release")
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    target = tmp_path / "release" / verify.__version__ / "ComfyUI-Flux3-PE"
    target.mkdir(parents=True)
    (tmp_path / "artifacts").mkdir()
    contents = {"ComfyUI-Flux3-PE/nodes.py": b"alpha fixture", "ComfyUI-Flux3-PE/examples/01.json": b"{}"}
    audit = {"source_files_sha256": {"nodes.py": hashlib.sha256(contents["ComfyUI-Flux3-PE/nodes.py"]).hexdigest()}, "workflow_files_sha256": {"examples/01.json": hashlib.sha256(contents["ComfyUI-Flux3-PE/examples/01.json"]).hexdigest()}}
    contents["ComfyUI-Flux3-PE/validation/validation-summary.json"] = json.dumps(audit).encode()
    for name, data in contents.items():
        path = target.parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    archive = tmp_path / "release" / f"ComfyUI-Flux3-PE-{verify.__version__}.zip"
    def write(files=None):
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
            for name, data in (contents if files is None else files).items():
                output.writestr(name, data)
    write()
    return verify, archive, target, contents, write


def test_complete_zip_passes_and_ignores_only_post_run_cache_and_logs(release):
    verify, archive, target, _, _ = release
    (target / "logs").mkdir()
    (target / "logs/llama.log").write_text("runtime-only")
    (target / "__pycache__").mkdir()
    (target / "__pycache__/nodes.pyc").write_bytes(b"runtime-only")
    verify.main()
    report = json.loads((verify.ROOT / "artifacts" / f"release-integrity-{verify.__version__}.json").read_text(encoding="utf-8"))
    assert report["entries"] == 3 and report["crc_all_entries_passed"]
    assert archive.with_suffix(".zip.sha256").read_text(encoding="ascii").startswith(report["sha256"])


def test_separately_stored_archive_is_fully_verified(release, tmp_path):
    verify, archive, _, _, _ = release
    destination = tmp_path / "archive volume [qa]"
    destination.mkdir()
    moved = destination / archive.name
    archive.rename(moved)
    verify.main(["--zip-directory", str(destination)])
    report = json.loads((verify.ROOT / "artifacts" / f"release-integrity-{verify.__version__}.json").read_text())
    assert report["zip"] == str(moved) and report["entries"] == 3
    assert moved.with_suffix(".zip.sha256").is_file()


@pytest.mark.parametrize("kind", ["missing", "extra", "changed"])
def test_missing_extra_and_changed_zip_members_are_rejected(release, kind):
    verify, _, _, contents, write = release
    files = dict(contents)
    if kind == "missing":
        del files["ComfyUI-Flux3-PE/nodes.py"]
    elif kind == "extra":
        files["ComfyUI-Flux3-PE/unexpected.dat"] = b"unexpected"
    else:
        files["ComfyUI-Flux3-PE/nodes.py"] = b"changed but valid CRC"
    write(files)
    with pytest.raises(RuntimeError, match="member list|checksum"):
        verify.main()


def test_duplicate_zip_path_is_rejected(release):
    verify, archive, _, contents, _ = release
    with zipfile.ZipFile(archive, "a") as output:
        with pytest.warns(UserWarning, match="Duplicate name"):
            output.writestr("ComfyUI-Flux3-PE/nodes.py", contents["ComfyUI-Flux3-PE/nodes.py"])
    with pytest.raises(RuntimeError, match="member list"):
        verify.main()


def test_corrupted_zip_member_crc_is_rejected(release):
    verify, archive, _, _, _ = release
    data = bytearray(archive.read_bytes())
    offset = data.index(b"alpha fixture")
    data[offset] ^= 1
    archive.write_bytes(data)
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        verify.main()


def test_consistently_changed_staged_core_and_zip_still_fail_audit_binding(release):
    verify, _, target, contents, write = release
    changed = b"unreviewed nodes after test evidence was recorded"
    (target / "nodes.py").write_bytes(changed)
    files = dict(contents)
    files["ComfyUI-Flux3-PE/nodes.py"] = changed
    write(files)
    with pytest.raises(RuntimeError, match="audit|validation"):
        verify.main()


@pytest.mark.parametrize("quant", ["BF16", "Q4_0"])
def test_unvalidated_quant_cannot_reuse_production_q4_validation(monkeypatch, quant):
    package = script("package_release")
    monkeypatch.setattr(sys, "argv", ["package_release.py", "--quant", quant])
    def unexpected_validation(*args, **kwargs):
        raise AssertionError("An unvalidated quant reached the K_M validation gate")
    monkeypatch.setattr(package.subprocess, "run", unexpected_validation)
    with pytest.raises(SystemExit) as error:
        package.main()
    assert error.value.code == 2


@pytest.mark.parametrize("field", ["source_files_sha256", "workflow_files_sha256"])
def test_missing_audit_hash_map_cannot_silently_disable_binding(release, field):
    verify, _, target, contents, write = release
    audit_name = "ComfyUI-Flux3-PE/validation/validation-summary.json"
    audit = json.loads(contents[audit_name])
    del audit[field]
    contents[audit_name] = json.dumps(audit).encode()
    (target / "validation/validation-summary.json").write_bytes(contents[audit_name])
    write()
    with pytest.raises(RuntimeError, match="audit|binding|hash"):
        verify.main()


def test_matching_staging_and_zip_cannot_add_unaudited_executable_source(release):
    verify, _, target, contents, write = release
    name = "ComfyUI-Flux3-PE/scripts/extra.py"
    contents[name] = b"print('unreviewed executable source')"
    (target / "scripts").mkdir()
    (target / "scripts/extra.py").write_bytes(contents[name])
    write()
    with pytest.raises(RuntimeError, match="audit|binding|hash"):
        verify.main()


def rewrite_audit(release, change):
    _, _, target, contents, write = release
    name = "ComfyUI-Flux3-PE/validation/validation-summary.json"
    audit = json.loads(contents[name])
    change(audit)
    contents[name] = json.dumps(audit).encode()
    (target / "validation/validation-summary.json").write_bytes(contents[name])
    write()


@pytest.mark.parametrize("field", ["source_files_sha256", "workflow_files_sha256"])
@pytest.mark.parametrize("invalid", [{}, [], None, "checksum"])
def test_empty_or_wrong_type_audit_map_fails_explicitly(release, field, invalid):
    rewrite_audit(release, lambda audit: audit.update({field: invalid}))
    with pytest.raises(RuntimeError, match="binding|hash"):
        release[0].main()


def test_windows_backslash_audit_paths_remain_compatible(release):
    def change(audit):
        audit["workflow_files_sha256"] = {name.replace("/", "\\"): digest for name, digest in audit["workflow_files_sha256"].items()}
    rewrite_audit(release, change)
    release[0].main()


def test_normalized_duplicate_audit_paths_are_rejected(release):
    def change(audit):
        hashes = audit["workflow_files_sha256"]
        hashes["examples\\01.json"] = hashes["examples/01.json"]
    rewrite_audit(release, change)
    with pytest.raises(RuntimeError, match="duplicate normalized"):
        release[0].main()


def test_extra_unaudited_gui_file_in_matching_staging_and_zip_is_rejected(release):
    verify, _, target, contents, write = release
    contents["ComfyUI-Flux3-PE/examples/extra_canvas.json"] = b'{"nodes": []}'
    (target / "examples/extra_canvas.json").write_bytes(contents["ComfyUI-Flux3-PE/examples/extra_canvas.json"])
    write()
    with pytest.raises(RuntimeError, match="cover every"):
        verify.main()


@pytest.mark.parametrize("name", ["../nodes.py", "/nodes.py", "D:/nodes.py", "examples/../nodes.py"])
def test_audit_paths_cannot_replace_the_real_relative_member_with_an_alias(release, name):
    def change(audit):
        digest = audit["source_files_sha256"].pop("nodes.py")
        audit["source_files_sha256"][name] = digest
    rewrite_audit(release, change)
    with pytest.raises(RuntimeError, match="differs from its tested audit"):
        release[0].main()


@pytest.mark.parametrize("field", ["source_files_sha256", "workflow_files_sha256"])
@pytest.mark.parametrize("wrong", [None, 123, "not-a-sha256", "0" * 64])
def test_wrong_checksum_contents_cannot_match_staged_files(release, field, wrong):
    def change(audit):
        first = next(iter(audit[field]))
        audit[field][first] = wrong
    rewrite_audit(release, change)
    with pytest.raises(RuntimeError, match="differs from its tested audit"):
        release[0].main()


def test_source_and_workflow_maps_must_cover_the_correct_category(release):
    def change(audit):
        audit["source_files_sha256"], audit["workflow_files_sha256"] = audit["workflow_files_sha256"], audit["source_files_sha256"]
    rewrite_audit(release, change)
    with pytest.raises(RuntimeError, match="cover every"):
        release[0].main()
