"""Evidence identity uses tiny isolated sources and never runs model inference."""
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from flux3_pe import identity


ROOT = Path(__file__).resolve().parents[1]


def load_script(path):
    spec = importlib.util.spec_from_file_location("audit_identity_fixture_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source_copy(tmp_path):
    for name in identity.CORE_FILES:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    return tmp_path


def test_imported_identity_stays_fixed_when_disk_sources_change(source_copy):
    captured = load_script(source_copy / "flux3_pe/identity.py")
    imported = captured.SOURCE_IDENTITY
    assert imported == identity.source_identity(source_copy)
    with (source_copy / "flux3_pe/protocol.py").open("ab") as stream:
        stream.write(b"\n# source drift fixture\n")
    assert captured.SOURCE_IDENTITY == imported
    assert captured.source_identity(source_copy) != imported


def test_missing_core_file_cannot_claim_a_complete_source_identity(source_copy):
    (source_copy / "nodes.py").unlink()
    with pytest.raises(FileNotFoundError):
        identity.source_identity(source_copy)


def test_validation_inputs_bind_tests_and_scripts_but_ignore_python_cache(source_copy):
    (source_copy / "tests/__pycache__").mkdir(parents=True)
    (source_copy / "scripts").mkdir()
    (source_copy / "tests/test_actual.py").write_text("test fixture")
    (source_copy / "scripts/run_actual.py").write_text("runner fixture")
    before = identity.validation_inputs(source_copy)
    (source_copy / "tests/__pycache__/fixture.pyc").write_bytes(b"cache")
    assert identity.validation_inputs(source_copy) == before
    (source_copy / "scripts/run_actual.py").write_text("changed runner")
    after = identity.validation_inputs(source_copy)
    assert after != before
    assert "tests/test_actual.py" in after and "scripts/run_actual.py" in after


@pytest.fixture
def runner(tmp_path, monkeypatch):
    module = load_script(ROOT / "scripts/run_checks.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    state = {"inputs": {"core.py": "tested source hash"}}
    monkeypatch.setattr(module, "inputs", lambda: dict(state["inputs"]))
    monkeypatch.setattr(module, "source_identity", lambda: "fixture source identity")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    receipt = artifacts / "test-identity.json"
    junit = artifacts / "unit-tests.xml"
    receipt.write_text(json.dumps({"status": "passed", "old_receipt": True}))
    return module, state, receipt, junit


GREEN = '<testsuites><testsuite><testcase name="real fixture"/></testsuite></testsuites>'


def install_fake_runner(monkeypatch, runner, *, code=0, xml=GREEN, drift=False):
    module, state, receipt, junit = runner
    def execute(args, cwd, env):
        assert env["PYTEST_ADDOPTS"] == ""
        assert json.loads(receipt.read_text())["status"] == "running"
        if xml is not None:
            junit.write_text(xml)
        if drift:
            state["inputs"]["core.py"] = "changed during tests"
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr(module.subprocess, "run", execute)


def test_runner_passed_receipt_binds_fresh_junit_and_tested_inputs(runner, monkeypatch):
    module, state, receipt, junit = runner
    install_fake_runner(monkeypatch, runner)
    module.main()
    report = json.loads(receipt.read_text())
    assert report["status"] == "passed" and not report.get("old_receipt")
    assert report["inputs_sha256"] == state["inputs"]
    assert report["junit_sha256"] == module.sha256_file(junit)


@pytest.mark.parametrize("code,xml,drift", [
    (1, GREEN, False),
    (0, '<testsuites><testsuite><testcase><skipped/></testcase></testsuite></testsuites>', False),
    (0, '<testsuites><testsuite><testcase><failure/></testcase></testsuite></testsuites>', False),
    (0, GREEN, True),
])
def test_runner_failed_skipped_or_drifted_inputs_never_get_passed_receipt(runner, monkeypatch, code, xml, drift):
    module, _, receipt, _ = runner
    install_fake_runner(monkeypatch, runner, code=code, xml=xml, drift=drift)
    with pytest.raises(SystemExit):
        module.main()
    report = json.loads(receipt.read_text())
    assert report["status"] == "failed" and "junit_sha256" not in report


def test_old_green_junit_cannot_be_rebound_if_runner_writes_no_new_report(runner, monkeypatch):
    module, _, receipt, junit = runner
    junit.write_text(GREEN)
    install_fake_runner(monkeypatch, runner, xml=None)
    with pytest.raises((SystemExit, FileNotFoundError)):
        module.main()
    assert json.loads(receipt.read_text())["status"] != "passed"
