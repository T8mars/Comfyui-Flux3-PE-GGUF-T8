import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_audit_v2_identity import source_copy
from test_audit_v2_identity import load_script


ROOT = Path(__file__).resolve().parents[1]


def test_full_runner_ignores_inherited_pytest_selection(tmp_path, monkeypatch):
    module = load_script(ROOT / "scripts/run_checks.py")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "inputs", lambda: {"tiny.py": "fixed input"})
    monkeypatch.setattr(module, "source_identity", lambda: "tiny source")
    (tmp_path / "test_fixture.py").write_text("def test_kept(): pass\ndef test_other(): pass\n")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k kept")
    module.main()
    receipt = json.loads((tmp_path / "artifacts/test-identity.json").read_text())
    assert receipt["tests"] == 2, "Whole-suite receipt was issued for a filtered test subset"
    assert receipt["status"] == "passed"
    assert os.environ["PYTEST_ADDOPTS"] == "-k kept", "Runner must isolate only its child environment"


@pytest.mark.parametrize("name", ["pyproject.toml", "requirements.txt"])
def test_test_receipt_inputs_bind_execution_configuration(source_copy, monkeypatch, name):
    module = load_script(ROOT / "scripts/run_checks.py")
    monkeypatch.setattr(module, "ROOT", source_copy)
    for filename in ("pyproject.toml", "requirements.txt"):
        (source_copy / filename).write_text("initial execution configuration")
    before = module.inputs()
    (source_copy / name).write_text("changed test/dependency configuration")
    assert module.inputs() != before, f"Changed {name} reused the prior test input identity"
