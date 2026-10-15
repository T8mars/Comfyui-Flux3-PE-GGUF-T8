"""The evaluation gate rejects changed corpus pixels/scripts and incomplete runs."""
import copy
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location("audit_evidence_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("change", ["prompt", "box", "reference_order", "reference_pixels", "script"])
def test_evaluation_identity_changes_when_actual_evaluation_inputs_change(monkeypatch, tmp_path, change):
    evaluate = script("evaluate")
    baseline = evaluate.evaluation_input_identity()
    if change in ("prompt", "box", "reference_order"):
        cases = copy.deepcopy(evaluate.cases())
        if change == "prompt":
            cases[0]["prompt"] += " changed corpus"
        elif change == "box":
            cases[20]["boxes"][0]["bbox"][0] += 0.01
        else:
            cases[15]["images"].reverse()
        monkeypatch.setattr(evaluate, "cases", lambda: cases)
    elif change == "reference_pixels":
        images = evaluate.fixtures()
        images["a"].putpixel((0, 0), (1, 2, 3))
        monkeypatch.setattr(evaluate, "fixtures", lambda: images)
    else:
        changed = tmp_path / "evaluate.py"
        changed.write_bytes(Path(evaluate.__file__).read_bytes() + b"\n# changed script fixture\n")
        monkeypatch.setattr(evaluate, "__file__", str(changed))
    assert evaluate.evaluation_input_identity() != baseline


def test_evaluation_identity_is_independent_of_fixture_dictionary_order(monkeypatch):
    evaluate = script("evaluate")
    baseline = evaluate.evaluation_input_identity()
    images = evaluate.fixtures()
    monkeypatch.setattr(evaluate, "fixtures", lambda: dict(reversed(list(images.items()))))
    assert evaluate.evaluation_input_identity() == baseline


@pytest.mark.parametrize("change", ["input_identity", "missing", "duplicate", "reordered"])
def test_gate_rejects_stale_or_incomplete_corpus_before_later_evidence(monkeypatch, change):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    gate = script("validation_summary")
    cases = gate.cases()
    evaluation = {"input_identity": gate.evaluation_input_identity(), "results": [
        {"request": case, "passed": True} for case in cases
    ]}
    if change == "input_identity":
        evaluation["input_identity"] = "stale corpus checksum"
    elif change == "missing":
        evaluation["results"].pop()
    elif change == "duplicate":
        evaluation["results"][-1] = copy.deepcopy(evaluation["results"][0])
    else:
        evaluation["results"].reverse()
    def read(name):
        assert name == "evaluation-q4.json", "Invalid corpus reached unrelated later evidence"
        return evaluation
    monkeypatch.setattr(gate, "read", read)
    with pytest.raises(RuntimeError, match="corpus|requests|script"):
        gate.main()
