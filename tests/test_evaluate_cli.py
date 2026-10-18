from pathlib import Path
import subprocess
import sys
import importlib.util
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arguments,message", [
    (["--ids", "does_not_exist"], "Unknown case IDs"),
    (["--ids", "t2i_01", "typo_id"], "Unknown case IDs"),
    (["--limit", "0"], "limit must be positive"),
    (["--limit", "-1"], "limit must be positive"),
    (["--max-tokens", "0"], "max-tokens must be in"),
    (["--max-tokens", "3585"], "max-tokens must be in"),
])
def test_invalid_selection_or_budget_fails_before_model_loading(tmp_path, arguments, message):
    output = tmp_path / "must-not-be-written.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), *arguments, "--output", str(output)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert message in result.stderr
    assert not output.exists()


@pytest.fixture
def evaluation():
    spec = importlib.util.spec_from_file_location("evaluate_gpu_monitor_fixture", ROOT / "scripts/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_host_without_nvidia_utility_never_spawns_a_process(evaluation, monkeypatch):
    monkeypatch.setattr(evaluation.shutil, "which", lambda _: None)
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("process spawned")))
    assert evaluation.gpu_sample() is None


@pytest.mark.parametrize("output,expected", [("7000, 80\n", [7000, 80]), ("7000, 80\n1000, 20\n", [8000, 80]), ("N/A, N/A\n", None), ("", None)])
def test_gpu_monitor_accepts_single_and_multiple_devices_and_unsupported_metrics(evaluation, monkeypatch, output, expected):
    monkeypatch.setattr(evaluation.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=output))
    assert evaluation.gpu_sample() == expected


def test_missing_driver_during_monitoring_does_not_kill_its_thread(evaluation, monkeypatch):
    monkeypatch.setattr(evaluation.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(evaluation.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("driver unavailable")))
    assert evaluation.gpu_sample() is None


def test_cpu_evaluation_completes_without_starting_gpu_sampling(evaluation, tmp_path, monkeypatch):
    import json
    import threading
    import time

    model = tmp_path / "models/gguf/flux3-klein-pup-Q4_K_M.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"fixture")
    output = tmp_path / "evaluation.json"
    errors, calls = [], []
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "cases", lambda: [{"id": "cpu", "prompt": "A ball", "images": [], "boxes": None}])
    monkeypatch.setattr(evaluation, "fixtures", lambda: {})
    monkeypatch.setattr(evaluation, "evaluation_input_identity", lambda: "fixture")
    monkeypatch.setattr(evaluation.ModelSpec, "load", lambda *a, **k: SimpleNamespace(source_model="fixture/pup", source_revision="fixture-revision"))
    monkeypatch.setattr(evaluation, "release_model", lambda: None)
    monkeypatch.setattr(evaluation, "gpu_sample", lambda: calls.append("sample") or [100, 50])
    monkeypatch.setattr(threading, "excepthook", lambda event: errors.append(event.exc_value))
    def enhance(*args, **kwargs):
        time.sleep(0.7)  # Allow a monitoring tick if the CPU guard is missing.
        return "A red ball.", {"generation_seconds": 0.7}
    monkeypatch.setattr(evaluation, "enhance", enhance)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--gpu-layers", "0", "--output", str(output)])
    evaluation.main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["Q4_K_M"]["passed"] == 1
    assert report["source_model"] == "fixture/pup" and report["source_revision"] == "fixture-revision"
    assert report["results"][0]["gpu_peak_mib_total"] is None
    assert calls == errors == []
