"""Cancellation covers preparation and failed cleanup during startup."""
from pathlib import Path
import subprocess
import time
from unittest.mock import Mock

import pytest

import flux3_pe.engine as backend
from flux3_pe.protocol import PEError


def spec():
    return backend.ModelSpec(Path("model"), Path("vision"), Path("server"), "audit4-cancel")


def test_pre_cancelled_enhance_does_not_prepare_reference_payload(monkeypatch):
    prepare = Mock(side_effect=AssertionError("Reference preparation must not run"))
    monkeypatch.setattr(backend, "prepare_request", prepare)
    def interrupt():
        raise InterruptedError("cancelled before preparation")
    with pytest.raises(InterruptedError, match="before preparation"):
        backend.enhance(spec(), "A ball", interrupt=interrupt)
    prepare.assert_not_called()


def test_startup_cancel_cleanup_uses_short_waits_and_retains_owned_handle(tmp_path, monkeypatch):
    class Stubborn:
        exited = False
        def __init__(self):
            self.waits = []
        def poll(self):
            return 0 if self.exited else None
        def terminate(self):
            pass
        def kill(self):
            pass
        def wait(self, timeout):
            self.waits.append(timeout)
            if self.exited:
                return 0
            time.sleep(min(timeout, 1))
            raise subprocess.TimeoutExpired("owned CPU fixture", timeout)
    process = Stubborn()
    monkeypatch.setattr(backend.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setenv("FLUX3_PE_LOG_DIR", str(tmp_path))
    engine = backend.LlamaEngine(spec())
    calls = 0
    def interrupt():
        nonlocal calls
        calls += 1
        if calls > 1:
            raise InterruptedError("startup cancelled")
    started = time.monotonic()
    try:
        with pytest.raises(PEError, match="confirm.*exit"):
            engine.start(interrupt)
        elapsed = time.monotonic() - started
        assert process.waits and max(process.waits) <= 0.25
        assert elapsed < 1.2, "Startup cancellation waited for long process cleanup"
        assert engine.process is process and engine._cleanup_failed
        assert engine._log is None
        with pytest.raises(PEError, match="awaiting cleanup"):
            engine.start()
    finally:
        process.exited = True
        engine.close()
