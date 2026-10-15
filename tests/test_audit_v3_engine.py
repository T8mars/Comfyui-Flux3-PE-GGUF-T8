"""An already cancelled call must not submit a new backend generation."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import flux3_pe.engine as backend
from flux3_pe.engine import LlamaEngine, ModelSpec


def test_already_cancelled_complete_never_sends_http_or_starts_worker():
    engine = LlamaEngine(ModelSpec(Path("model"), Path("vision"), Path("server"), "pre-cancel"))
    engine.url = "http://127.0.0.1:1"
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.ok = True
    response.json.return_value = {"choices": []}
    post = Mock(return_value=response)
    engine.session.close()
    engine.session = SimpleNamespace(post=post, close=Mock())
    def interrupt():
        raise InterruptedError("already cancelled")
    try:
        with pytest.raises(InterruptedError, match="already cancelled"):
            engine.complete({"messages": []}, interrupt)
        assert post.call_count == 0, "A pre-cancelled request still submitted generation"
        assert engine._request_worker is None
    finally:
        engine.close()


def test_already_cancelled_start_never_spawns_backend(tmp_path, monkeypatch):
    engine = LlamaEngine(ModelSpec(Path("model"), Path("vision"), Path("server"), "pre-cancel-start"))
    spawn = Mock(side_effect=AssertionError("Must not spawn after cancellation"))
    monkeypatch.setattr(backend.subprocess, "Popen", spawn)
    monkeypatch.setenv("FLUX3_PE_LOG_DIR", str(tmp_path))
    def interrupt():
        raise InterruptedError("already cancelled")
    try:
        with pytest.raises(InterruptedError, match="already cancelled"):
            engine.start(interrupt)
        assert spawn.call_count == 0 and engine.process is None
        assert engine._log is None
    finally:
        engine.close()


def test_http_thread_start_failure_closes_owned_backend_without_join_error(monkeypatch):
    engine = LlamaEngine(ModelSpec(Path("model"), Path("vision"), Path("server"), "thread-start-failure"))
    process = Mock()
    process.poll.return_value = None
    process.wait.return_value = 0
    engine.process = process
    def cannot_start(worker):
        raise RuntimeError("thread resources unavailable")
    monkeypatch.setattr(backend.threading.Thread, "start", cannot_start)
    try:
        with pytest.raises(RuntimeError, match="thread resources unavailable"):
            engine.complete({"messages": []})
        process.terminate.assert_called_once()
        assert engine.process is None and engine._request_worker is None
    finally:
        engine.close()
