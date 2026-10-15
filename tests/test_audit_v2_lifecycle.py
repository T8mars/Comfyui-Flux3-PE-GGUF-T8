"""Owned-process cleanup races use CPU fakes; no user processes are touched."""
from pathlib import Path
from types import SimpleNamespace
import io
import subprocess

import pytest

import flux3_pe.engine as backend
from flux3_pe.protocol import PEError


def spec(revision="audit-close"):
    return backend.ModelSpec(Path("model"), Path("vision"), Path("server"), revision)


class Process:
    def __init__(self, terminate_error=None, waits=None, kill_error=None, exited=False):
        self.terminate_error = terminate_error
        self.kill_error = kill_error
        self.waits = list(waits or [0])
        self.exited = exited
        self.calls = []

    def poll(self):
        return 0 if self.exited else None

    def terminate(self):
        self.calls.append("terminate")
        if self.terminate_error:
            raise self.terminate_error

    def wait(self, timeout):
        self.calls.append("wait")
        result = self.waits.pop(0)
        if isinstance(result, BaseException):
            raise result
        self.exited = True
        return result

    def kill(self):
        self.calls.append("kill")
        if self.kill_error:
            raise self.kill_error


def engine(process):
    value = backend.LlamaEngine(spec())
    value.process = process
    log = io.StringIO()
    value._log = log
    session = SimpleNamespace(close_calls=0)
    def close_session():
        session.close_calls += 1
    session.close = close_session
    value.session.close()
    value.session = session
    return value, log, session


@pytest.mark.parametrize("terminate_error", [ProcessLookupError("race"), PermissionError("blocked")])
def test_terminate_error_still_confirms_exit_and_closes_resources(terminate_error):
    process = Process(terminate_error=terminate_error)
    value, log, session = engine(process)
    value.close()
    assert value.process is None and log.closed and value._log is None
    assert session.close_calls >= 1
    value.close()


def test_wait_timeout_escalates_to_kill_and_confirms_exit():
    process = Process(waits=[subprocess.TimeoutExpired("owned", 10), 0])
    value, log, session = engine(process)
    value.close()
    assert process.calls == ["terminate", "wait", "kill", "wait"]
    assert value.process is None and log.closed and session.close_calls >= 1


@pytest.mark.parametrize("kill_error", [None, PermissionError("cannot kill")])
def test_unconfirmed_exit_keeps_owned_handle_but_closes_resources(kill_error):
    process = Process(waits=[subprocess.TimeoutExpired("owned", 10), subprocess.TimeoutExpired("owned", 10)], kill_error=kill_error)
    value, log, session = engine(process)
    with pytest.raises(PEError, match="confirm|exit|terminat"):
        value.close()
    assert value.process is process and log.closed and value._log is None
    assert session.close_calls >= 1
    process.waits = [0]
    process.kill_error = None
    value.close()
    assert value.process is None


@pytest.mark.parametrize("action", ["release", "switch", "request_failure"])
def test_failed_global_cleanup_preserves_only_retryable_engine_handle(monkeypatch, action):
    backend.release_model()
    backend._results.clear()
    process = Process(waits=[subprocess.TimeoutExpired("owned", 10)] * 4, kill_error=PermissionError("cannot kill"))
    value, log, session = engine(process)
    value.start = lambda interrupt: None
    def fail_complete(payload, interrupt):
        raise PEError("request failed")
    value.complete = fail_complete
    backend._engine = value
    try:
        with pytest.raises(PEError):
            if action == "release":
                backend.release_model()
            else:
                backend.enhance(spec("other" if action == "switch" else "audit-close"), "A ball", use_cache=False)
        assert backend._engine is value and value.process is process
        assert not backend._lock.locked() and log.closed and session.close_calls >= 1
    finally:
        process.waits = [0]
        process.kill_error = None
        backend.release_model()
        backend._results.clear()


def test_start_reclaims_log_and_session_of_previously_exited_process(tmp_path, monkeypatch):
    value = backend.LlamaEngine(spec())
    old_log = io.StringIO()
    value._log = old_log
    value.process = Process(exited=True)
    old_session = value.session
    closed = []
    monkeypatch.setattr(old_session, "close", lambda: closed.append(True))
    monkeypatch.setenv("FLUX3_PE_LOG_DIR", str(tmp_path))
    def fail_spawn(*args, **kwargs):
        assert old_log.closed, "Restart overwrote an unclosed prior process log"
        assert closed, "Restart did not release old session pools"
        raise OSError("intentional stop before process creation")
    monkeypatch.setattr(backend.subprocess, "Popen", fail_spawn)
    with pytest.raises(OSError, match="intentional stop"):
        value.start()
    assert value._log is None and value.process is None
