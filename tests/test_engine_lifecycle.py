from pathlib import Path

import pytest
from types import SimpleNamespace

import flux3_pe.engine as backend
from flux3_pe.protocol import PEError


@pytest.fixture
def engine_fixture(monkeypatch):
    backend.release_model()
    backend._results.clear()

    class FakeEngine:
        start_error = None
        complete_error = None
        close_error = None
        caption = "A red ball on white."
        calls = 0
        instances = []

        def __init__(self, spec):
            self.spec = spec
            self.closed = False
            self.process = None
            self.instances.append(self)

        def start(self, interrupt):
            interrupt()
            if self.start_error:
                raise self.start_error
            self.process = SimpleNamespace(poll=lambda: None)

        def complete(self, payload, interrupt):
            interrupt()
            type(self).calls += 1
            if self.complete_error:
                raise self.complete_error
            return {"choices": [{"message": {"content": self.caption}, "finish_reason": "stop"}]}

        def close(self):
            self.closed = True
            self.process = None
            if self.close_error:
                raise self.close_error

    monkeypatch.setattr(backend, "LlamaEngine", FakeEngine)
    yield FakeEngine
    # A failing cleanup must not strand the test suite behind a locked mutex.
    if backend._lock.locked():
        backend._lock.release()
    backend._engine = None
    backend._results.clear()


def model():
    return backend.ModelSpec(Path("model"), Path("vision"), Path("server"), "fixture")


@pytest.mark.parametrize("phase", ["start", "complete", "invalid_output"])
def test_failed_request_releases_even_when_keep_loaded(engine_fixture, phase):
    if phase == "start":
        engine_fixture.start_error = PEError("startup failed")
    elif phase == "complete":
        engine_fixture.complete_error = PEError("backend failed")
    else:
        engine_fixture.caption = ""
    with pytest.raises(PEError):
        backend.enhance(model(), "A ball", keep_loaded=True)
    assert backend._engine is None
    assert engine_fixture.instances[0].closed
    assert not backend._results
    assert not backend._lock.locked()


def test_cache_hit_can_release_previously_resident_engine(engine_fixture):
    first, _ = backend.enhance(model(), "A ball", keep_loaded=True)
    assert backend._engine is not None
    second, metadata = backend.enhance(model(), "A ball", keep_loaded=False)
    assert first == second and metadata["cache_hit"]
    assert engine_fixture.calls == 1
    assert engine_fixture.instances[0].closed
    assert backend._engine is None


def test_cancel_while_waiting_does_not_release_another_requests_lock(engine_fixture):
    backend._lock.acquire()
    def interrupt():
        raise InterruptedError("cancel waiting request")
    with pytest.raises(InterruptedError):
        backend.enhance(model(), "A ball", interrupt=interrupt)
    assert backend._lock.locked()
    assert not engine_fixture.instances
    backend._lock.release()


def test_cleanup_failure_does_not_strand_request_mutex(engine_fixture):
    engine_fixture.close_error = RuntimeError("simulated process cleanup error")
    with pytest.raises(RuntimeError, match="cleanup"):
        backend.enhance(model(), "A ball", use_cache=False)
    assert not backend._lock.locked()
    engine_fixture.close_error = None
    caption, _ = backend.enhance(model(), "A second ball", use_cache=False)
    assert caption == "A red ball on white."
    assert engine_fixture.calls == 2


def test_subprocess_spawn_failure_closes_startup_log(tmp_path, monkeypatch):
    engine = backend.LlamaEngine(model())
    monkeypatch.setenv("FLUX3_PE_LOG_DIR", str(tmp_path))
    def fail_spawn(*args, **kwargs):
        raise OSError("invalid executable")
    monkeypatch.setattr(backend.subprocess, "Popen", fail_spawn)
    try:
        with pytest.raises(OSError, match="invalid executable"):
            engine.start()
        assert engine._log is None
        assert engine.process is None
    finally:
        engine.close()


def test_before_load_runs_on_miss_but_never_on_cached_result(engine_fixture):
    calls = []
    callback = lambda: calls.append("unload")
    backend.enhance(model(), "A ball", before_load=callback)
    assert calls == ["unload"]
    _, metadata = backend.enhance(model(), "A ball", before_load=callback)
    assert metadata["cache_hit"]
    assert calls == ["unload"]


def test_before_load_not_repeated_for_successfully_resident_engine(engine_fixture):
    calls = []
    callback = lambda: calls.append("unload")
    for prompt in ("A red ball", "A blue ball", "A green ball"):
        backend.enhance(model(), prompt, keep_loaded=True, use_cache=False, before_load=callback)
    assert calls == ["unload"]
    assert len(engine_fixture.instances) == 1
    assert not engine_fixture.instances[0].closed
    assert backend._engine is engine_fixture.instances[0]
    backend.release_model()
    assert engine_fixture.instances[0].closed


def test_before_load_failure_releases_engine_and_mutex(engine_fixture):
    def callback():
        raise RuntimeError("unload failed")
    with pytest.raises(RuntimeError, match="unload failed"):
        backend.enhance(model(), "A ball", keep_loaded=True, before_load=callback)
    assert backend._engine is None
    assert engine_fixture.instances[0].closed
    assert not backend._lock.locked()
    assert not backend._results
    caption, _ = backend.enhance(model(), "A second ball", use_cache=False)
    assert caption == "A red ball on white."
