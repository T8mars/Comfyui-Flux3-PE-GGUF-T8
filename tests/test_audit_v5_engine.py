"""Cached text cannot bypass a retained backend whose cleanup failed."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import flux3_pe.engine as backend
from flux3_pe.protocol import PEError


@pytest.fixture
def retained_backend(monkeypatch):
    backend.release_model()
    backend._results.clear()

    class FakeEngine:
        deny_close = True
        instances = []
        calls = 0
        closes = 0

        def __init__(self, spec):
            self.spec = spec
            self.process = None
            self._cleanup_failed = False
            self.instances.append(self)

        def start(self, interrupt):
            interrupt()
            if self._cleanup_failed:
                raise PEError("previous request is awaiting cleanup")
            self.process = SimpleNamespace(poll=lambda: None)

        def complete(self, payload, interrupt):
            interrupt()
            type(self).calls += 1
            return {"choices": [{"message": {"content": "A red ball on white."}, "finish_reason": "stop"}]}

        def close(self):
            type(self).closes += 1
            self._cleanup_failed = self.deny_close
            if self.deny_close:
                raise PEError("Cannot confirm owned llama-server exit")
            self.process = None

    monkeypatch.setattr(backend, "LlamaEngine", FakeEngine)
    yield FakeEngine
    FakeEngine.deny_close = False
    backend.release_model()
    backend._results.clear()


def spec(name="retained-cache"):
    return backend.ModelSpec(Path("model"), Path("vision"), Path("server"), name)


@pytest.mark.parametrize("keep_loaded", [False, True])
def test_cached_result_requires_explicit_release_after_unconfirmed_exit(retained_backend, keep_loaded):
    with pytest.raises(PEError, match="confirm"):
        backend.enhance(spec(), "A red ball")
    retained = backend._engine
    assert retained._cleanup_failed and retained.process is not None
    assert not backend._lock.locked()

    with pytest.raises(PEError, match="cleanup"):
        backend.enhance(spec(), "A red ball", keep_loaded=keep_loaded)
    assert retained_backend.calls == retained_backend.closes == 1
    assert len(retained_backend.instances) == 1 and backend._engine is retained
    assert not backend._lock.locked()

    retained_backend.deny_close = False
    backend.release_model()
    assert backend._engine is None
    caption, metadata = backend.enhance(spec(), "A red ball", keep_loaded=keep_loaded)
    assert caption == "A red ball on white." and metadata["cache_hit"]
    assert retained_backend.calls == 1 and len(retained_backend.instances) == 1


def test_another_models_cached_result_cannot_hide_failed_owner(retained_backend):
    retained_backend.deny_close = False
    backend.enhance(spec("A"), "A red ball")
    retained_backend.deny_close = True
    with pytest.raises(PEError, match="confirm"):
        backend.enhance(spec("B"), "A blue ball")
    retained = backend._engine
    with pytest.raises(PEError, match="cleanup"):
        backend.enhance(spec("A"), "A red ball", keep_loaded=True)
    assert backend._engine is retained and retained._cleanup_failed
    assert retained_backend.calls == 2 and len(retained_backend.instances) == 2
    assert not backend._lock.locked()
