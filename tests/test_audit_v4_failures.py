from pathlib import Path

import pytest

import flux3_pe.engine as backend


@pytest.mark.parametrize("response", [
    None, [], {}, {"choices": []}, {"choices": None},
    {"choices": [None]}, {"choices": [{}]},
    {"choices": [{"finish_reason": "stop"}]},
    {"choices": [{"finish_reason": "stop", "message": None}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": ["bad array content"]}}]},
])
def test_malformed_backend_response_releases_owned_engine_and_never_caches(monkeypatch, response):
    closed = []
    class FakeEngine:
        process = None
        def __init__(self, spec): self.spec = spec
        def start(self, interrupt): interrupt()
        def complete(self, payload, interrupt): return response
        def close(self): closed.append(True)
    monkeypatch.setattr(backend, "LlamaEngine", FakeEngine)
    monkeypatch.setattr(backend, "_engine", None)
    monkeypatch.setattr(backend, "_results", {})
    model = backend.ModelSpec(Path("model"), Path("vision"), Path("server"), "malformed-response")
    with pytest.raises((KeyError, TypeError, IndexError, AttributeError)):
        backend.enhance(model, "A ball", keep_loaded=True)
    assert backend._results == {} and backend._engine is None and closed == [True]
    acquired = backend._lock.acquire(blocking=False)
    assert acquired, "Failure left the shared model mutex locked"
    if acquired: backend._lock.release()
