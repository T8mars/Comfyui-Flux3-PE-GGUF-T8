"""Real caller threads, coordinated with Events rather than timing sleeps."""
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

import flux3_pe.engine as backend
from flux3_pe.protocol import PEError


@pytest.fixture
def callers(monkeypatch):
    backend.release_model()
    backend._results.clear()
    entered = threading.Event()
    proceed = threading.Event()
    workers = []

    class FakeEngine:
        instances = []
        calls = []
        active = 0
        max_active = 0
        closes = 0
        deny_close = False

        def __init__(self, spec):
            self.spec = spec
            self.process = None
            self.instances.append(self)

        def start(self, interrupt):
            interrupt()
            self.process = SimpleNamespace(poll=lambda: None)

        def complete(self, payload, interrupt):
            type(self).active += 1
            type(self).max_active = max(self.max_active, self.active)
            type(self).calls.append(payload["messages"][-1]["content"])
            entered.set()
            try:
                assert proceed.wait(2), "Caller fixture gate was not released"
                interrupt()
                return {"choices": [{"message": {"content": "A red ball."}, "finish_reason": "stop"}]}
            finally:
                type(self).active -= 1

        def close(self):
            type(self).closes += 1
            if self.deny_close:
                raise PEError("cannot confirm cleanup")
            self.process = None

    monkeypatch.setattr(backend, "LlamaEngine", FakeEngine)

    def launch(function):
        outcome = {"done": threading.Event()}
        def target():
            try:
                outcome["value"] = function()
            except BaseException as error:
                outcome["error"] = error
            finally:
                outcome["done"].set()
        worker = threading.Thread(target=target, name="audit-enhance-caller")
        workers.append(worker)
        worker.start()
        outcome["worker"] = worker
        return outcome

    yield FakeEngine, entered, proceed, launch
    proceed.set()
    for worker in workers:
        worker.join(timeout=3)
        assert not worker.is_alive(), "Real caller thread was stranded"
    FakeEngine.deny_close = False
    backend.release_model()
    backend._results.clear()


def spec(revision="threads"):
    return backend.ModelSpec(Path("model"), Path("vision"), Path("server"), revision)


def result(outcome):
    assert outcome["done"].wait(3), "Caller did not finish"
    outcome["worker"].join(timeout=3)
    assert not outcome["worker"].is_alive()
    assert "error" not in outcome, repr(outcome.get("error"))
    return outcome["value"]


def test_same_request_contention_computes_once_then_hits_cache(callers):
    engine, entered, proceed, launch = callers
    first = launch(lambda: backend.enhance(spec(), "A ball", keep_loaded=True))
    assert entered.wait(2)
    waiting = [threading.Event() for _ in range(3)]
    others = [launch(lambda event=event: backend.enhance(spec(), "A ball", keep_loaded=True, interrupt=event.set)) for event in waiting]
    assert all(event.wait(2) for event in waiting)
    assert len(engine.calls) == len(engine.instances) == 1
    proceed.set()
    results = [result(first)] + [result(other) for other in others]
    assert [metadata["cache_hit"] for _, metadata in results] == [False, True, True, True]
    assert engine.max_active == 1 and len(engine.calls) == 1


def test_different_requests_are_serialized_and_not_cross_cached(callers):
    engine, entered, proceed, launch = callers
    first = launch(lambda: backend.enhance(spec(), "A red ball", keep_loaded=True))
    assert entered.wait(2)
    waiting = [threading.Event() for _ in range(2)]
    others = [launch(lambda index=index, event=event: backend.enhance(spec(), f"A ball {index}", keep_loaded=True, interrupt=event.set)) for index, event in enumerate(waiting)]
    assert all(event.wait(2) for event in waiting)
    proceed.set()
    results = [result(first)] + [result(other) for other in others]
    assert all(not metadata["cache_hit"] for _, metadata in results)
    assert set(engine.calls) == {"A red ball", "A ball 0", "A ball 1"}
    assert engine.max_active == 1 and len(engine.instances) == 1


def test_waiting_cancel_does_not_close_or_release_active_request(callers):
    engine, entered, proceed, launch = callers
    first = launch(lambda: backend.enhance(spec(), "A ball", keep_loaded=True))
    assert entered.wait(2)
    waiting = threading.Event()
    def interrupt():
        waiting.set()
        raise InterruptedError("waiting caller cancelled")
    second = launch(lambda: backend.enhance(spec(), "A second ball", interrupt=interrupt))
    assert waiting.wait(2) and second["done"].wait(2)
    second["worker"].join(2)
    assert isinstance(second.get("error"), InterruptedError)
    assert backend._lock.locked() and engine.active == 1 and engine.closes == 0
    proceed.set()
    assert result(first)[1]["cache_hit"] is False
    assert len(engine.calls) == 1


def test_concurrent_release_waits_for_active_request_and_then_closes(callers):
    engine, entered, proceed, launch = callers
    first = launch(lambda: backend.enhance(spec(), "A ball", keep_loaded=True))
    assert entered.wait(2)
    release_entered = threading.Event()
    def release():
        release_entered.set()
        backend.release_model()
    cleanup = launch(release)
    assert release_entered.wait(2)
    assert not cleanup["done"].is_set() and engine.active == 1 and engine.closes == 0
    proceed.set()
    result(first)
    result(cleanup)
    assert backend._engine is None and engine.closes == 1


def test_unconfirmed_cleanup_blocks_new_backend_until_explicit_retry(callers):
    engine, entered, proceed, launch = callers
    engine.deny_close = True
    first = launch(lambda: backend.enhance(spec(), "A ball", use_cache=False))
    assert entered.wait(2)
    proceed.set()
    assert first["done"].wait(2)
    assert isinstance(first.get("error"), PEError)
    retained = backend._engine
    assert retained is engine.instances[0] and retained.process is not None
    second = launch(lambda: backend.enhance(spec("new-model"), "A second ball", use_cache=False))
    assert second["done"].wait(2)
    assert isinstance(second.get("error"), PEError)
    assert len(engine.instances) == 1 and backend._engine is retained
    engine.deny_close = False
    cleanup = launch(backend.release_model)
    result(cleanup)
    assert backend._engine is None and retained.process is None
    third = launch(lambda: backend.enhance(spec("new-model"), "A third ball", use_cache=False))
    assert result(third)[0] == "A red ball."
    assert len(engine.instances) == 2 and engine.max_active == 1
