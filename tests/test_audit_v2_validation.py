"""Reject invalid request controls before unload, spawn or resident cleanup."""
from pathlib import Path

import pytest

import flux3_pe.engine as backend
from flux3_pe.protocol import prepare_request


INVALID_BUDGETS = [0, 3585, True, False, 1.0, 1.5, "1", None]


@pytest.fixture
def tracked_backend(monkeypatch):
    backend.release_model()
    backend._results.clear()
    effects = []

    class FakeEngine:
        def __init__(self, spec):
            self.spec = spec
            self.process = None
            effects.append("construct")

        def start(self, interrupt):
            interrupt()
            effects.append("start")

        def complete(self, payload, interrupt):
            effects.append(("complete", payload["max_tokens"]))
            return {"choices": [{"message": {"content": "A red ball."}, "finish_reason": "stop"}]}

        def close(self):
            effects.append("close")

    monkeypatch.setattr(backend, "LlamaEngine", FakeEngine)
    yield effects
    backend.release_model()
    backend._results.clear()


def spec():
    return backend.ModelSpec(Path("model"), Path("vision"), Path("server"), "validation")


@pytest.mark.parametrize("budget", INVALID_BUDGETS)
def test_payload_requires_real_integer_budget_in_supported_range(budget):
    with pytest.raises(ValueError, match="max_tokens"):
        prepare_request("A ball").payload(budget)


@pytest.mark.parametrize("budget", INVALID_BUDGETS)
def test_invalid_budget_has_no_resource_effects(tracked_backend, budget):
    with pytest.raises(ValueError, match="max_tokens"):
        backend.enhance(spec(), "A ball", max_tokens=budget, before_load=lambda: tracked_backend.append("unload"))
    assert tracked_backend == []
    assert backend._engine is None and not backend._results


@pytest.mark.parametrize("budget", INVALID_BUDGETS)
def test_invalid_request_does_not_unload_an_existing_resident(tracked_backend, budget):
    backend.enhance(spec(), "A first ball", keep_loaded=True)
    resident = backend._engine
    tracked_backend.clear()
    with pytest.raises(ValueError, match="max_tokens"):
        backend.enhance(spec(), "A second ball", max_tokens=budget, before_load=lambda: tracked_backend.append("unload"))
    assert tracked_backend == [] and backend._engine is resident


@pytest.mark.parametrize("budget", [1, 3584])
def test_valid_boundaries_reach_backend_unchanged(tracked_backend, budget):
    assert prepare_request("A ball").payload(budget)["max_tokens"] == budget
    backend.enhance(spec(), "A ball", max_tokens=budget)
    assert ("complete", budget) in tracked_backend


def test_already_cancelled_request_cannot_unload_or_construct_backend(tracked_backend):
    def interrupt():
        raise InterruptedError("cancelled before acquisition")
    with pytest.raises(InterruptedError):
        backend.enhance(spec(), "A ball", interrupt=interrupt, before_load=lambda: tracked_backend.append("unload"))
    assert tracked_backend == [] and backend._engine is None
