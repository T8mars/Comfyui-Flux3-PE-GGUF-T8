"""Independent second-audit coverage for result ownership and bounded caching."""
from pathlib import Path

from PIL import Image
import pytest

import flux3_pe.engine as backend
from flux3_pe import __version__
from flux3_pe.identity import SOURCE_IDENTITY


@pytest.fixture
def cached_backend(monkeypatch):
    backend.release_model()
    backend._results.clear()

    class FakeEngine:
        calls = 0

        def __init__(self, spec):
            self.spec = spec
            self.process = None

        def start(self, interrupt):
            interrupt()

        def complete(self, payload, interrupt):
            interrupt()
            type(self).calls += 1
            return {
                "choices": [{"message": {"content": "A red ball on white."}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 6, "nested": {"truth": True}},
                "timings": {"predicted_n": 6, "nested": {"truth": True}},
            }

        def close(self):
            pass

    monkeypatch.setattr(backend, "LlamaEngine", FakeEngine)
    yield FakeEngine
    backend.release_model()
    backend._results.clear()


def spec():
    return backend.ModelSpec(Path("model"), Path("vision"), Path("server"), "audit-cache")


def run(prompt="A ball", **kwargs):
    return backend.enhance(spec(), prompt, images=[Image.new("RGB", (32, 32))], **kwargs)


def test_first_return_nested_metadata_cannot_mutate_cached_facts(cached_backend):
    _, first = run()
    first["reference_sizes"][0] = (999, 999)
    first["usage"]["nested"]["truth"] = False
    first["attempts"][0]["finish_reason"] = "length"
    first["attempts"][0]["usage"]["completion_tokens"] = -1
    first["timings"]["nested"]["truth"] = False

    _, second = run()
    assert second["reference_sizes"] == [(32, 32)]
    assert second["usage"] == {"completion_tokens": 6, "nested": {"truth": True}}
    assert second["attempts"][0]["finish_reason"] == "stop"
    assert second["timings"] == {}
    assert second["cache_hit"] is True
    assert second["load_seconds"] == second["generation_seconds"] == 0
    assert cached_backend.calls == 1


def test_cache_hit_nested_metadata_cannot_mutate_subsequent_hits(cached_backend):
    run()
    _, second = run()
    second["reference_sizes"].clear()
    second["usage"]["nested"]["truth"] = False
    second["attempts"].append({"finish_reason": "fabricated"})
    second["attempts"][0]["usage"]["completion_tokens"] = -1

    _, third = run()
    assert third["reference_sizes"] == [(32, 32)]
    assert third["usage"] == {"completion_tokens": 6, "nested": {"truth": True}}
    assert third["attempts"] == [{"finish_reason": "stop", "usage": {"completion_tokens": 6, "nested": {"truth": True}}}]
    assert third["cache_hit"] is True and third["timings"] == {}
    assert third["load_seconds"] == third["generation_seconds"] == 0
    assert cached_backend.calls == 1


def test_cache_eviction_stays_at_32_and_evicted_entry_recomputes(cached_backend):
    for index in range(33):
        _, metadata = run(f"A ball {index}")
        metadata["usage"]["completion_tokens"] = -index
    assert len(backend._results) == 32
    assert cached_backend.calls == 33
    assert all(value[1]["usage"]["completion_tokens"] == 6 for value in backend._results.values())
    _, hit = run("A ball 32")
    assert hit["cache_hit"] is True and cached_backend.calls == 33
    _, miss = run("A ball 0")
    assert miss["cache_hit"] is False and cached_backend.calls == 34
    assert len(backend._results) == 32


def test_initial_and_cached_metadata_bind_imported_plugin_identity(cached_backend):
    for hit in (False, True):
        _, details = run()
        assert details["cache_hit"] is hit
        assert details["plugin_version"] == __version__
        assert details["source_identity"] == SOURCE_IDENTITY
