from pathlib import Path
import copy

from PIL import Image
import pytest

import flux3_pe.engine as backend
from flux3_pe.protocol import PEError, has_repetition, prepare_request, recovery_payload


def test_repetition_detection_does_not_treat_long_unique_text_as_a_loop():
    assert has_repetition(("the original roof is already red and the user wants red " * 20))
    assert not has_repetition(" ".join(f"detail_{index}" for index in range(1000)))


def test_layout_recovery_preserves_original_request_and_images():
    request = prepare_request("Move the ball", [Image.new("RGB", (32, 32))], [{"id": "ball", "from": "ref_image_0", "src_bbox": [0.1, 0.1, 0.5, 0.5], "tgt_bbox": [0.2, 0.2, 0.6, 0.6]}])
    original = copy.deepcopy(request.payload())
    retry = recovery_payload(request, 3584)
    assert request.payload() == original
    assert retry["messages"][1]["content"][0] == original["messages"][1]["content"][0]
    assert "<ball>" in retry["messages"][1]["content"][-1]["text"]
    assert retry["presence_penalty"] == 1.0
    assert original["presence_penalty"] == 0.0


def response(text, reason="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": reason}], "usage": {"completion_tokens": 100}}


@pytest.fixture
def fake_backend(monkeypatch):
    backend.release_model()
    backend._results.clear()

    class FakeEngine:
        responses = []
        payloads = []
        closed = False

        def __init__(self, spec):
            self.spec = spec

        def start(self, interrupt):
            interrupt()

        def complete(self, payload, interrupt):
            interrupt()
            self.payloads.append(payload)
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        def close(self):
            self.closed = True

    monkeypatch.setattr(backend, "LlamaEngine", FakeEngine)
    yield FakeEngine
    backend.release_model()
    backend._results.clear()


def spec():
    return backend.ModelSpec(Path("model.gguf"), Path("mmproj.gguf"), Path("server.exe"), "fixture")


def test_repetitive_truncation_recovers_once_and_marks_result(fake_backend):
    fake_backend.responses = [response("the original roof is already red and the user wants red " * 20, "length"), response("Change only the roof to red; preserve the yellow house and background.")]
    caption, metadata = backend.enhance(spec(), "Change the roof", [Image.new("RGB", (32, 32))], use_cache=False)
    assert caption.startswith("Change only")
    assert metadata["recovered"] is True
    assert len(metadata["attempts"]) == 2
    assert fake_backend.payloads[0]["presence_penalty"] == 0.0
    assert fake_backend.payloads[1]["presence_penalty"] == 1.0


def test_genuinely_long_nonrepetitive_output_is_not_rewritten(fake_backend):
    fake_backend.responses = [response(" ".join(f"detail_{index}" for index in range(1000)), "length")]
    with pytest.raises(PEError):
        backend.enhance(spec(), "Describe many things", use_cache=False)
    assert len(fake_backend.payloads) == 1
    assert not backend._results


def test_recovery_is_bounded_and_never_caches_failure(fake_backend):
    fake_backend.responses = [response("", "stop"), response("", "stop")]
    with pytest.raises(PEError):
        backend.enhance(spec(), "A ball")
    assert len(fake_backend.payloads) == 2
    assert not backend._results


def test_strict_mode_and_cancellation_do_not_retry(fake_backend):
    fake_backend.responses = [response("", "stop")]
    with pytest.raises(PEError):
        backend.enhance(spec(), "A ball", recover_invalid=False)
    assert len(fake_backend.payloads) == 1
    fake_backend.responses = [InterruptedError("cancel")]
    with pytest.raises(InterruptedError):
        backend.enhance(spec(), "A ball")
    assert len(fake_backend.payloads) == 2
