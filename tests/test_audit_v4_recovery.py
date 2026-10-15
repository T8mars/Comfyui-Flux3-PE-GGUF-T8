"""Recovery must keep explicit copied-object identities and remain bounded."""
import copy
import json
from pathlib import Path

from PIL import Image
import pytest

import flux3_pe.engine as backend
from flux3_pe.protocol import PEError, prepare_request, recovery_payload


BOX = {"id": "red_box", "desc": "the red rectangle", "from": "ref_image_0", "src_bbox": [0.208, 0.125, 0.69, 0.407], "tgt_bbox": [0.208, 0.15, 0.69, 0.432]}


def test_layout_retry_repeats_object_meaning_and_source_without_mutating_request():
    request = prepare_request("Copy the red rectangle", [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32), "blue")], [BOX])
    original = copy.deepcopy(request.payload())
    retry = recovery_payload(request, 3584)
    cue = retry["messages"][-1]["content"][-1]["text"]
    assert '"id": "red_box", "from": "ref_image_0", "desc": "the red rectangle"' in cue
    assert "caption text" in cue and "target-origin alias" in cue
    assert "override the default" in retry["messages"][0]["content"]
    assert json.dumps(request.locked, ensure_ascii=False) in retry["messages"][0]["content"]
    assert request.payload() == original
    assert retry["messages"][-1]["content"][:2] == original["messages"][-1]["content"][:2]


@pytest.mark.parametrize("second_valid", [False, True])
def test_copied_object_alias_is_never_silently_accepted_or_retried_more_than_once(monkeypatch, second_valid):
    backend.release_model()
    backend._results.clear()
    locked = {"id": "red_box", "from": "ref_image_0", "src_bbox": [208, 125, 690, 407], "tgt_bbox": [208, 150, 690, 432], "desc": "the red rectangle"}
    alias = {"id": "rectangle_1", "from": "target", "src_bbox": None, "tgt_bbox": [208, 150, 690, 432], "desc": "a red rectangle"}
    invalid = "Place <rectangle_1> on the lawn. " + json.dumps([locked, alias])
    valid = "Place <red_box> on the lawn. " + json.dumps([locked])
    payloads = []
    replies = [invalid, valid if second_valid else invalid]
    class Fake:
        def __init__(self, spec):
            self.spec = spec
        def start(self, interrupt):
            interrupt()
        def complete(self, payload, interrupt):
            payloads.append(payload)
            return {"choices": [{"finish_reason": "stop", "message": {"content": replies.pop(0)}}]}
        def close(self):
            pass
    monkeypatch.setattr(backend, "LlamaEngine", Fake)
    spec = backend.ModelSpec(Path("model"), Path("vision"), Path("server"), "alias-recovery")
    try:
        if second_valid:
            caption, details = backend.enhance(spec, "Copy the red rectangle", [Image.new("RGB", (32, 32))], [BOX])
            assert "<red_box>" in caption and "<rectangle_1>" not in caption
            assert details["recovered"] and len(details["attempts"]) == 2
        else:
            with pytest.raises(PEError, match="cite every box id"):
                backend.enhance(spec, "Copy the red rectangle", [Image.new("RGB", (32, 32))], [BOX])
            assert not backend._results
        assert len(payloads) == 2
    finally:
        backend.release_model()
        backend._results.clear()
