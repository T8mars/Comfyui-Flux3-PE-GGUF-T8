"""New citation validation participates in the existing bounded recovery path."""
import json

import pytest

import flux3_pe.engine as backend
from flux3_pe.protocol import PEError
from test_recovery import fake_backend, response, spec


@pytest.mark.parametrize("bad_citation", ["lamp_2", "ref_image_9"])
@pytest.mark.parametrize("recovered", [False, True])
def test_unknown_prose_binding_recovers_at_most_once_and_never_caches_failure(fake_backend, bad_citation, recovered, monkeypatch):
    instances = []
    def create(specification):
        value = fake_backend(specification)
        instances.append(value)
        return value
    monkeypatch.setattr(backend, "LlamaEngine", create)
    rows = [{"id": "ball", "bbox": [100, 100, 400, 400], "desc": "a red ball"}]
    valid = "A red ball <ball>. " + json.dumps(rows)
    invalid = f"A red ball <ball> beside <{bad_citation}>. " + json.dumps(rows)
    fake_backend.responses = [response(invalid), response(valid if recovered else invalid)]
    boxes = [{"id": "ball", "bbox": [0.1, 0.1, 0.4, 0.4], "desc": "a red ball"}]
    if recovered:
        caption, metadata = backend.enhance(spec(), "A red ball", boxes=boxes)
        assert bad_citation not in caption
        assert metadata["recovered"] and metadata["recovery_reason"] == "layout"
        assert len(metadata["attempts"]) == 2 and len(backend._results) == 1
    else:
        with pytest.raises(PEError) as failure:
            backend.enhance(spec(), "A red ball", boxes=boxes, keep_loaded=True)
        assert failure.value.kind == "layout" and not backend._results
    assert len(fake_backend.payloads) == 2
    assert backend._engine is None and instances[0].closed
    assert not backend._lock.locked()
