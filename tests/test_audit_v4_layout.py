import json

from PIL import Image
import pytest

from flux3_pe.bbox import split
from flux3_pe.protocol import PEError, finish_caption, prepare_request


def edit_request(origin="ref_image_0"):
    row = {"id": "ball", "from": origin, "src_bbox": [0.1, 0.1, 0.5, 0.5] if origin != "target" else None,
           "tgt_bbox": [0.1, 0.1, 0.5, 0.5], "desc": "a red ball"}
    return prepare_request("Move the ball from the first reference", [Image.new("RGB", (32, 32))] * 2, [row])


@pytest.mark.parametrize("origin", ["ref_image_1", "target", None])
def test_iou_identity_alias_cannot_cross_reference_provenance(origin):
    row = {"id": "cat", "bbox": [100, 100, 500, 500], "desc": "a black cat"} if origin is None else {
        "id": "cat", "from": origin, "src_bbox": [100, 100, 500, 500] if origin != "target" else None,
        "tgt_bbox": [100, 100, 500, 500], "desc": "a black cat"}
    with pytest.raises(PEError) as error:
        finish_caption("A black cat <cat>. " + json.dumps([row]), edit_request(), "stop")
    assert error.value.kind == "layout"


@pytest.mark.parametrize("origin", ["ref_image_0", "target"])
def test_iou_identity_alias_preserves_matching_edit_provenance(origin):
    row = {"id": "ball_1", "from": origin, "src_bbox": [100, 100, 500, 500] if origin != "target" else None,
           "tgt_bbox": [100, 100, 500, 500], "desc": "a glossy red ball"}
    caption = finish_caption("A glossy red ball <ball_1>. " + json.dumps([row]), edit_request(origin), "stop")
    prose, rows = split(caption)
    assert "<ball>" in prose and "<ball_1>" not in prose
    assert rows[0]["from"] == origin and rows[0]["desc"] == "a glossy red ball"


def test_explicit_locked_identity_keeps_authoritative_edit_provenance():
    row = {"id": "ball", "from": "ref_image_1", "src_bbox": [100, 100, 500, 500], "tgt_bbox": [100, 100, 500, 500], "desc": "a glossy red ball"}
    _, rows = split(finish_caption("A glossy red ball <ball>. " + json.dumps([row]), edit_request(), "stop"))
    assert rows[0]["from"] == "ref_image_0"


def test_regular_t2i_iou_alias_keeps_original_supported_behavior():
    request = prepare_request("A red ball", boxes=[{"id": "ball", "bbox": [0.1, 0.1, 0.5, 0.5]}])
    row = {"id": "ball_1", "bbox": [100, 100, 500, 500], "desc": "a red ball"}
    assert "<ball>" in finish_caption("A red ball <ball_1>. " + json.dumps([row]), request, "stop")


@pytest.mark.parametrize("target", [None, [600, 600, 900, 900]])
def test_moved_reference_identity_matches_source_coordinates(target):
    request = prepare_request("Move the ball", [Image.new("RGB", (32, 32))], [{
        "id": "ball", "from": "ref_image_0", "src_bbox": [0.1, 0.1, 0.5, 0.5],
        "tgt_bbox": [0.6, 0.6, 0.9, 0.9], "desc": "a red ball"}])
    row = {"id": "ball_1", "from": "ref_image_0", "src_bbox": [100, 100, 500, 500],
           "tgt_bbox": target, "desc": "a red ball"}
    prose, rows = split(finish_caption("Move <ball_1>. " + json.dumps([row]), request, "stop"))
    assert "<ball>" in prose and "<ball_1>" not in prose
    assert rows[0]["src_bbox"] == [100, 100, 500, 500]
    assert rows[0]["tgt_bbox"] == [600, 600, 900, 900]


@pytest.mark.parametrize("source", [None, [600, 600, 900, 900]])
def test_target_overlap_cannot_identify_a_different_source_object(source):
    row = {"id": "cat", "from": "ref_image_0", "src_bbox": source,
           "tgt_bbox": [100, 100, 500, 500], "desc": "a black cat"}
    with pytest.raises(PEError):
        finish_caption("A black cat <cat>. " + json.dumps([row]), edit_request(), "stop")
