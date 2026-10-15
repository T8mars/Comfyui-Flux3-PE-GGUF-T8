"""Layout prose and table must cite the same set of object identities."""
import json

from PIL import Image
import pytest

from flux3_pe.protocol import PEError, finish_caption, prepare_request


def request(edit=False):
    row = {"id": "obj_1", "desc": "a red ball", "bbox": [0.1, 0.1, 0.4, 0.4]}
    if edit:
        row = {"id": "obj_1", "from": "ref_image_0", "src_bbox": [0.1, 0.1, 0.4, 0.4], "tgt_bbox": [0.1, 0.2, 0.4, 0.5], "desc": "a red ball"}
    return prepare_request("Move a red ball" if edit else "A red ball", [Image.new("RGB", (32, 32))] if edit else [], [row])


@pytest.mark.parametrize("edit", [False, True])
def test_layout_rejects_object_citation_without_any_declared_table_row(edit):
    prepared = request(edit)
    text = "A red ball <obj_1> stands beside a lamp <lamp_2>. " + json.dumps(prepared.locked)
    with pytest.raises(PEError, match="missing.*box|undeclared|unknown") as failure:
        finish_caption(text, prepared, "stop")
    assert failure.value.kind == "layout" and failure.value.partial_text == text


def test_editing_accepts_attached_reference_image_citation():
    prepared = request(True)
    text = "Move <obj_1> from <ref_image_0> slightly to the right. " + json.dumps(prepared.locked)
    assert "<ref_image_0>" in finish_caption(text, prepared, "stop")


@pytest.mark.parametrize("edit", [False, True])
def test_layout_rejects_unattached_reference_image_citation(edit):
    prepared = request(edit)
    text = "Move <obj_1> from <ref_image_9> slightly to the right. " + json.dumps(prepared.locked)
    with pytest.raises(PEError, match="missing reference|unknown|undeclared") as failure:
        finish_caption(text, prepared, "stop")
    assert failure.value.kind == "layout"


def test_valid_added_object_citation_and_quoted_non_id_angle_text_remain_valid():
    prepared = request()
    added = {"id": "lamp_2", "bbox": [100, 500, 600, 900], "desc": "a brass lamp bearing the text \"< 5\""}
    text = 'A red ball <obj_1> and a lamp <lamp_2> bearing "< 5". ' + json.dumps([*prepared.locked, added])
    assert '"id": "lamp_2"' in finish_caption(text, prepared, "stop")


def test_quoted_on_image_angle_text_is_not_an_object_citation():
    prepared = request()
    text = 'A red ball <obj_1> bearing "<SALE>" and "<ref_image_9>". ' + json.dumps(prepared.locked)
    assert '"<SALE>"' in finish_caption(text, prepared, "stop")


def test_quoted_locked_id_cannot_substitute_for_an_annotation_citation():
    prepared = request()
    text = 'A sign reading "<obj_1>". ' + json.dumps(prepared.locked)
    with pytest.raises(PEError, match="cite every box id") as failure:
        finish_caption(text, prepared, "stop")
    assert failure.value.kind == "layout"


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("explicit_null", [False, True])
def test_reference_origin_requires_a_non_null_source_box(inline, explicit_null):
    row = {"id": "ball", "from": "ref_image_0", "tgt_bbox": [100, 200, 400, 500] if inline else [0.1, 0.2, 0.4, 0.5], "desc": "a red ball"}
    if explicit_null:
        row["src_bbox"] = None
    with pytest.raises(ValueError, match="source|src_bbox"):
        prepare_request("Move a ball " + (json.dumps([row]) if inline else ""), [Image.new("RGB", (32, 32))], None if inline else [row])


@pytest.mark.parametrize("origin", ["ref_image_00", "ref_image_٠", "ref_image_²"])
def test_reference_origin_must_match_the_exact_attachment_token(origin):
    row = {"id": "ball", "from": origin, "src_bbox": [0.1, 0.1, 0.4, 0.4], "tgt_bbox": None}
    with pytest.raises(ValueError, match="reference"):
        prepare_request("Remove a ball", [Image.new("RGB", (32, 32))], [row])


@pytest.mark.parametrize("origin", ["ref_image_00", "ref_image_٠", "ref_image_²"])
def test_added_reference_row_must_match_the_exact_attachment_token(origin):
    prepared = request(True)
    added = {"id": "lamp_2", "from": origin, "src_bbox": [100, 100, 400, 400], "tgt_bbox": [100, 500, 400, 900], "desc": "a brass lamp"}
    text = "Move <obj_1> and <lamp_2>. " + json.dumps([*prepared.locked, added])
    with pytest.raises(PEError) as failure:
        finish_caption(text, prepared, "stop")
    assert failure.value.kind == "layout"
