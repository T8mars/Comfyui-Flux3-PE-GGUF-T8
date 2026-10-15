import json

from PIL import Image
import pytest

from flux3_pe.protocol import PEError, finish_caption, parse_boxes, prepare_request


def layout_request():
    return prepare_request("A red ball", [Image.new("RGB", (32, 32))], [{"id": "ball", "bbox": [0, 0, 1, 1]}])


@pytest.mark.parametrize("bad_id", [["ball"], {"name": "ball"}])
def test_malformed_model_identity_is_a_recoverable_layout_error(bad_id):
    text = "A red ball <ball>. " + json.dumps([{"id": bad_id, "bbox": [0, 0, 1000, 1000]}])
    with pytest.raises(PEError) as error:
        finish_caption(text, layout_request(), "stop")
    assert error.value.kind == "layout"


@pytest.mark.parametrize("model_id", [None, "ball"])
def test_reconcile_can_restore_valid_locked_rows_and_drop_bad_or_duplicate_rows(model_id):
    text = "A red ball <ball>. " + json.dumps([
        {"id": model_id, "bbox": [0, 0, 1000, 1000], "desc": "a red ball"},
        {"id": "ball", "bbox": [0, 0, 1000, 1000], "desc": "a red ball"},
    ])
    caption = finish_caption(text, layout_request(), "stop")
    rows = json.loads(caption[caption.index("[{"):])
    assert len(rows) == 1 and rows[0]["id"] == "ball"


def test_added_object_cannot_refer_to_a_reference_image_not_attached():
    text = "A red ball <ball> beside an added flower <flower>. " + json.dumps([
        {"id": "ball", "bbox": [0, 0, 1000, 1000], "desc": "red ball"},
        {"id": "flower", "from": "ref_image_999", "src_bbox": [10, 10, 200, 200], "tgt_bbox": [20, 20, 250, 250], "desc": "a flower"},
    ])
    with pytest.raises(PEError) as error:
        finish_caption(text, layout_request(), "stop")
    assert error.value.kind == "layout"


@pytest.mark.parametrize("deletion", [False, True])
def test_new_and_deleted_objects_preserve_valid_provenance(deletion):
    row = {"id": "ball", "desc": "a red ball", "from": "ref_image_0" if deletion else "target", "src_bbox": [0, 0, 0.5, 0.5] if deletion else None, "tgt_bbox": None if deletion else [0.1, 0.1, 0.6, 0.6]}
    request = prepare_request("Remove the ball" if deletion else "Add a ball", [Image.new("RGB", (32, 32))], [row])
    model_row = {"id": "ball", "desc": "a red ball", "from": row["from"], "src_bbox": [0, 0, 500, 500] if deletion else None, "tgt_bbox": None if deletion else [100, 100, 600, 600]}
    text = "Remove the ball <ball>. " if deletion else "A newly added ball <ball>. "
    result = finish_caption(text + json.dumps([model_row]), request, "stop")
    output = json.loads(result[result.index("[{"):])[0]
    assert output == model_row


def test_inline_canvas_table_is_detected_and_keeps_1000_scale():
    row = {"id": "ball", "from": "ref_image_0", "src_bbox": [10, 20, 500, 600], "tgt_bbox": [100, 200, 700, 800], "desc": "a ball"}
    request = prepare_request("Move the ball " + json.dumps([row]), [Image.new("RGB", (32, 32))])
    assert request.mode == "i2i_bbox" and request.locked == [row]
    output = finish_caption("Move the ball <ball>. " + json.dumps([row]), request, "stop")
    assert json.loads(output[output.index("[{"):]) == [row]


@pytest.mark.parametrize("origin", ["ref_image_bad", "ref_image_-1", "ref_image_999"])
def test_invalid_inline_provenance_is_rejected_before_backend(origin):
    row = {"id": "ball", "from": origin, "src_bbox": [0, 0, 500, 500], "tgt_bbox": [100, 100, 600, 600]}
    with pytest.raises(ValueError):
        prepare_request("Move the ball " + json.dumps([row]), [Image.new("RGB", (32, 32))])


def test_invalid_inline_target_is_rejected_even_with_valid_source_box():
    row = {"id": "ball", "from": "ref_image_0", "src_bbox": [0, 0, 500, 500], "tgt_bbox": [900, 100, 100, 600]}
    with pytest.raises(ValueError):
        prepare_request("Move the ball " + json.dumps([row]), [Image.new("RGB", (32, 32))])


def test_whitespace_only_optional_boxes_means_no_layout():
    assert parse_boxes("  \n \t", 0) is None
    assert prepare_request("A ball", boxes=" \n").mode == "t2i"


@pytest.mark.parametrize("prompt", ["", "  \n", None])
def test_empty_request_is_rejected(prompt):
    with pytest.raises(ValueError, match="empty"):
        prepare_request(prompt)


@pytest.mark.parametrize("canvas", [(0, 512), (512, 0), (-1, 512)])
def test_single_sided_or_negative_canvas_is_rejected(canvas):
    with pytest.raises(ValueError, match="positive"):
        prepare_request("A ball", canvas=canvas)


@pytest.mark.parametrize("tokens", [0, 3585])
def test_output_budget_outside_supported_limits_is_rejected(tokens):
    with pytest.raises(ValueError, match="max_tokens"):
        prepare_request("A ball").payload(tokens)
