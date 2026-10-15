import json

import pytest

from flux3_pe.protocol import PEError, finish_caption, prepare_request


def request():
    return prepare_request('A ball and a flower; a sign reading "<SALE>".', boxes=[{"id": "ball", "bbox": [0, 0, 0.5, 0.5], "desc": "a red ball"}])


@pytest.mark.parametrize("added", [
    {"id": "flower", "bbox": [900, 0, 100, 100], "desc": "a flower"},
    {"id": "flower", "from": "target", "src_bbox": [0, 0, 100, 100], "tgt_bbox": [200, 200, 400, 400], "desc": "a flower"},
])
def test_dropped_invalid_added_row_does_not_leave_a_dangling_layout_binding(added):
    text = 'A red ball <ball> beside a flower <flower>. ' + json.dumps([
        {"id": "ball", "bbox": [0, 0, 500, 500], "desc": "a red ball"}, added,
    ])
    with pytest.raises(PEError) as error:
        finish_caption(text, request(), 'stop')
    assert error.value.kind == 'layout'


def test_upstream_reconciled_identity_rename_still_succeeds():
    text = 'A red ball <ball_1>. ' + json.dumps([{"id": "ball_1", "bbox": [0, 0, 500, 500], "desc": "a red ball"}])
    assert '<ball>' in finish_caption(text, request(), 'stop')


def test_literal_angle_bracket_text_is_not_misclassified_as_a_layout_binding():
    text = 'A red ball <ball> beside a sign reading "<SALE>". ' + json.dumps([{"id": "ball", "bbox": [0, 0, 500, 500], "desc": "a red ball"}])
    assert '"<SALE>"' in finish_caption(text, request(), 'stop')


@pytest.mark.parametrize("added", [
    {"id": "flower", "bbox": [0, 0, 0, 100], "desc": "flower"},
    {"id": "flower", "bbox": [0, 0, 100, 100], "desc": "  "},
    {"id": "flower", "from": "ref_image_999", "src_bbox": [0, 0, 100, 100], "tgt_bbox": [200, 200, 400, 400], "desc": "flower"},
    {"id": "flower", "from": "ref_image_bad", "src_bbox": [0, 0, 100, 100], "tgt_bbox": [200, 200, 400, 400], "desc": "flower"},
    {"id": "", "bbox": [800, 800, 900, 900], "desc": "flower"},
])
def test_declared_cited_invalid_added_rows_fail_as_layout_errors(added):
    text = 'A red ball <ball> and a flower <' + added['id'] + '>. ' + json.dumps([
        {"id": "ball", "bbox": [0, 0, 500, 500], "desc": "ball"}, added,
    ])
    with pytest.raises(PEError) as error:
        finish_caption(text, request(), 'stop')
    assert error.value.kind == 'layout'


def test_uncited_invalid_row_can_be_discarded_without_forbidding_literal_text():
    text = 'A red ball <ball> by a sign "<SALE>" and "<3". ' + json.dumps([
        {"id": "ball", "bbox": [0, 0, 500, 500], "desc": "ball"},
        {"id": "unused", "bbox": [800, 800, 700, 700], "desc": "invalid"},
    ])
    output = finish_caption(text, request(), 'stop')
    assert '"<SALE>"' in output and '"<3"' in output
    assert '"unused"' not in output


def test_duplicate_declared_identity_reconciles_to_one_authoritative_binding():
    text = 'A red ball <ball>. ' + json.dumps([
        {"id": "ball", "bbox": [0, 0, 500, 500], "desc": "ball"},
        {"id": "ball", "bbox": [700, 700, 900, 900], "desc": "duplicate"},
    ])
    from flux3_pe.bbox import split
    _, rows = split(finish_caption(text, request(), 'stop'))
    assert len(rows) == 1 and rows[0]['id'] == 'ball'
    assert rows[0]['bbox'] == [0, 0, 500, 500]
