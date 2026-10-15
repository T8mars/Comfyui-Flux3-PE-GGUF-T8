import base64
import io
import json

from PIL import Image
import pytest

from flux3_pe.protocol import PEError, finish_caption, parse_boxes, prepare_reference, prepare_request


@pytest.mark.parametrize("count,side", [(1, 1024), (2, 704), (3, 512), (10, 512)])
def test_reference_budget(count, side):
    data, size = prepare_reference(Image.new("RGB", (2048, 2048)), count)
    assert size == (side, side)
    assert Image.open(io.BytesIO(data)).size == size


def test_alpha_is_composited_on_white():
    data, _ = prepare_reference(Image.new("RGBA", (32, 32), (255, 0, 0, 0)), 1)
    assert Image.open(io.BytesIO(data)).getpixel((16, 16)) == (255, 255, 255)


@pytest.mark.parametrize("images,boxes,mode", [(0, None, "t2i"), (1, None, "i2i"), (0, [{"id": "ball", "bbox": [0.1, 0.2, 0.5, 0.6]}], "t2i_bbox"), (1, [{"id": "ball", "from": "ref_image_0", "src_bbox": [0.1, 0.2, 0.5, 0.6], "tgt_bbox": None}], "i2i_bbox")])
def test_four_official_modes(images, boxes, mode):
    request = prepare_request("A red ball", [Image.new("RGB", (32, 32))] * images, boxes)
    assert request.mode == mode
    payload = request.payload()
    assert payload["presence_penalty"] == (0.0 if images else 1.5)
    assert payload["temperature"] == 0.0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["penalties_generated_only"] is True


def test_reference_order_and_unequal_sizes_survive():
    red = Image.new("RGB", (64, 32), "red")
    blue = Image.new("RGB", (32, 96), "blue")
    request = prepare_request("Put the first object in the second scene", [red, blue])
    assert request.reference_sizes == [(64, 32), (32, 96)]
    content = request.messages[1]["content"]
    for item, color in zip(content, [(254, 0, 0), (0, 0, 254)]):
        image = Image.open(io.BytesIO(base64.b64decode(item["image_url"]["url"].split(",", 1)[1])))
        assert image.getpixel((0, 0)) == color
    reverse = prepare_request("Put the first object in the second scene", [blue, red])
    assert request.fingerprint != reverse.fingerprint


@pytest.mark.parametrize("box", [[0.5, 0, 0.2, 1], [0, 0, 1, 2], [0, 0, float("nan"), 1], [0, 0, True, 1]])
def test_invalid_boxes_are_rejected(box):
    with pytest.raises(ValueError):
        parse_boxes([{"id": "ball", "bbox": box}], 0)


def test_reference_provenance_is_validated():
    with pytest.raises(ValueError, match="missing reference"):
        parse_boxes([{"id": "ball", "from": "ref_image_1", "src_bbox": [0, 0, 1, 1]}], 1)


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parse_boxes([{"id": "ball", "bbox": [0, 0, 1, 1]}] * 2, 0)


def test_locked_box_survives_model_coordinate_drift():
    request = prepare_request("A red ball", boxes=[{"id": "ball", "desc": "ball", "bbox": [0.1, 0.2, 0.5, 0.6]}])
    text = 'A red ball <ball> against white. [{"id":"ball","desc":"a glossy red ball","bbox":[101,201,501,601]}]'
    result = finish_caption(text, request, "stop")
    rows = json.loads(result[result.index("[{"):])
    assert rows[0]["bbox"] == [100, 200, 500, 600]
    assert rows[0]["id"] == "ball"
    assert rows[0]["desc"] == "a glossy red ball"


@pytest.mark.parametrize("reason", ["length", "error", None])
def test_truncation_is_not_a_success(reason):
    with pytest.raises(PEError):
        finish_caption("Incomplete but plausible text", prepare_request("A ball"), reason)


def test_missing_box_citation_is_not_silently_passed():
    request = prepare_request("A red ball", boxes=[{"id": "ball", "bbox": [0, 0, 1, 1]}])
    with pytest.raises(PEError, match="cite"):
        finish_caption('A red ball. [{"id":"ball","bbox":[0,0,1000,1000]}]', request, "stop")


def test_ten_images_supported_and_eleven_rejected():
    image = Image.new("RGB", (32, 32))
    assert len(prepare_request("Combine", [image] * 10).reference_sizes) == 10
    with pytest.raises(ValueError):
        prepare_request("Combine", [image] * 11)
