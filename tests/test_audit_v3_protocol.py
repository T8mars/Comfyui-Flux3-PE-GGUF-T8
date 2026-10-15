import pytest

from flux3_pe.protocol import prepare_request


@pytest.mark.parametrize("canvas", [
    (True, 512), (512, False), (512.0, 512), (512, 512.5),
    (float("nan"), 512), (512, float("inf")), ("512", 512),
    (512,), (512, 512, 512), [512, 512, 512], "512:512",
])
def test_canvas_requires_two_positive_integer_dimensions(canvas):
    with pytest.raises(ValueError, match="positive integer"):
        prepare_request("A red ball", canvas=canvas)


@pytest.mark.parametrize("canvas", [(1, 1), (512, 768), [768, 512]])
def test_valid_canvas_preserves_literal_dimensions(canvas):
    request = prepare_request("A red ball", canvas=canvas)
    assert f"TARGET CANVAS: {canvas[0]}:{canvas[1]}\nA red ball" == request.messages[1]["content"]


def test_absent_canvas_preserves_request_fingerprint():
    assert prepare_request("A red ball").fingerprint == prepare_request("A red ball", canvas=None).fingerprint
