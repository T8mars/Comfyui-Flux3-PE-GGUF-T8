"""PIL image contracts and allowed extreme-aspect reference boundaries."""
import io

from PIL import Image, ImageOps
import pytest

from flux3_pe.protocol import prepare_reference, prepare_request, reference_size


@pytest.mark.parametrize("count", [1, 2, 3, 10])
@pytest.mark.parametrize("size", [(20000, 100), (100, 20000), (10000, 64), (64, 10000)])
def test_aligned_allowed_aspect_never_exceeds_official_processor_limit(count, size):
    width, height = reference_size(*size, count)
    assert max(width, height) / min(width, height) <= 200
    assert width % 32 == height % 32 == 0 and min(width, height) >= 32
    cap = 1024**2 if count == 1 else 724**2 if count == 2 else 512**2
    assert width * height <= cap


@pytest.mark.parametrize("size", [(1, 1), (16, 48), (31, 33), (200, 1), (1, 200)])
@pytest.mark.parametrize("count", [1, 2, 10])
def test_small_allowed_references_keep_alignment_and_pixel_budget(size, count):
    width, height = reference_size(*size, count)
    assert width >= 32 and height >= 32 and width % 32 == height % 32 == 0
    assert max(width, height) / min(width, height) <= 200
    cap = 1024**2 if count == 1 else 724**2 if count == 2 else 512**2
    assert width * height <= cap


def test_exif_rotated_reference_has_same_payload_identity_as_upright_pixels():
    image = Image.new("RGB", (32, 64), "red")
    image.paste("blue", (16, 0, 32, 64))
    image.getexif()[274] = 6
    upright = ImageOps.exif_transpose(image)
    first = prepare_request("Describe", [image])
    second = prepare_request("Describe", [upright])
    assert first.reference_sizes == second.reference_sizes == [(64, 32)]
    assert first.messages == second.messages and first.fingerprint == second.fingerprint
    assert image.getexif()[274] == 6 and image.size == (32, 64)


@pytest.mark.parametrize("mode", ["P", "LA", "RGBA", "PA"])
def test_transparent_modes_have_white_background_before_jpeg(mode):
    if mode == "P":
        image = Image.new("P", (32, 32), 0)
        image.putpalette([255, 0, 0] + [0, 0, 0] * 255)
        image.info["transparency"] = 0
    else:
        image = Image.new(mode, (32, 32), (0, 0) if mode in ("LA", "PA") else (255, 0, 0, 0))
    data, size = prepare_reference(image, 1)
    converted = Image.open(io.BytesIO(data))
    assert converted.mode == "RGB" and size == (32, 32)
    assert converted.getpixel((16, 16)) == (255, 255, 255)


def test_grayscale_values_and_order_survive_reference_conversion():
    gray = Image.new("L", (32, 64), 127)
    blue = Image.new("RGB", (96, 32), "blue")
    data, size = prepare_reference(gray, 2)
    assert Image.open(io.BytesIO(data)).getpixel((16, 32)) == (127, 127, 127)
    forward = prepare_request("Compare", [gray, blue])
    backward = prepare_request("Compare", [blue, gray])
    assert forward.reference_sizes == [(32, 64), (96, 32)]
    assert backward.reference_sizes == [(96, 32), (32, 64)]
    assert forward.fingerprint != backward.fingerprint


@pytest.mark.parametrize("count,expected", [(1, (1024, 1024)), (2, (704, 704)), (3, (512, 512)), (10, (512, 512))])
def test_usual_reference_resize_dimensions_are_unchanged(count, expected):
    assert reference_size(2048, 2048, count) == expected
    assert reference_size(256, 128, count) == (256, 128)
