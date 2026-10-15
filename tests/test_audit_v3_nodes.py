"""New audit: node image contracts before any inference side effect."""
from test_nodes import run_node_script


def test_invalid_tensor_shapes_never_silently_change_reference_mode():
    run_node_script("""
        import torch
        calls = []
        nodes.enhance = lambda *args, **kwargs: (calls.append(args[2]) or 'caption', {})
        for shape in ((8, 32, 3), (0, 32, 32, 3), (1, 0, 32, 3), (1, 32, 0, 3), (1, 32, 32, 5), (1, 32, 32)):
            calls.clear()
            try:
                nodes.Flux3PromptEnhance.execute(None, 'Edit the reference', images={'image_1': torch.zeros(shape)})
            except ValueError as error:
                assert 'image_1' in str(error) and 'BHWC' in str(error), (shape, str(error))
            else:
                raise AssertionError(f'Invalid shape {shape} silently accepted as {len(calls[0])} references')
            assert not calls, shape
    """)


def test_expanded_image_limit_is_checked_before_any_tensor_conversion():
    run_node_script("""
        import torch
        original_detach = torch.Tensor.detach
        torch.Tensor.detach = lambda self: (_ for _ in ()).throw(AssertionError('pixels converted before count validation'))
        try:
            nodes.Flux3PromptEnhance.execute(None, 'Combine', images={'image_1': torch.zeros((6, 32, 32, 3)), 'image_2': torch.zeros((5, 32, 32, 3))})
        except ValueError as error:
            assert '10' in str(error)
        else:
            raise AssertionError('11 references accepted')
        finally:
            torch.Tensor.detach = original_detach
    """)


def test_la_image_tensor_reaches_white_alpha_compositing():
    run_node_script("""
        import io
        import torch
        from PIL import Image
        from node_test_package.flux3_pe.protocol import prepare_reference
        pixels = []
        def fake_enhance(spec, prompt, images, *args, **kwargs):
            data, size = prepare_reference(images[0], 1)
            pixels.append(Image.open(io.BytesIO(data)).getpixel((0, 0)))
            return 'caption', {}
        nodes.enhance = fake_enhance
        nodes.Flux3PromptEnhance.execute(None, 'Describe', images={'image_1': torch.zeros((1, 32, 32, 2))})
        assert pixels == [(255, 255, 255)]
    """)
