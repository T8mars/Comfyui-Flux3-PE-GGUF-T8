from test_nodes import run_node_script


def test_pre_cancelled_node_does_not_transfer_reference_frames():
    run_node_script("""
        class Cancelled(RuntimeError): pass
        class UnexpectedTensor:
            shape = (1, 32, 32, 3)
            def __iter__(self):
                raise AssertionError('Reference tensor iterated after cancellation')
        def interrupt(): raise Cancelled('already cancelled')
        management.throw_exception_if_processing_interrupted = interrupt
        nodes.enhance = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('enhance ran'))
        try:
            nodes.Flux3PromptEnhance.execute(None, 'A ball', images={'image_1': UnexpectedTensor()})
        except Cancelled: pass
        else: raise AssertionError('Cancellation lost')
    """)


def test_node_cancel_between_frames_stops_following_cpu_transfers():
    run_node_script("""
        import numpy as np
        class Cancelled(RuntimeError): pass
        transfers = []
        class Frame:
            def __init__(self, index): self.index = index
            def detach(self): return self
            def float(self): return self
            def cpu(self):
                transfers.append(self.index)
                return self
            def numpy(self): return np.zeros((32, 32, 3), dtype=np.float32)
        class Tensor:
            shape = (3, 32, 32, 3)
            def __iter__(self): return iter([Frame(i) for i in range(3)])
        def interrupt():
            if transfers: raise Cancelled('cancelled between frames')
        management.throw_exception_if_processing_interrupted = interrupt
        nodes.enhance = lambda *args, **kwargs: ('caption', {})
        try:
            nodes.Flux3PromptEnhance.execute(None, 'A ball', images={'image_1': Tensor()})
        except Cancelled: pass
        else: raise AssertionError('Cancellation lost')
        assert transfers == [0], transfers
    """)


def test_noncontiguous_half_double_and_two_channel_frames_keep_colors_and_order():
    run_node_script("""
        import torch
        captured = []
        nodes.enhance = lambda spec, prompt, images, *args, **kwargs: (captured.extend(images) or 'caption', {})
        half = torch.zeros((2, 64, 32, 3), dtype=torch.float16)
        half[0, ..., 0] = 1; half[1, ..., 2] = 1
        half = half.transpose(1, 2)
        assert not half.is_contiguous()
        double = torch.zeros((1, 32, 32, 3), dtype=torch.float64)
        double[..., 0] = -0.5; double[..., 1] = 1.5; double[..., 2] = .5
        la = torch.ones((1, 32, 32, 2)); la[..., 0] = .25; la[..., 1] = .5
        nodes.Flux3PromptEnhance.execute(None, 'Combine', images={'image_10': la, 'image_2': double, 'image_1': half})
        assert [image.mode for image in captured] == ['RGB', 'RGB', 'RGB', 'LA']
        assert [image.size for image in captured] == [(64, 32), (64, 32), (32, 32), (32, 32)]
        assert [image.getpixel((0, 0)) for image in captured] == [(255, 0, 0), (0, 0, 255), (0, 255, 128), (64, 128)]
        from node_test_package.flux3_pe.protocol import prepare_request
        import base64, io
        from PIL import Image
        request = prepare_request('Combine', captured)
        jpeg = base64.b64decode(request.messages[1]['content'][3]['image_url']['url'].split(',', 1)[1])
        assert abs(Image.open(io.BytesIO(jpeg)).getpixel((16, 16))[0] - 159) <= 2
    """)
