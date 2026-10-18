"""Exercise node caching in a subprocess without importing Comfy into pytest."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRELUDE = """
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

root = Path(sys.argv[1])
folder_paths = types.ModuleType('folder_paths')
folder_paths.models_dir = str(root / 'models')
folder_paths.add_model_folder_path = lambda *args: None
folder_paths.get_full_path_or_raise = lambda category, name: str(root / name)
sys.modules['folder_paths'] = folder_paths
comfy = types.ModuleType('comfy')
management = types.ModuleType('comfy.model_management')
management.unload_all_models = lambda: None
management.soft_empty_cache = lambda: None
management.throw_exception_if_processing_interrupted = lambda: None
comfy.model_management = management
sys.modules['comfy'] = comfy
sys.modules['comfy.model_management'] = management
latest = types.ModuleType('comfy_api.latest')
latest.ComfyExtension = object
latest.io = types.SimpleNamespace(
    ComfyNode=type('ComfyNode', (), {}), Custom=lambda name: object(),
    NodeOutput=lambda *values: values,
)
sys.modules['comfy_api'] = types.ModuleType('comfy_api')
sys.modules['comfy_api.latest'] = latest
package = types.ModuleType('node_test_package')
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
spec = importlib.util.spec_from_file_location('node_test_package.nodes', root / 'nodes.py')
nodes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nodes
spec.loader.exec_module(nodes)
from node_test_package.flux3_pe import engine
"""


def run_node_script(script):
    completed = subprocess.run(
        [sys.executable, "-c", PRELUDE + textwrap.dedent(script), str(ROOT)],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_repeated_node_runs_enter_internal_cache_and_report_fresh_metadata():
    run_node_script("""
        class FakeEngine:
            calls = 0
            closes = 0
            def __init__(self, spec): self.spec = spec
            def start(self, interrupt): interrupt()
            def complete(self, payload, interrupt):
                type(self).calls += 1
                return {'choices': [{'message': {'content': 'A red bicycle against a white wall.'}, 'finish_reason': 'stop'}]}
            def close(self): type(self).closes += 1
        engine.LlamaEngine = FakeEngine
        model = engine.ModelSpec(Path('model'), Path('vision'), Path('server'), 'test')
        for use_cache in (True, False):
            engine.release_model()
            engine._results.clear()
            FakeEngine.calls = 0
            fingerprints = [nodes.Flux3PromptEnhance.fingerprint_inputs(use_cache=use_cache) for _ in range(2)]
            assert all(math.isnan(value) for value in fingerprints)
            assert fingerprints[0] != fingerprints[1]
            first = nodes.Flux3PromptEnhance.execute(model, 'A bicycle', use_cache=use_cache, keep_loaded=True)
            second = nodes.Flux3PromptEnhance.execute(model, 'A bicycle', use_cache=use_cache, keep_loaded=False)
            assert first[0] == second[0]
            assert json.loads(second[1])['cache_hit'] is use_cache
            assert FakeEngine.calls == (1 if use_cache else 2)
            assert engine._engine is None
        engine.release_model()
    """)


def test_loader_fingerprint_and_execution_forward_context_without_stale_handle():
    run_node_script("""
        calls = []
        def load(path, context):
            calls.append((path, context))
            return engine.ModelSpec(Path(path), Path('vision'), Path('server'), 'same-model', context)
        nodes.ModelSpec.load = load
        for context in (4096, 8192):
            assert nodes.Flux3PELoader.fingerprint_inputs('model.gguf', context_size=context) == 'same-model'
            result = nodes.Flux3PELoader.execute('model.gguf', context_size=context)
            assert result[0].context == context
        assert [context for _, context in calls] == [4096, 4096, 8192, 8192]
    """)


def test_loader_model_options_use_portable_subdirectory_paths():
    run_node_script(r"""
        folder_paths.get_filename_list = lambda _: [
            r'heretic\pup-Q4_K_M.gguf', 'heretic/pup-Q4_K_M.gguf',
            r'heretic\mmproj-pup.gguf', r'heretic\pup-Q4_K_M.partial.gguf',
            'official-Q4_K_M.gguf',
        ]
        latest.io.Combo = types.SimpleNamespace(Input=lambda name, **options: dict(name=name, **options))
        latest.io.Int = types.SimpleNamespace(Input=lambda name, **options: dict(name=name, **options))
        latest.io.Schema = lambda **schema: schema
        nodes.MODEL_TYPE = types.SimpleNamespace(Output=lambda **options: options)
        schema = nodes.Flux3PELoader.define_schema()
        options = schema['inputs'][0]['options']
        assert options == ['heretic/pup-Q4_K_M.gguf', 'official-Q4_K_M.gguf'], options
    """)


def test_internal_cache_hit_does_not_unload_comfy_image_models():
    run_node_script("""
        unloads = []
        management.unload_all_models = lambda: unloads.append('unload')
        management.soft_empty_cache = lambda: unloads.append('empty')
        class FakeEngine:
            def __init__(self, spec): self.spec = spec
            def start(self, interrupt): interrupt()
            def complete(self, payload, interrupt):
                return {'choices': [{'message': {'content': 'A bicycle.'}, 'finish_reason': 'stop'}]}
            def close(self): pass
        engine.LlamaEngine = FakeEngine
        model = engine.ModelSpec(Path('model'), Path('vision'), Path('server'), 'test')
        nodes.Flux3PromptEnhance.execute(model, 'A bicycle')
        assert unloads == ['unload', 'empty']
        nodes.Flux3PromptEnhance.execute(model, 'A bicycle')
        assert unloads == ['unload', 'empty']
    """)


def test_image_inputs_keep_numeric_slot_order_batch_order_and_independent_sizes():
    run_node_script("""
        import torch
        captured = []
        def fake_enhance(spec, prompt, images, *args, **kwargs):
            captured.extend(images)
            return 'caption', {}
        nodes.enhance = fake_enhance
        green = torch.zeros((1, 32, 64, 3)); green[..., 1] = 1
        red_blue = torch.zeros((2, 96, 32, 3)); red_blue[0, ..., 0] = 1; red_blue[1, ..., 2] = 1
        nodes.Flux3PromptEnhance.execute(None, 'Combine', images={'image_10': red_blue, 'image_1': None, 'image_2': green})
        assert [image.size for image in captured] == [(64, 32), (32, 96), (32, 96)]
        assert [image.getpixel((0, 0)) for image in captured] == [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
    """)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_grayscale_rgb_and_rgba_comfy_images_reach_reference_preprocessing(channels):
    run_node_script(f"""
        import io
        import torch
        from PIL import Image
        from node_test_package.flux3_pe.protocol import prepare_reference
        captured = []
        def fake_enhance(spec, prompt, images, *args, **kwargs):
            data, size = prepare_reference(images[0], 1)
            captured.append(Image.open(io.BytesIO(data)).getpixel((0, 0)))
            return 'caption', {{}}
        nodes.enhance = fake_enhance
        image = torch.zeros((1, 32, 32, {channels}))
        nodes.Flux3PromptEnhance.execute(None, 'Describe', images={{'image_1': image}})
        assert captured == ([(255, 255, 255)] if {channels} == 4 else [(0, 0, 0)])
    """)


def test_nonfinite_comfy_image_pixels_are_rejected_without_silent_replacement():
    run_node_script("""
        import torch
        nodes.enhance = lambda *args, **kwargs: ('caption', {})
        for invalid in (float('nan'), float('inf'), float('-inf')):
            image = torch.full((1, 32, 32, 3), invalid)
            try:
                nodes.Flux3PromptEnhance.execute(None, 'Describe', images={'image_1': image})
            except ValueError:
                pass
            else:
                raise AssertionError('Nonfinite image was silently converted to a fabricated reference')
    """)


def test_expanded_image_batch_limit_is_enforced_before_backend_launch():
    run_node_script("""
        import torch
        model = engine.ModelSpec(Path('model'), Path('vision'), Path('server'), 'test')
        class UnexpectedEngine:
            def __init__(self, *args): raise AssertionError('backend launched for invalid image count')
        engine.LlamaEngine = UnexpectedEngine
        try:
            nodes.Flux3PromptEnhance.execute(model, 'Combine', images={'image_1': torch.zeros((11, 32, 32, 3))})
        except ValueError as error:
            assert '10' in str(error)
        else:
            raise AssertionError('Image batch above10 was accepted')
    """)
