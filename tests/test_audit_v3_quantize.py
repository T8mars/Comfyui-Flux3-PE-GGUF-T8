import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

from test_audit_v2_build_inputs import script


@pytest.fixture
def quantized(tmp_path, monkeypatch):
    module = script('quantize')
    source, output = tmp_path / 'original', tmp_path / 'gguf'
    source.mkdir(); output.mkdir()
    header = json.dumps({f'tensor_{i}': {} for i in range(4)}).encode()
    original = source / 'model.safetensors'
    original.write_bytes(struct.pack('<Q', len(header)) + header)
    real_hash = module.sha256_file
    pinned = 'f5062a1fba3e63afe5c88349a311903fec06c959b362f1247127e63d5c860ed4'
    monkeypatch.setattr(module, 'sha256_file', lambda path: pinned if path == original else real_hash(path))
    for name in ('flux3-klein-pup-BF16.gguf', 'mmproj-flux3-klein-pup-F16.gguf', 'flux3-klein-pup-Q4_K_M.gguf'):
        (output / name).write_bytes(b'tiny existing converted fixture')
    settings = {'quant_file_type': 15, 'projector_file_type': 1}
    def reader(path):
        projector = path.name.startswith('mmproj-')
        arch = 'clip' if projector else 'qwen35'
        value = settings['projector_file_type'] if projector else settings['quant_file_type'] if 'Q4_K_M' in path.name else 32
        fields = {
            'general.architecture': SimpleNamespace(contents=lambda: arch),
            'general.file_type': SimpleNamespace(contents=lambda: value),
            'qwen35.block_count': SimpleNamespace(contents=lambda: 32),
            'clip.projector_type': SimpleNamespace(contents=lambda: 'qwen3vl_merger'),
        }
        names = ('v.patch_embd.weight', 'v.patch_embd.weight.1') if projector else ('token_embd.weight', 'output.weight', 'blk.31.ffn_down.weight')
        tensors = [SimpleNamespace(name=name, shape=[4096, 248320], tensor_type=SimpleNamespace(name='F16' if projector else 'BF16' if value == 32 else 'Q4_K')) for name in names]
        return SimpleNamespace(fields=fields, tensors=tensors)
    monkeypatch.setattr(module, 'GGUFReader', reader)
    monkeypatch.setattr(module, 'run', lambda args: (_ for _ in ()).throw(AssertionError('Real converter must not run')))
    monkeypatch.setattr(sys, 'argv', ['quantize.py', '--source', str(source), '--output', str(output)])
    return module, output, settings, pinned


@pytest.mark.parametrize('wrong_type', [1, 2, 14, 32])
def test_q4_k_m_named_artifact_must_have_requested_gguf_file_type(quantized, wrong_type):
    module, output, settings, _ = quantized
    settings['quant_file_type'] = wrong_type
    with pytest.raises(RuntimeError, match='file type|quantization'):
        module.main()
    assert not (output / 'manifest.json').exists()


def test_f16_projector_named_artifact_must_have_f16_file_type(quantized):
    module, output, settings, _ = quantized
    settings['projector_file_type'] = 0
    with pytest.raises(RuntimeError, match='file type|F16'):
        module.main()
    assert not (output / 'manifest.json').exists()


@pytest.mark.parametrize('key,value', [('revision', 'another-revision'), ('model_id', 'another/model'), ('sha256', '0' * 64)])
def test_existing_foreign_manifest_is_preserved_and_rejected_before_conversion(quantized, key, value):
    module, output, _, pinned = quantized
    manifest = {'source': {'model_id': module.SOURCE_MODEL, 'revision': module.SOURCE_REVISION, 'sha256': pinned, 'tensor_count': 4}, 'models': {}}
    manifest['source'][key] = value
    path = output / 'manifest.json'
    path.write_text(json.dumps(manifest))
    original = path.read_bytes()
    monkey_reader = module.GGUFReader
    module.GGUFReader = lambda path: (_ for _ in ()).throw(AssertionError('GGUF read before foreign-manifest rejection'))
    try:
        with pytest.raises(RuntimeError, match='source|revision'):
            module.main()
    finally:
        module.GGUFReader = monkey_reader
    assert path.read_bytes() == original


def test_correct_requested_file_types_produce_supported_manifest(quantized):
    module, output, settings, pinned = quantized
    module.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['source']['sha256'] == pinned
    assert manifest['models']['Q4_K_M']['file'] == 'flux3-klein-pup-Q4_K_M.gguf'
