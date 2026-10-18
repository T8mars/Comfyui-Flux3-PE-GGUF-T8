"""Keep Heretic artifacts distinct and reject incomplete/mislabeled conversions."""
import dataclasses
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import pytest

from flux3_pe import engine, wheel_backend
from flux3_pe.protocol import (
    HERETIC_MODEL, HERETIC_REVISION, HERETIC_SOURCE_SHA256, SOURCE_MODEL,
    SOURCE_REVISION, PEError,
)
from test_model_spec import model_files, load
from test_audit_v2_cache import cached_backend, spec


def heretic(files):
    files['manifest']['source'] = {
        'model_id': HERETIC_MODEL, 'revision': HERETIC_REVISION,
        'sha256': HERETIC_SOURCE_SHA256,
    }
    files['manifest']['mmproj']['source_revision'] = HERETIC_REVISION
    files['save']()


def test_heretic_manifest_loads_with_own_provenance_and_no_server(model_files, monkeypatch):
    heretic(model_files)
    monkeypatch.setattr(wheel_backend, 'wheel_identity', lambda: 'wheel-fixture')
    value = engine.ModelSpec.load(model_files['model'], backend='llama-cpp-python')
    assert value.source_model == HERETIC_MODEL and value.source_revision == HERETIC_REVISION
    assert value.mmproj == model_files['projector'] and value.server is None


@pytest.mark.parametrize('field,value,match', [
    ('revision', SOURCE_REVISION, 'revision'),
    ('sha256', 'incorrect', 'checksum'),
    ('model_id', 'unknown/pup', 'revision'),
])
def test_heretic_does_not_accept_mislabeled_source(model_files, field, value, match):
    heretic(model_files)
    model_files['manifest']['source'][field] = value
    model_files['save']()
    with pytest.raises(PEError, match=match):
        load(model_files)


def test_heretic_requires_a_projector_with_matching_provenance(model_files):
    heretic(model_files)
    model_files['manifest']['mmproj']['source_revision'] = SOURCE_REVISION
    model_files['save']()
    with pytest.raises(PEError, match='projector'):
        load(model_files)


@pytest.mark.parametrize('artifact', ['model', 'projector'])
def test_same_size_corruption_of_heretic_is_detected(model_files, artifact):
    heretic(model_files)
    load(model_files)
    path = model_files[artifact]
    path.write_bytes(b'X' * path.stat().st_size)
    with pytest.raises(PEError, match='checksum'):
        load(model_files)


def test_source_provenance_is_part_of_backend_identity(model_files, monkeypatch):
    monkeypatch.setattr(wheel_backend, 'wheel_identity', lambda: 'wheel-fixture')
    official = engine.ModelSpec.load(model_files['model'], backend='llama-cpp-python')
    heretic(model_files)
    variant = engine.ModelSpec.load(model_files['model'], backend='llama-cpp-python')
    assert official.revision != variant.revision
    assert official.source_model == SOURCE_MODEL


def test_heretic_execution_and_cache_report_actual_source(cached_backend):
    value = dataclasses.replace(spec(), source_model=HERETIC_MODEL, source_revision=HERETIC_REVISION)
    for hit in (False, True):
        _, details = engine.enhance(value, 'A red ball on white.')
        assert details['cache_hit'] is hit
        assert details['source_model'] == HERETIC_MODEL
        assert details['source_revision'] == HERETIC_REVISION
        assert details['protocol_source_revision'] == SOURCE_REVISION


def test_heretic_converter_emits_own_names_and_source_metadata(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    module_spec = importlib.util.spec_from_file_location('heretic_quantize_fixture', root / 'scripts/quantize.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    source, output = tmp_path / 'source', tmp_path / 'output'
    source.mkdir()
    header = json.dumps({'language': {}, 'vision': {}}).encode()
    original = source / 'model.safetensors'
    original.write_bytes(struct.pack('<Q', len(header)) + header)
    monkeypatch.setattr(module, 'sha256_file', lambda path: HERETIC_SOURCE_SHA256)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100 * 1024**3))
    monkeypatch.setattr(module, 'metadata', lambda path, expected_type=None: {'file': path.name, 'tensor_count': 1})
    monkeypatch.setattr(module, 'mmproj_metadata', lambda path: {'file': path.name, 'source_tensor_count': 1})
    calls = []
    def run(arguments):
        calls.append(arguments)
        destination = Path(arguments[arguments.index('--outfile') + 1]) if '--outfile' in arguments else Path(arguments[2])
        destination.write_bytes(b'converter fixture')
    monkeypatch.setattr(module, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['quantize.py', '--variant', 'heretic', '--source', str(source), '--output', str(output)])
    module.main()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['source']['model_id'] == HERETIC_MODEL
    assert manifest['source']['revision'] == HERETIC_REVISION
    assert manifest['source']['sha256'] == HERETIC_SOURCE_SHA256
    assert manifest['mmproj']['source_revision'] == HERETIC_REVISION
    assert manifest['models']['Q4_K_M']['file'] == 'flux3-klein-pup-heretic-Q4_K_M.gguf'
    assert manifest['mmproj']['file'] == 'mmproj-flux3-klein-pup-heretic-F16.gguf'
    assert len(calls) == 3
