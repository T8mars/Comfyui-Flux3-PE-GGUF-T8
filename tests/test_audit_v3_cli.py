import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["caption.txt", "caption.json", "caption.JSON", "caption"])
def test_cli_preserves_caption_and_separate_metadata(tmp_path, monkeypatch, capsys, name):
    spec = importlib.util.spec_from_file_location("audit_v3_cli", ROOT / "scripts/cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.ModelSpec, "load", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "enhance", lambda *args, **kwargs: ("A red bicycle.", {"mode": "t2i"}))
    released = []
    monkeypatch.setattr(module, "release_model", lambda: released.append(True))
    output = tmp_path / "nested" / name
    monkeypatch.setattr(sys, "argv", ["cli.py", "A bicycle", "--output", str(output)])
    module.main()
    assert output.read_text(encoding="utf-8") == "A red bicycle.\n"
    metadata = output.with_name(output.name + ".metadata.json") if output.suffix.lower() == ".json" else output.with_suffix(".json")
    assert json.loads(metadata.read_text(encoding="utf-8")) == {"mode": "t2i"}
    assert released == [True]
    assert capsys.readouterr().out.strip() == "A red bicycle."


def cli_module():
    spec = importlib.util.spec_from_file_location("audit_v3_cli_paths", ROOT / "scripts/cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("flag", ["--boxes", "--image", "--model"])
@pytest.mark.parametrize("hardlink", [False, True])
def test_cli_output_never_overwrites_input_before_loading(tmp_path, monkeypatch, flag, hardlink):
    module = cli_module()
    original = tmp_path / "input.dat"
    original.write_bytes(b"ORIGINAL INPUT MUST BE PRESERVED")
    output = tmp_path / "output.dat" if hardlink else original
    if hardlink:
        import os
        os.link(original, output)
    monkeypatch.setattr(module.ModelSpec, "load", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("model loaded before output validation")))
    monkeypatch.setattr(sys, "argv", ["cli.py", "Edit", flag, str(original), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert original.read_bytes() == b"ORIGINAL INPUT MUST BE PRESERVED"


def test_cli_metadata_avoids_box_input_and_both_caption_and_metadata_hardlinks(tmp_path, monkeypatch):
    module = cli_module()
    boxes = tmp_path / "layout.json"
    boxes.write_text('[{"id":"ball","bbox":[0,0,1,1]}]', encoding="utf-8")
    original = boxes.read_bytes()
    output = tmp_path / "layout.txt"
    # Force a second fallback too: this candidate aliases the protected boxes.
    import os
    os.link(boxes, tmp_path / "layout.txt.metadata.json")
    monkeypatch.setattr(module.ModelSpec, "load", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "enhance", lambda *args, **kwargs: ("A red ball.", {"mode": "t2i_bbox"}))
    monkeypatch.setattr(module, "release_model", lambda: None)
    monkeypatch.setattr(sys, "argv", ["cli.py", "A ball", "--boxes", str(boxes), "--output", str(output)])
    module.main()
    assert boxes.read_bytes() == original
    assert output.read_text() == "A red ball.\n"
    assert json.loads((tmp_path / "layout.txt.metadata-2.json").read_text()) == {"mode": "t2i_bbox"}


def test_existing_caption_and_sidecar_hardlinks_do_not_clobber_caption(tmp_path):
    import os
    module = cli_module()
    output = tmp_path / "caption.txt"
    output.write_text("old")
    os.link(output, output.with_suffix(".json"))
    assert module.output_metadata_path(output, []) == tmp_path / "caption.txt.metadata.json"


@pytest.fixture
def implicit_inputs(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = cli_module()
    model, mmproj = tmp_path / 'pup.gguf', tmp_path / 'mmproj.gguf'
    server = tmp_path / 'runtime' / 'llama-server.exe'
    server.parent.mkdir()
    library = server.parent / 'llama-common.dll'
    for path in (model, mmproj, server, library):
        path.write_bytes(b'REQUIRED INPUT')
    model_manifest, runtime_manifest = tmp_path / 'manifest.json', server.parent / 'flux3-pe-runtime.json'
    model_manifest.write_text('{"required":"model manifest"}')
    runtime_manifest.write_text(json.dumps({'core_sha256': {library.name: 'testhash'}}))
    fake = SimpleNamespace(model=model, mmproj=mmproj, server=server)
    monkeypatch.setattr(module.ModelSpec, 'load', lambda *args, **kwargs: fake)
    return module, fake, {'model_manifest': model_manifest, 'runtime_manifest': runtime_manifest, 'mmproj': mmproj, 'server': server, 'library': library}


@pytest.mark.parametrize('name', ['model_manifest', 'runtime_manifest', 'mmproj', 'server', 'library'])
def test_explicit_caption_does_not_overwrite_implicit_model_or_runtime_inputs(implicit_inputs, monkeypatch, name):
    module, spec, paths = implicit_inputs
    path = paths[name]
    original = path.read_bytes()
    monkeypatch.setattr(module, 'enhance', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('GPU called before path validation')))
    monkeypatch.setattr(sys, 'argv', ['cli.py', 'A ball', '--model', str(spec.model), '--output', str(path)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and path.read_bytes() == original


@pytest.mark.parametrize('name', ['model_manifest', 'runtime_manifest'])
def test_caption_sidecar_avoids_implicit_json_manifests(implicit_inputs, monkeypatch, name):
    module, spec, paths = implicit_inputs
    manifest = paths[name]
    output = manifest.with_suffix('.txt')
    original = manifest.read_bytes()
    monkeypatch.setattr(module, 'enhance', lambda *args, **kwargs: ('A red ball.', {'mode': 't2i'}))
    monkeypatch.setattr(module, 'release_model', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['cli.py', 'A ball', '--model', str(spec.model), '--output', str(output)])
    module.main()
    assert manifest.read_bytes() == original
    assert output.read_text() == 'A red ball.\n'
    assert json.loads(output.with_name(output.name + '.metadata.json').read_text()) == {'mode': 't2i'}
