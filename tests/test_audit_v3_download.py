import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_audit_v2_build_inputs import CONFIG_FILES, script


@pytest.fixture
def auxiliary(tmp_path, monkeypatch):
    module = script("download_model")
    data = b"verified tiny model"
    monkeypatch.setattr(module, "SIZE", len(data))
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(data).hexdigest())
    (tmp_path / "model.safetensors").write_bytes(data)
    for name in CONFIG_FILES:
        (tmp_path / name).write_text('{}' if name.endswith('.json') else 'official source fixture', encoding='utf-8')
    path = tmp_path / "config.json"
    path.unlink()
    requests = []
    def get(url, **kwargs):
        requests.append(url)
        return SimpleNamespace(content=b'{"model_type":"qwen3_5"}', raise_for_status=lambda: None)
    monkeypatch.setattr(module.requests, "get", get)
    monkeypatch.setattr(sys, "argv", ["download_model.py", "--destination", str(tmp_path)])
    return module, path, requests


def test_interrupted_auxiliary_write_never_installs_truncated_final_file(auxiliary, monkeypatch):
    module, path, calls = auxiliary
    original_write = Path.write_bytes
    def fail_write(target, data):
        original_write(target, data[:3])
        raise OSError("simulated disk interruption")
    with monkeypatch.context() as local:
        local.setattr(Path, "write_bytes", fail_write)
        with pytest.raises(OSError, match="disk interruption"):
            module.main()
    assert not path.exists(), 'Incomplete config installed as final and skipped forever'
    module.main()
    assert json.loads(path.read_text()) == {"model_type": "qwen3_5"}
    assert len(calls) == 2


def test_existing_truncated_json_config_is_refetched(auxiliary):
    module, path, calls = auxiliary
    path.write_text('{"model_type":')
    module.main()
    assert json.loads(path.read_text()) == {"model_type": "qwen3_5"}
    assert len(calls) == 1


@pytest.mark.parametrize("body", [b'{"model_type":', b'', b'<html>temporary upstream failure</html>', b'[]'])
def test_invalid_auxiliary_response_not_promoted(auxiliary, monkeypatch, body):
    module, path, calls = auxiliary
    monkeypatch.setattr(module.requests, "get", lambda *args, **kwargs: SimpleNamespace(content=body, raise_for_status=lambda: None))
    with pytest.raises(ValueError):
        module.main()
    assert not path.exists()


def test_valid_existing_auxiliary_files_do_not_trigger_network(auxiliary):
    module, path, calls = auxiliary
    path.write_text('{"model_type":"qwen3_5"}')
    module.main()
    assert calls == []
