"""CLI inference failure must release ownership and retain prior user outputs."""
import sys

import pytest

from flux3_pe.protocol import PEError
from test_audit_v3_cli import cli_module


@pytest.mark.parametrize("failure", [PEError("invalid generated layout"), InterruptedError("cancelled"), OSError("backend transport failed")])
def test_cli_failed_inference_releases_without_replacing_existing_outputs(tmp_path, monkeypatch, failure):
    module = cli_module()
    output = tmp_path / "existing.txt"
    metadata = output.with_suffix(".json")
    output.write_bytes(b"EXISTING CAPTION")
    metadata.write_bytes(b'{"existing":"metadata"}')
    monkeypatch.setattr(module.ModelSpec, "load", lambda *args, **kwargs: object())
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(module, "enhance", fail)
    released = []
    monkeypatch.setattr(module, "release_model", lambda: released.append(True))
    monkeypatch.setattr(sys, "argv", ["cli.py", "A red ball", "--output", str(output)])
    with pytest.raises(type(failure), match=str(failure)):
        module.main()
    assert released == [True]
    assert output.read_bytes() == b"EXISTING CAPTION"
    assert metadata.read_bytes() == b'{"existing":"metadata"}'
