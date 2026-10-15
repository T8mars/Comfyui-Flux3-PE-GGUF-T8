import pytest

from flux3_pe.engine import ModelSpec
from test_model_spec import model_files, load


@pytest.mark.parametrize("context", [True, False, 4096.0, 8192.5, float("nan"), float("inf"), "8192", None, 4095, 0])
def test_context_rejected_before_model_manifest_or_runtime_reads(tmp_path, context):
    with pytest.raises(ValueError, match="integer.*4096"):
        ModelSpec.load(tmp_path / "missing.gguf", context=context)


@pytest.mark.parametrize("context", [4096, 8192, 65536])
def test_positive_supported_integer_context_preserved(model_files, context):
    assert load(model_files, context=context).context == context


@pytest.mark.parametrize("context", [4096.0, 8192.5, float("nan"), float("inf")])
def test_noninteger_context_is_not_accepted_with_valid_model_files(model_files, context):
    with pytest.raises(ValueError, match="integer.*4096"):
        load(model_files, context=context)
