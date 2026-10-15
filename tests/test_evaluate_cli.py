from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("arguments,message", [
    (["--ids", "does_not_exist"], "Unknown case IDs"),
    (["--ids", "t2i_01", "typo_id"], "Unknown case IDs"),
    (["--limit", "0"], "limit must be positive"),
    (["--limit", "-1"], "limit must be positive"),
    (["--max-tokens", "0"], "max-tokens must be in"),
    (["--max-tokens", "3585"], "max-tokens must be in"),
])
def test_invalid_selection_or_budget_fails_before_model_loading(tmp_path, arguments, message):
    output = tmp_path / "must-not-be-written.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), *arguments, "--output", str(output)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert message in result.stderr
    assert not output.exists()
