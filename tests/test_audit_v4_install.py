import os
from pathlib import Path

import pytest

from test_audit_v2_scripts import powershell, PWSH


ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "nt" or PWSH is None, reason="Windows PowerShell installer")


def test_fresh_custom_nodes_parent_and_repeated_unicode_backups(tmp_path):
    comfy = tmp_path / "Comfy 新目录 [v4]"
    comfy.mkdir()
    (comfy / "main.py").write_text("")
    user, inputs = tmp_path / "用户 [工作流]", tmp_path / "图片 [引用]"
    options = ("-ComfyUI", comfy, "-UserDirectory", str(user) + os.sep, "-InputDirectory", str(inputs) + os.sep)
    result = powershell(ROOT / "scripts/install_comfy.ps1", *options)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (comfy / "custom_nodes/ComfyUI-Flux3-PE").resolve() == ROOT
    workflow = user / "default/workflows/FLUX3_PE/04_reference_editing.json"
    reference = inputs / "FLUX3_PE_reference_scene.png"
    for index in range(2):
        workflow.write_bytes(f"user graph {index}".encode())
        reference.write_bytes(f"user image {index}".encode())
        result = powershell(ROOT / "scripts/install_comfy.ps1", *options)
        assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(path.read_bytes() for path in workflow.parent.glob(workflow.name + ".*.bak")) == [b"user graph 0", b"user graph 1"]
    assert sorted(path.read_bytes() for path in reference.parent.glob(reference.name + ".*.bak")) == [b"user image 0", b"user image 1"]
    assert workflow.read_bytes() == (ROOT / "examples" / workflow.name).read_bytes()
    assert reference.read_bytes() == (ROOT / "examples/reference_images/reference_scene.png").read_bytes()
