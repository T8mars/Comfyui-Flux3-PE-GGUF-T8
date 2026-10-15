"""Manager-selected folder names must not create a second node installation."""
from pathlib import Path
import shutil

import pytest

from test_install import ROOT, install, pytestmark


@pytest.mark.parametrize("name", ["flux3-pe-gguf-t8", "Comfyui-Flux3-PE-GGUF-T8"])
def test_manager_folder_installs_workflows_without_duplicate_junction(tmp_path, name):
    comfy = tmp_path / "ComfyUI"
    node = comfy / "custom_nodes" / name
    (node / "scripts").mkdir(parents=True)
    (comfy / "main.py").write_text("")
    shutil.copy2(ROOT / "scripts/install_comfy.ps1", node / "scripts/install_comfy.ps1")
    shutil.copytree(ROOT / "examples", node / "examples")
    result = install(node / "scripts/install_comfy.ps1", comfy)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in (comfy / "custom_nodes").iterdir()) == [name]
    assert (comfy / "user/default/workflows/FLUX3_PE/04_reference_editing.json").is_file()
    assert (comfy / "input/FLUX3_PE_reference_objects.png").is_file()
