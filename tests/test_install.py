import os
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest


ROOT = Path(__file__).resolve().parents[1]
PWSH = shutil.which("pwsh")
if PWSH is None:
    bundled = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe"
    PWSH = str(bundled) if bundled.is_file() else None
pytestmark = pytest.mark.skipif(os.name != "nt" or PWSH is None, reason="Windows PowerShell installer")


def workspace():
    root = ROOT / "artifacts/review20/installer" / uuid.uuid4().hex
    comfy = root / "ComfyUI with spaces"
    comfy.mkdir(parents=True)
    (comfy / "main.py").write_text("", encoding="utf-8")
    (comfy / "custom_nodes").mkdir()
    return root, comfy


def install(script, comfy, *options):
    return subprocess.run([PWSH, "-NoProfile", "-File", str(script), "-ComfyUI", str(comfy), *map(str, options)], capture_output=True, text=True, encoding="utf-8", timeout=30)


@pytest.mark.parametrize("custom_directories", [False, True])
def test_junction_install_is_repeatable_and_backs_up_changed_examples(custom_directories):
    root, comfy = workspace()
    script = ROOT / "scripts/install_comfy.ps1"
    options = []
    user = comfy / "user"
    image = comfy / "input"
    if custom_directories:
        user, image = root / "custom user", root / "custom inputs"
        options = ["-UserDirectory", user, "-InputDirectory", image]
    first = install(script, comfy, *options)
    assert first.returncode == 0, first.stdout + first.stderr
    node = comfy / "custom_nodes/ComfyUI-Flux3-PE"
    assert node.resolve() == ROOT
    workflows = user / "default/workflows/FLUX3_PE"
    inputs = image
    assert sorted(path.name for path in workflows.glob("*.json")) == sorted(path.name for path in (ROOT / "examples").glob("*.json") if ".api." not in path.name)
    assert len(list(inputs.glob("*.png"))) == 2
    mtimes = {path: path.stat().st_mtime_ns for folder in (workflows, inputs) for path in folder.iterdir()}
    second = install(script, comfy, *options)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "Already installed" in second.stdout
    assert all(path.stat().st_mtime_ns == modified for path, modified in mtimes.items())
    assert not list(workflows.glob("*.bak")) and not list(inputs.glob("*.bak"))
    workflow = workflows / "01_prompt_enhancement.json"
    reference = inputs / "FLUX3_PE_reference_objects.png"
    workflow.write_bytes(b"user workflow should be preserved")
    reference.write_bytes(b"user reference should be preserved")
    third = install(script, comfy, *options)
    assert third.returncode == 0, third.stdout + third.stderr
    assert workflow.read_bytes() == (ROOT / "examples" / workflow.name).read_bytes()
    assert reference.read_bytes() == (ROOT / "examples/reference_images/reference_objects.png").read_bytes()
    assert [path.read_bytes() for path in workflows.glob("01_prompt_enhancement.json.*.bak")] == [b"user workflow should be preserved"]
    assert [path.read_bytes() for path in inputs.glob("FLUX3_PE_reference_objects.png.*.bak")] == [b"user reference should be preserved"]


def test_installer_already_inside_custom_nodes_installs_examples():
    _, comfy = workspace()
    node = comfy / "custom_nodes/ComfyUI-Flux3-PE"
    (node / "scripts").mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/install_comfy.ps1", node / "scripts/install_comfy.ps1")
    shutil.copytree(ROOT / "examples", node / "examples")
    for _ in range(2):
        result = install(node / "scripts/install_comfy.ps1", comfy)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "Already installed" in result.stdout
    assert (comfy / "user/default/workflows/FLUX3_PE/04_reference_editing.json").is_file()
    assert (comfy / "input/FLUX3_PE_reference_objects.png").is_file()
