"""Run launch/install helpers only against disposable CPU stub directories."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
PWSH = shutil.which("pwsh")
if PWSH is None:
    candidate = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe"
    PWSH = str(candidate) if candidate.is_file() else None
pytestmark = pytest.mark.skipif(os.name != "nt" or PWSH is None, reason="Windows native helper contracts")


def powershell(script, *arguments):
    return subprocess.run([PWSH, "-NoProfile", "-File", str(script), *map(str, arguments)], capture_output=True, text=True, encoding="utf-8", timeout=15)


def test_start_helper_creates_fresh_dirs_and_preserves_spaced_literal_arguments(tmp_path):
    plugin = tmp_path / "node with spaces [qa] 测试"
    scripts = plugin / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "start_test_comfy.ps1"
    shutil.copy2(ROOT / "scripts/start_test_comfy.ps1", script)
    shutil.copy2(ROOT / "scripts/launch_comfy.py", scripts / "launch_comfy.py")
    comfy = tmp_path / "Comfy with spaces [qa]"
    comfy.mkdir()
    args_path = comfy / "observed-args.json"
    (comfy / "main.py").write_text(
        "import json,pathlib,sys; print('stub stdout',flush=True); print('stub stderr',file=sys.stderr,flush=True); "
        "pathlib.Path(__file__).with_name('observed-args.json').write_text(json.dumps({'args':sys.argv[1:],'cwd':str(pathlib.Path.cwd())}))",
        encoding="utf-8",
    )
    result = powershell(script, "-Port", 17958, "-ComfyUI", comfy, "-Python", sys.executable)
    assert result.returncode == 0, result.stdout + result.stderr
    deadline = time.monotonic() + 3
    while not args_path.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert args_path.is_file(), result.stdout + result.stderr
    observed = json.loads(args_path.read_text())
    args = observed["args"]
    assert Path(observed["cwd"]) == comfy
    for option, expected in (
        ("--port", "17958"),
        ("--listen", "127.0.0.1"),
        ("--output-directory", str(plugin / "artifacts/comfy-output")),
        ("--input-directory", str(plugin / "artifacts/comfy-input")),
        ("--user-directory", str(plugin / "artifacts/comfy-user-review")),
    ):
        assert args[args.index(option) + 1] == expected
    for name in ("logs", "artifacts/comfy-input", "artifacts/comfy-output", "artifacts/comfy-user-review"):
        assert (plugin / name).is_dir()
    assert "stub stdout" in (plugin / "logs/comfy-review-17958-out.log").read_text()
    assert "stub stderr" in (plugin / "logs/comfy-review-17958-err.log").read_text()


def test_installer_preserves_literal_brackets_and_unicode_in_custom_paths(tmp_path):
    comfy = tmp_path / "Comfy [qa] 测试"
    comfy.mkdir()
    (comfy / "main.py").write_text("")
    (comfy / "custom_nodes").mkdir()
    user, inputs = tmp_path / "用户 [qa]", tmp_path / "输入 [qa]"
    result = powershell(ROOT / "scripts/install_comfy.ps1", "-ComfyUI", comfy, "-UserDirectory", user, "-InputDirectory", inputs)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (comfy / "custom_nodes/ComfyUI-Flux3-PE").resolve() == ROOT
    assert (user / "default/workflows/FLUX3_PE/01_prompt_enhancement.json").is_file()
    assert (inputs / "FLUX3_PE_reference_objects.png").is_file()


def test_installer_rejects_invalid_comfy_before_changing_destination(tmp_path):
    comfy = tmp_path / "not Comfy"
    comfy.mkdir()
    result = powershell(ROOT / "scripts/install_comfy.ps1", "-ComfyUI", comfy)
    assert result.returncode != 0 and "main.py" in result.stderr
    assert list(comfy.iterdir()) == []


def test_installer_refuses_to_overwrite_existing_unrelated_node(tmp_path):
    comfy = tmp_path / "Comfy"
    destination = comfy / "custom_nodes/ComfyUI-Flux3-PE"
    destination.mkdir(parents=True)
    marker = destination / "user.txt"
    marker.write_text("keep user files")
    (comfy / "main.py").write_text("")
    result = powershell(ROOT / "scripts/install_comfy.ps1", "-ComfyUI", comfy)
    assert result.returncode != 0 and "Destination already exists" in result.stderr
    assert marker.read_text() == "keep user files"
    assert not (comfy / "user").exists()
