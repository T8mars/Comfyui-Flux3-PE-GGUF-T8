import os
from pathlib import Path
import sys

import pytest

from test_audit_v2_identity import runner, install_fake_runner
from test_audit_v2_scripts import powershell, PWSH
from test_audit_v2_build_inputs import script


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name != 'nt' or PWSH is None, reason='Windows PowerShell installer')
def test_installer_requires_main_py_file_not_directory(tmp_path):
    comfy = tmp_path / 'Fake Comfy [qa] 测试'
    (comfy / 'main.py').mkdir(parents=True)
    result = powershell(ROOT / 'scripts/install_comfy.ps1', '-ComfyUI', comfy)
    assert result.returncode != 0 and 'main.py' in result.stderr
    assert not (comfy / 'custom_nodes').exists()
    assert not (comfy / 'user').exists()


@pytest.mark.parametrize('port', [-1, 0, 65536])
def test_launcher_invalid_port_never_creates_dirs_or_spawns(tmp_path, monkeypatch, port):
    module = script('launch_comfy')
    comfy = tmp_path / 'Comfy'
    comfy.mkdir()
    (comfy / 'main.py').write_text('')
    project = tmp_path / 'fresh-plugin'
    monkeypatch.setattr(module, 'ROOT', project)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('Invalid port spawned child')))
    monkeypatch.setattr(sys, 'argv', ['launch_comfy.py', '--comfy-ui', str(comfy), '--port', str(port)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2 and not project.exists()


@pytest.mark.parametrize('xml', ['<testsuites/>', '<testsuites><testsuite>', '<testsuites><testsuite><testcase><error/></testcase></testsuite></testsuites>'])
def test_empty_malformed_or_error_junit_never_passes_receipt(runner, monkeypatch, xml):
    import json
    module, _, receipt, _ = runner
    install_fake_runner(monkeypatch, runner, xml=xml)
    with pytest.raises(SystemExit):
        module.main()
    assert json.loads(receipt.read_text())['status'] == 'failed'
