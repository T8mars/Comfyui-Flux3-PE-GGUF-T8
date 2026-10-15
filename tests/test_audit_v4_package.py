from pathlib import Path
import os
import sys
from types import SimpleNamespace
import zipfile

import pytest

from test_audit_v2_package import package


def test_archive_write_failure_never_exposes_a_partial_formal_zip(package, monkeypatch):
    module, target, _ = package
    archive = module.ROOT / "release" / f"ComfyUI-Flux3-PE-{module.__version__}.zip"
    real_write = zipfile.ZipFile.write
    calls = []
    def fail_write(self, *args, **kwargs):
        if calls:
            raise OSError("simulated disk full during ZIP write")
        calls.append(True)
        return real_write(self, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "write", fail_write)
    monkeypatch.setattr(sys, "argv", ["package_release.py", "--zip"])
    with pytest.raises(OSError, match="disk full"):
        module.main()
    assert target.is_dir(), "Staged source must be preserved for diagnosis"
    assert not archive.exists(), "Partial bytes appeared under the formal release filename"
    assert len(list(archive.parent.glob("*.partial.zip"))) == 1


def test_formal_archive_created_during_build_is_preserved_at_publish(package, monkeypatch):
    module, _, _ = package
    archive = module.ROOT / "release" / f"ComfyUI-Flux3-PE-{module.__version__}.zip"
    real_write = zipfile.ZipFile.write
    def race_write(self, *args, **kwargs):
        if not archive.exists():
            archive.write_bytes(b"ANOTHER COMPLETE RELEASE MUST REMAIN")
        return real_write(self, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, "write", race_write)
    monkeypatch.setattr(sys, "argv", ["package_release.py", "--zip"])
    with pytest.raises(FileExistsError):
        module.main()
    assert archive.read_bytes() == b"ANOTHER COMPLETE RELEASE MUST REMAIN"
    assert len(list(archive.parent.glob("*.partial.zip"))) == 1


@pytest.mark.parametrize("competing_archive", [False, True])
def test_posix_publication_branch_is_create_exclusive(package, monkeypatch, competing_archive):
    module, _, _ = package
    monkeypatch.setattr(module, "os", SimpleNamespace(name="posix", link=os.link))
    archive = module.ROOT / "release" / f"ComfyUI-Flux3-PE-{module.__version__}.zip"
    if competing_archive:
        real_write = zipfile.ZipFile.write
        def create_competitor(self, *args, **kwargs):
            if not archive.exists(): archive.write_bytes(b"COMPETING FORMAL RELEASE")
            return real_write(self, *args, **kwargs)
        monkeypatch.setattr(zipfile.ZipFile, "write", create_competitor)
    monkeypatch.setattr(sys, "argv", ["package_release.py", "--zip"])
    if competing_archive:
        with pytest.raises(FileExistsError): module.main()
        assert archive.read_bytes() == b"COMPETING FORMAL RELEASE"
    else:
        module.main()
        with zipfile.ZipFile(archive) as output:
            assert output.testzip() is None
        assert not list(archive.parent.glob("*.partial.zip"))
