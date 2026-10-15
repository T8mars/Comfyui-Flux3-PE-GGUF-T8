"""Run the whole test suite and bind its JUnit result to the tested sources."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux3_pe import __version__
from flux3_pe.engine import sha256_file
from flux3_pe.identity import source_identity, validation_inputs


def inputs():
    bound = validation_inputs(ROOT)
    for name in ("pyproject.toml", "requirements.txt"):
        bound[name] = sha256_file(ROOT / name)
    return bound


def main():
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    receipt = artifacts / "test-identity.json"
    junit = artifacts / "unit-tests.xml"
    before = inputs()
    report = {"status": "running", "version": __version__, "started_utc": datetime.now(timezone.utc).isoformat(), "source_identity": source_identity(), "inputs_sha256": before}

    def save():
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(receipt)

    save()
    # A successful process exit alone must never rebind yesterday's green XML.
    junit.unlink(missing_ok=True)
    # This command attests the entire project suite. An inherited developer
    # selection such as PYTEST_ADDOPTS='-k smoke' must not narrow that claim.
    environment = {**os.environ, "PYTEST_ADDOPTS": ""}
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml=" + str(junit)], cwd=ROOT, env=environment)
    report["exit_code"] = result.returncode
    report["status"] = "failed"
    if result.returncode == 0 and before == inputs():
        try:
            unit = ET.parse(junit).getroot()
            report["tests"] = len(unit.findall(".//testcase"))
            if report["tests"] and not any(unit.findall(".//" + kind) for kind in ("failure", "error", "skipped")):
                report["status"] = "passed"
                report["junit_sha256"] = sha256_file(junit)
        except (OSError, ET.ParseError):
            report["reason"] = "The test process did not write a complete new JUnit report"
    save()
    print(json.dumps({key: report[key] for key in ("status", "version", "source_identity")}, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(result.returncode or 1)


if __name__ == "__main__":
    main()
