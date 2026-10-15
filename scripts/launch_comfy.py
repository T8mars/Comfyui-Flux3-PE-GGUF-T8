"""Start only the requested isolated ComfyUI with literal paths and file logs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-ui", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8267)
    args = parser.parse_args()
    comfy = args.comfy_ui.resolve()
    if not (comfy / "main.py").is_file():
        parser.error("ComfyUI must contain main.py")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    paths = {name: ROOT / relative for name, relative in (
        ("logs", "logs"), ("input", "artifacts/comfy-input"),
        ("output", "artifacts/comfy-output"), ("user", "artifacts/comfy-user-review"),
    )}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(comfy / "main.py"), "--port", str(args.port),
        "--listen", "127.0.0.1", "--disable-all-custom-nodes", "--whitelist-custom-nodes",
        "ComfyUI-Flux3-PE", "--disable-api-nodes", "--offline", "--disable-auto-launch",
        "--output-directory", str(paths["output"]), "--input-directory", str(paths["input"]),
        "--user-directory", str(paths["user"]), "--database-url", "sqlite:///:memory:",
    ]
    with (paths["logs"] / f"comfy-review-{args.port}-out.log").open("w", encoding="utf-8") as stdout:
        with (paths["logs"] / f"comfy-review-{args.port}-err.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(
                command, cwd=comfy, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
    print(f"Started isolated review instance: port={args.port} pid={process.pid}", flush=True)


if __name__ == "__main__":
    main()
