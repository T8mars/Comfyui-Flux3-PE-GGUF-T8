import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux3_pe.engine import sha256_file


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models/gguf"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for entry in [*manifest["models"].values(), manifest["mmproj"]]:
        path = folder / entry["file"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"Artifact validation failed: {path}")
        print("PASS", path.name, entry["bytes"])


if __name__ == "__main__":
    main()
