"""Identity of the Python code actually imported by a PE backend."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_FILES = (
    "__init__.py", "nodes.py", "flux3_pe/__init__.py", "flux3_pe/engine.py",
    "flux3_pe/protocol.py", "flux3_pe/bbox.py", "flux3_pe/identity.py",
    "flux3_pe/resources/system_prompts.json",
)


def source_identity(root=ROOT):
    digest = hashlib.sha256()
    for name in sorted(CORE_FILES):
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return digest.hexdigest()


def validation_inputs(root=ROOT):
    paths = {root / name for name in CORE_FILES}
    paths.update(path for folder in ("tests", "scripts") for path in (root / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


# Capture once at import; an already-running ComfyUI must not claim new disk code.
SOURCE_IDENTITY = source_identity()
