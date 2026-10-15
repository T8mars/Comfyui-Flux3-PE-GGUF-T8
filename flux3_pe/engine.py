from __future__ import annotations

import atexit
import copy
import hashlib
import json
import os
import secrets
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import requests

from .protocol import PEError, SOURCE_REVISION, finish_caption, has_repetition, prepare_request, recovery_payload
from .identity import SOURCE_IDENTITY
from . import __version__

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=32)
def runtime_checksum(path: Path, size: int, modified_ns: int) -> str:
    return sha256_file(path)


@dataclass(frozen=True)
class ModelSpec:
    model: Path
    mmproj: Path
    server: Path
    revision: str
    context: int = 8192
    gpu_layers: str = "all"

    @classmethod
    def load(cls, model: str | Path, server: str | Path | None = None, context: int = 8192, gpu_layers: str = "all"):
        if isinstance(context, bool) or not isinstance(context, int) or context < 4096:
            raise ValueError("PUP context must be an integer of at least 4096 tokens")
        model = Path(model).resolve()
        manifest_path = model.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source"]["revision"] != SOURCE_REVISION:
            raise PEError("This model manifest does not describe the supported PUP revision")
        if manifest["mmproj"].get("source_revision") != SOURCE_REVISION:
            raise PEError("The vision projector must use the supported PUP revision")
        entry = next((item for item in manifest["models"].values() if item["file"] == model.name), None)
        if entry is None:
            raise PEError(f"Model is not registered in {manifest_path}")
        mmproj = (model.parent / manifest["mmproj"]["file"]).resolve()
        if mmproj.parent != model.parent:
            raise PEError("mmproj must be beside its model manifest")
        for path, metadata in ((model, entry), (mmproj, manifest["mmproj"])):
            if not path.is_file() or path.stat().st_size != metadata["bytes"]:
                raise PEError(f"Model artifact is missing or incomplete: {path}")
        server = Path(server or os.environ.get("FLUX3_PE_LLAMA_SERVER", ROOT / "runtime/bin/llama-server.exe")).resolve()
        runtime_manifest = json.loads((server.parent / "flux3-pe-runtime.json").read_text(encoding="utf-8"))
        if runtime_manifest.get("penalties_generated_only") is not True:
            raise PEError("Install the FLUX3 PE runtime with generated-token-only penalties")
        if not server.is_file():
            raise PEError(f"llama-server is missing: {server}")
        if sha256_file(server) != runtime_manifest["server_sha256"]:
            raise PEError("llama-server checksum does not match its runtime manifest")
        checksums = runtime_manifest.get("core_sha256", {})
        if not {"llama-common.dll", "llama-server-impl.dll"}.issubset(checksums):
            raise PEError("Runtime manifest must bind the patched shared libraries")
        for name, expected in checksums.items():
            library = (server.parent / name).resolve()
            if library.parent != server.parent or not library.is_file():
                raise PEError(f"Invalid or missing runtime library: {name}")
            stat = library.stat()
            if runtime_checksum(library, stat.st_size, stat.st_mtime_ns) != expected:
                raise PEError(f"Runtime library checksum failed: {name}")
        revision = hashlib.sha256(json.dumps([entry["sha256"], model.stat().st_mtime_ns, manifest["mmproj"]["sha256"], mmproj.stat().st_mtime_ns, runtime_manifest["server_sha256"], checksums], sort_keys=True).encode()).hexdigest()
        return cls(model, mmproj, server, revision, context, gpu_layers)


class LlamaEngine:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.process: subprocess.Popen | None = None
        self.session = requests.Session()
        self.session.trust_env = False
        self.url = ""
        self._log = None
        self._request_worker: threading.Thread | None = None
        self._unusable = False
        self._cleanup_failed = False

    def start(self, interrupt: Callable[[], None] = lambda: None):
        if self._cleanup_failed or (self._request_worker is not None and self._request_worker.is_alive()):
            raise PEError("The previous PE request is awaiting cleanup; release the owned backend before retrying")
        if self.process is not None:
            if self.process.poll() is None:
                return
            # A resident backend can exit independently. Reclaim its resources
            # before assigning a new subprocess or overwriting the log handle.
            self.close()
            self.session = requests.Session()
            self.session.trust_env = False
        self._unusable = False
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        key = secrets.token_urlsafe(24)
        self.session.headers["Authorization"] = f"Bearer {key}"
        self.url = f"http://127.0.0.1:{port}"
        log_dir = Path(os.environ.get("FLUX3_PE_LOG_DIR", ROOT / "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"llama-{os.getpid()}-{time.time_ns()}.log"
        self._log = log_path.open("w", encoding="utf-8")
        args = [
            str(self.spec.server), "--model", str(self.spec.model), "--mmproj", str(self.spec.mmproj),
            "--host", "127.0.0.1", "--port", str(port), "--api-key", key,
            "--ctx-size", str(self.spec.context), "--parallel", "1", "--n-gpu-layers", self.spec.gpu_layers,
            "--flash-attn", "on", "--no-warmup", "--no-cache-prompt", "--no-context-shift",
            "--no-webui", "--alias", "flux3-pe",
        ]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        deadline = time.monotonic() + 180
        try:
            interrupt()
            self.process = subprocess.Popen(args, cwd=self.spec.server.parent, stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags)
            while time.monotonic() < deadline:
                interrupt()
                if self.process.poll() is not None:
                    raise PEError(f"llama-server exited with {self.process.returncode}; see {log_path}")
                try:
                    response = self.session.get(self.url + "/health", timeout=0.5)
                    if response.status_code == 200:
                        props = self.session.get(self.url + "/props", timeout=5).json()
                        params = props.get("default_generation_settings", {}).get("params", {})
                        if "penalties_generated_only" not in params:
                            raise PEError("The running backend lacks the required PE sampling extension")
                        return
                except (requests.ConnectionError, requests.Timeout):
                    pass
                time.sleep(0.1)
            raise PEError(f"llama-server startup timed out; see {log_path}")
        except BaseException:
            self.close(wait_timeout=0.25)
            raise

    def complete(self, payload: dict, interrupt: Callable[[], None] = lambda: None) -> dict:
        if self._unusable or (self._request_worker is not None and self._request_worker.is_alive()):
            raise PEError("The previous PE request is awaiting cleanup; refusing another POST")
        completed = threading.Event()
        outcome = {}
        session, url = self.session, self.url

        def request():
            try:
                with session.post(url + "/v1/chat/completions", json=payload, timeout=(10, 600)) as response:
                    if not response.ok:
                        raise PEError(f"PUP backend HTTP {response.status_code}: {response.text[:1500]}")
                    outcome["value"] = response.json()
            except BaseException as error:
                outcome["error"] = error
            finally:
                completed.set()

        # A server that cannot be terminated may leave its socket pending.
        # Such a request must neither block cancellation nor Python shutdown.
        worker = threading.Thread(target=request, name="Flux3PEHTTP", daemon=True)
        self._request_worker = worker
        deadline = time.monotonic() + 600
        try:
            interrupt()
            worker.start()
            while True:
                interrupt()
                if time.monotonic() > deadline:
                    raise PEError("PE request exceeded 600 seconds")
                if completed.wait(timeout=0.1):
                    if "error" in outcome:
                        raise outcome["error"]
                    return outcome["value"]
        except BaseException:
            self._unusable = True
            self.close(wait_timeout=0.25)
            raise
        finally:
            if worker.ident is not None:
                worker.join(timeout=0.25)
            if not worker.is_alive():
                self._request_worker = None

    def close(self, wait_timeout=10):
        self._unusable = True
        self._cleanup_failed = False
        try:
            if self.process is not None:
                process = self.process
                errors = []
                try:
                    exited = process.poll() is not None
                except OSError as error:
                    errors.append(error)
                    exited = False
                if not exited:
                    # A terminate race must not prevent a bounded wait/kill.
                    # Retain ownership until termination is actually confirmed.
                    for stop_name in ("terminate", "kill"):
                        try:
                            getattr(process, stop_name)()
                        except OSError as error:
                            errors.append(error)
                        try:
                            process.wait(timeout=wait_timeout)
                            exited = True
                            break
                        except (OSError, subprocess.TimeoutExpired) as error:
                            errors.append(error)
                if not exited:
                    self._cleanup_failed = True
                    raise PEError("Cannot confirm owned llama-server exit; its handle is retained for cleanup") from errors[-1]
                self.process = None
        finally:
            try:
                if self._log is not None:
                    log = self._log
                    log.close()
                    self._log = None
            finally:
                self.session.close()


_lock = threading.Lock()
_engine: LlamaEngine | None = None
_results: dict[str, tuple[str, dict]] = {}


def _close_engine():
    """Called only while holding _lock; never lose an unconfirmed process."""
    global _engine
    if _engine is not None:
        current = _engine
        try:
            current.close()
        finally:
            if getattr(current, "process", None) is None:
                _engine = None


def release_model():
    with _lock:
        _close_engine()


atexit.register(release_model)


def enhance(spec: ModelSpec, prompt: str, images=None, boxes=None, canvas=None, max_tokens=3584, keep_loaded=False, use_cache=True, interrupt: Callable[[], None] = lambda: None, recover_invalid=True, before_load: Callable[[], None] | None = None) -> tuple[str, dict]:
    global _engine
    interrupt()
    prepared = prepare_request(prompt, images, boxes, canvas)
    # Validate output controls before touching resident engines or Comfy models.
    payload = prepared.payload(max_tokens)
    key = hashlib.sha256(f"{spec.revision}:{spec.context}:{spec.gpu_layers}:{prepared.fingerprint}:{max_tokens}:{recover_invalid}".encode()).hexdigest()
    while not _lock.acquire(timeout=0.1):
        interrupt()
    succeeded = False
    try:
        interrupt()
        # A result cache is independent of backend ownership. Do not report a
        # successful request while an earlier owned process is still awaiting
        # explicit release after unconfirmed cleanup.
        if _engine is not None and getattr(_engine, "_cleanup_failed", False):
            raise PEError("The previous PE backend is awaiting cleanup; release the owned backend before retrying")
        if use_cache and key in _results:
            caption, metadata = _results[key]
            succeeded = True
            metadata = copy.deepcopy(metadata)
            metadata.update(cache_hit=True, load_seconds=0.0, generation_seconds=0.0, timings={})
            return caption, metadata
        if _engine is not None and _engine.spec != spec:
            _close_engine()
        if _engine is None:
            _engine = LlamaEngine(spec)
        started = time.monotonic()
        process = getattr(_engine, "process", None)
        if before_load is not None and (process is None or process.poll() is not None):
            before_load()
        _engine.start(interrupt)
        loaded = time.monotonic()
        attempts = []
        recovery_reason = None
        for attempt in range(2 if recover_invalid else 1):
            response = _engine.complete(payload, interrupt)
            choice = response["choices"][0]
            attempts.append({"finish_reason": choice["finish_reason"], "usage": response.get("usage", {})})
            try:
                caption = finish_caption(choice["message"].get("content") or "", prepared, choice["finish_reason"])
                break
            except PEError as error:
                retryable = error.kind in ("layout", "invalid_output") or (error.kind == "truncated" and has_repetition(error.partial_text))
                if attempt or not recover_invalid or not retryable:
                    raise
                payload = recovery_payload(prepared, max_tokens)
                recovery_reason = error.kind
        metadata = {
            "mode": prepared.mode, "reference_sizes": prepared.reference_sizes,
            "plugin_version": __version__, "source_identity": SOURCE_IDENTITY,
            "source_revision": SOURCE_REVISION, "model": spec.model.name, "model_revision": spec.revision,
            "gpu_layers": spec.gpu_layers, "context_size": spec.context,
            "load_seconds": round(loaded - started, 4), "generation_seconds": round(time.monotonic() - loaded, 4),
            "usage": response.get("usage", {}), "timings": response.get("timings", {}), "cache_hit": False,
            "penalties_generated_only": True,
            "recovered": recovery_reason is not None, "recovery_reason": recovery_reason, "attempts": attempts,
        }
        if use_cache:
            if len(_results) >= 32:
                _results.pop(next(iter(_results)))
            # Callers own returned dictionaries. Keep only small metadata in the
            # cache, isolated from both initial results and later cache hits.
            _results[key] = (caption, copy.deepcopy(metadata))
        succeeded = True
        return caption, metadata
    finally:
        try:
            if (not keep_loaded or not succeeded) and _engine is not None and not getattr(_engine, "_cleanup_failed", False):
                _close_engine()
        finally:
            _lock.release()
