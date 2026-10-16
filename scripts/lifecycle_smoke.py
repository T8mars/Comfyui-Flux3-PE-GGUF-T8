import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import flux3_pe.engine as backend
from flux3_pe import __version__
from flux3_pe.identity import SOURCE_IDENTITY
from flux3_pe.protocol import prepare_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["auto", "llama-cpp-python", "llama-server"], default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/lifecycle-smoke.json")
    args = parser.parse_args()
    spec = backend.ModelSpec.load(ROOT / "models/gguf/flux3-klein-pup-Q4_K_M.gguf", backend=args.backend)
    prompt = "A small orange robot under a giant green fern, macro photograph."
    first, first_metadata = backend.enhance(spec, prompt, keep_loaded=True)
    process = backend._engine.process
    resident = backend._engine
    started = time.perf_counter()
    second, cached = backend.enhance(spec, prompt, keep_loaded=True)
    cache_seconds = time.perf_counter() - started
    if first != second or not cached["cache_hit"] or (process is not None and process.poll() is not None):
        raise RuntimeError("Persistent model/result cache failed")
    backend.release_model()
    if backend._engine is not None or (process is not None and process.poll() is None) or getattr(resident, "is_loaded", False):
        raise RuntimeError("PE backend survived release")
    if spec.backend == "llama-cpp-python":
        from flux3_pe.wheel_backend import WheelEngine
        engine = WheelEngine(spec)
    else:
        engine = backend.LlamaEngine(spec)
    engine.start()
    process_cancel = engine.process
    deadline = time.monotonic() + 0.5

    def interrupt():
        if time.monotonic() >= deadline:
            raise InterruptedError("Validation cancellation")

    cancelled = False
    try:
        engine.complete(prepare_request("An intricate aerial city panorama with many distinct buildings, courtyards and streets; describe the whole composition in detail.").payload(), interrupt)
    except InterruptedError:
        cancelled = True
    finally:
        engine.close()
    if not cancelled or (process_cancel is not None and process_cancel.poll() is None) or getattr(engine, "is_loaded", False):
        raise RuntimeError("Cancellation failed to release the owned backend")
    report = {"backend": spec.backend, "cache_hit": True, "identical_cached_result": True, "cache_seconds": cache_seconds, "released": True, "release_process_exited": process.poll() is not None if process is not None else None, "cancelled": cancelled, "cancel_process_exited": process_cancel.poll() is not None if process_cancel is not None else None, "native_handles_released": not getattr(engine, "is_loaded", False), "first_generation_seconds": first_metadata["generation_seconds"]}
    report.update({"version": __version__, "source_identity": SOURCE_IDENTITY})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        backend.release_model()
