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
    spec = backend.ModelSpec.load(ROOT / "models/gguf/flux3-klein-pup-Q4_K_M.gguf")
    prompt = "A small orange robot under a giant green fern, macro photograph."
    first, first_metadata = backend.enhance(spec, prompt, keep_loaded=True)
    process = backend._engine.process
    started = time.perf_counter()
    second, cached = backend.enhance(spec, prompt, keep_loaded=True)
    cache_seconds = time.perf_counter() - started
    if first != second or not cached["cache_hit"] or process.poll() is not None:
        raise RuntimeError("Persistent model/result cache failed")
    backend.release_model()
    if process.poll() is None:
        raise RuntimeError("PE process survived release")
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
    if not cancelled or process_cancel.poll() is None:
        raise RuntimeError("Cancellation failed to terminate the owned backend")
    report = {"cache_hit": True, "identical_cached_result": True, "cache_seconds": cache_seconds, "release_process_exited": process.poll() is not None, "cancelled": cancelled, "cancel_process_exited": process_cancel.poll() is not None, "first_generation_seconds": first_metadata["generation_seconds"]}
    report.update({"version": __version__, "source_identity": SOURCE_IDENTITY})
    (ROOT / "artifacts/lifecycle-smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        backend.release_model()
