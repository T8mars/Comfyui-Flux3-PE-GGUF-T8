"""Audit the actual artifacts and test evidence before packaging."""
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux3_pe.engine import ModelSpec, sha256_file
from flux3_pe import __version__
from flux3_pe.identity import source_identity
from scripts.run_checks import inputs as test_inputs
from scripts.evaluate import cases, evaluation_input_identity


def read(name):
    return json.loads((ROOT / "artifacts" / name).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    evaluation = read("evaluation-q4.json")
    records = evaluation["results"]
    identity = source_identity()
    require(evaluation.get("input_identity") == evaluation_input_identity(), "The evaluated corpus or execution script changed")
    require([record["request"] for record in records] == cases(), "Evaluation requests do not match the complete current corpus")
    require(len(records) == 40 and all(record["passed"] for record in records), "The final Q4 regression did not pass all 40 tasks")
    require(all(record["model"] == "Q4_K_M" and record["metadata"]["model"] == "flux3-klein-pup-Q4_K_M.gguf" for record in records), "Regression evidence does not describe the release quantization")
    require(all(record["metadata"].get("source_identity") == identity and record["metadata"].get("plugin_version") == __version__ for record in records), "PE regression evidence belongs to different Python sources")
    require({record["metadata"]["mode"] for record in records} == {"t2i", "i2i", "t2i_bbox", "i2i_bbox"}, "Missing a PE mode")
    require(any(len(record["request"]["images"]) == 10 for record in records), "Ten-reference path was not tested")
    require(all(len(record["metadata"]["attempts"]) <= 2 for record in records), "Recovery exceeded its limit")
    tokens, vision = read("tokenizer-parity.json"), read("vision-token-parity.json")
    require(all(record["identical"] for record in tokens), "Tokenizer parity failed")
    require(all(record["identical_length"] for record in vision), "Vision token count parity failed")
    require({record["reference_count"] for record in vision}.issuperset({0, 1, 2, 10}), "Vision reference-count coverage missing")
    lifecycle = read("lifecycle-smoke.json")
    require(lifecycle.get("version") == __version__ and lifecycle.get("source_identity") == identity, "Lifecycle evidence belongs to different Python sources")
    require(all(lifecycle[key] for key in ("cache_hit", "identical_cached_result", "release_process_exited", "cancelled", "cancel_process_exited")), "Lifecycle check failed")
    native = (ROOT / "logs/pup-native-test.log").read_text(encoding="utf-8", errors="replace")
    require("PASS: default, prompt exclusion, generated penalty, full history, clone, reset" in native, "Native sampling regression failed")
    unit = ET.parse(ROOT / "artifacts/unit-tests.xml").getroot()
    test_identity = read("test-identity.json")
    require(test_identity.get("status") == "passed" and test_identity.get("version") == __version__ and test_identity.get("source_identity") == identity, "Tests did not pass against this Python source version")
    require(test_identity.get("inputs_sha256") == test_inputs() and test_identity.get("junit_sha256") == sha256_file(ROOT / "artifacts/unit-tests.xml"), "Tests or their inputs changed after the verified run")
    require(len(unit.findall(".//testcase")) >= 96 and not unit.findall(".//failure") and not unit.findall(".//error") and not unit.findall(".//skipped"), "Unit tests failed, were skipped, or the review regressions were not run")
    canvas = read("canvas-qa/execute.json")
    require(canvas.get("version") == __version__ and canvas.get("source_identity") == identity, "Canvas evidence belongs to another Python source version")
    qa_script_hash = sha256_file(ROOT / "scripts/canvas_qa.cjs")
    require(canvas.get("qa_script_sha256") == qa_script_hash, "Canvas evidence belongs to different QA checks")
    canvas_inputs = [*(ROOT / "examples").glob("*.api.json"), *(ROOT / "examples/reference_images").glob("*.png")]
    require(canvas.get("inputs_sha256") == {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in canvas_inputs}, "Canvas workflow inputs changed after the run")
    expected_names = {"01_prompt_enhancement", "02_layout_enhancement", "03_pe_to_flux3", "04_reference_editing"}
    require({item["name"] for item in canvas["saved"]} == expected_names and not canvas["errors"], "Editable canvas coverage or frontend check failed")
    require(all(item["dragged"] and item["reopened"] for item in canvas["saved"]), "Canvas drag/save/reopen was not checked")
    for item in canvas["saved"]:
        require(sha256_file(ROOT / "examples" / (item["name"] + ".json")) == sha256_file(ROOT / "artifacts/canvas-qa" / (item["name"] + ".json")), "A shipped canvas differs from its tested saved file")
    require(len(canvas["runs"]) == 7 and {item["name"] for item in canvas["runs"]} == expected_names, "Canvas Run coverage is incomplete")
    require(all(item["history"]["status"]["status_str"] == "success" and "2" not in item["cached"] for item in canvas["runs"]), "A canvas run failed or PE was skipped")
    require(all(item["details"].get("plugin_version") == __version__ and item["details"].get("source_identity") == identity for item in canvas["runs"]), "A canvas run used stale backend Python code")
    repeats = [item for item in canvas["runs"] if item["name"] == "01_prompt_enhancement"]
    require([item["details"]["cache_hit"] for item in repeats] == [False, True, False, False], "Repeated canvas execution did not honor both cache modes")
    require(all(item["history"]["prompt"][2]["2"]["inputs"]["use_cache"] is False for item in repeats[2:]), "Uncached canvas runs did not disable the cache")
    editing = next(item for item in canvas["runs"] if item["name"] == "04_reference_editing")
    require(editing["details"]["mode"] == "i2i" and len(editing["details"]["reference_sizes"]) == 2, "Reference canvas did not actually use both images")
    generation = next(item for item in canvas["runs"] if item["name"] == "03_pe_to_flux3")
    require(generation["history"]["outputs"].get("19", {}).get("images"), "Canvas did not produce an image")
    png = read("canvas-qa/png-roundtrip.json")
    require(png.get("qa_script_sha256") == qa_script_hash, "PNG roundtrip evidence belongs to different QA checks")
    image = generation["history"]["outputs"]["19"]["images"][0]
    image_path = ROOT / "artifacts/comfy-output" / image["subfolder"] / image["filename"]
    require(png.get("prompt_id") == generation["prompt_id"] and png.get("version") == __version__ and png.get("source_identity") == identity and png.get("image_sha256") == sha256_file(image_path), "PNG roundtrip evidence does not match the tested canvas run")
    require(png["frontend_imported"] and png["graph_matches_execution"] and not png["errors"], "The generated PNG did not restore its complete canvas")
    rounds = read("review20/rounds.json")
    require([item["round"] for item in rounds] == list(range(1, 21)), "The requested twenty review rounds are incomplete")
    previous_rounds = read("audit20-v2/rounds.json")
    require([item["round"] for item in previous_rounds] == list(range(1, 21)), "The second independent twenty-round audit is incomplete")
    third_rounds = read("audit20-v3/rounds.json")
    require([item["round"] for item in third_rounds] == list(range(1, 21)), "The third independent twenty-round audit is incomplete")
    fourth_rounds = read("audit20-v4/rounds.json")
    require([item["round"] for item in fourth_rounds] == list(range(1, 21)), "The fourth independent twenty-round audit is incomplete")
    current_rounds = read("audit20-v5/rounds.json")
    require([item["round"] for item in current_rounds] == list(range(1, 21)), "The fifth independent twenty-round audit is incomplete")
    comfy = read("comfy-smoke.json")
    require({entry["workflow"] for entry in comfy} == {"text", "layout", "two_references", "pe_to_image"}, "ComfyUI workflow coverage missing")
    require(all(entry["status"] == "success" for entry in comfy), "A ComfyUI workflow failed")
    require(all(json.loads(entry["history"]["outputs"]["4"]["text"][0]).get("source_identity") == identity for entry in comfy), "A ComfyUI workflow record belongs to old backend sources")
    fresh = read("comfy-e2e-fresh.json")
    require(len(fresh) == 1 and fresh[0]["status"] == "success", "Fresh PE-to-image workflow failed")
    require(fresh[0]["prompt_id"] == generation["prompt_id"], "Fresh PE-to-image evidence differs from the tested canvas generation")
    graph = fresh[0]["history"]["prompt"][2]
    require(graph["2"]["inputs"]["use_cache"] is False, "Fresh end-to-end run used result cache")
    cached_nodes = set()
    for kind, details in fresh[0]["history"]["status"].get("messages", []):
        if kind == "execution_cached":
            cached_nodes.update(details["nodes"])
    require("2" not in cached_nodes, "The fresh PE node was skipped by ComfyUI cache")
    images = fresh[0]["history"]["outputs"]["19"]["images"]
    for image in images:
        require((ROOT / "artifacts/comfy-output" / image["subfolder"] / image["filename"]).is_file(), "Rendered output image is missing")
    manifest = json.loads((ROOT / "models/gguf/manifest.json").read_text(encoding="utf-8"))
    require(manifest["models"]["BF16"]["tensor_count"] + manifest["mmproj"]["source_tensor_count"] == 760, "Source tensor accounting failed")
    spec = ModelSpec.load(ROOT / "models/gguf/flux3-klein-pup-Q4_K_M.gguf")
    comparison = read("comparison.json")
    benchmark = {}
    for quant in ("BF16", "Q4_K_M"):
        selected = [record for record in comparison["results"] if record["model"] == quant]
        require(len(selected) == 8 and all(record["passed"] for record in selected), "Controlled benchmark samples incomplete")
        benchmark[quant] = {"samples": len(selected), "median_generation_seconds": statistics.median(record["metadata"]["generation_seconds"] for record in selected), "median_decode_tokens_per_second": statistics.median(record["metadata"]["timings"]["predicted_per_second"] for record in selected), "gpu_peak_mib_total": max(record["gpu_peak_mib_total"] for record in selected), "model_bytes": manifest["models"][quant]["bytes"]}
    summary = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "release_quant": "Q4_K_M", "version": __version__,
        "source_identity": identity, "test_identity": test_identity,
        "source": manifest["source"], "runtime_commit": manifest["converter_commit"],
        "unit_tests": len(unit.findall(".//testcase")), "native_sampling_test": True,
        "functional_tasks": len(records), "functional_passed": sum(record["passed"] for record in records),
        "recovered_cases": [record["case"] for record in records if record["metadata"]["recovered"]],
        "tokenizer_parity": tokens, "vision_token_parity": vision,
        "lifecycle": lifecycle, "comfy_workflows": [entry["workflow"] for entry in comfy],
        "review_rounds": len(current_rounds), "previous_review_rounds": len(fourth_rounds), "earlier_review_rounds": len(third_rounds), "review_batches": {"review20": len(rounds), "audit20-v2": len(previous_rounds), "audit20-v3": len(third_rounds), "audit20-v4": len(fourth_rounds), "audit20-v5": len(current_rounds)}, "canvas_saved": canvas["saved"],
        "generated_png_restores_canvas": True,
        "canvas_runs": [{"name": item["name"], "iteration": item["iteration"], "prompt_id": item["prompt_id"], "cache_hit": item["details"]["cache_hit"]} for item in canvas["runs"]],
        "fresh_end_to_end_prompt_id": fresh[0]["prompt_id"], "rendered_images": images,
        "benchmark": benchmark, "benchmark_note": "Eight tasks on the same llama.cpp CUDA backend; natural output lengths differ; startup excluded from generation time. Whole-GPU peaks include the host context. Later functional runs can share the GPU with other applications.",
        "known_limits": ["Q4_0 comparison artifact failed a multi-reference layout binding case and is not the release default.", "One Q4_K_M Chinese edit required a bounded recovery attempt; its full latency is recorded.", "That recovered roof edit requests the correct red target but inaccurately describes the original brown roof color; see audit20-v2 round17 semantic review.", "Automatic checks cover structure, locked geometry, and sampled literal text, not universal semantic equivalence."],
        "source_files_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in [*(ROOT / name for name in ("__init__.py", "nodes.py", "pyproject.toml", "requirements.txt", "README.md", "LICENSE", "NOTICE")), *(path for folder in ("flux3_pe", "scripts", "tests", "patches", "licenses") for path in (ROOT / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts)]},
        "workflow_files_sha256": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in (ROOT / "examples").rglob("*") if path.is_file()},
        "model_identity": spec.revision,
    }
    (ROOT / "artifacts/validation-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("release_quant", "unit_tests", "functional_tasks", "functional_passed", "recovered_cases", "comfy_workflows")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
