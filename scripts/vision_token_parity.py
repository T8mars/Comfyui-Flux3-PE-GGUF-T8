import json
from pathlib import Path
import sys

from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import fixtures
from flux3_pe.protocol import prepare_request


def main():
    processor = AutoProcessor.from_pretrained(ROOT / "models/original", local_files_only=True)
    data = json.loads((ROOT / "artifacts/comparison.json").read_text(encoding="utf-8"))
    images, records = fixtures(), []
    for item in data["results"]:
        if item["model"] != "BF16" or not item["passed"]:
            continue
        case = item["request"]
        request = prepare_request(case["prompt"], [images[key] for key in case["images"]], case["boxes"])
        encoded = processor.apply_chat_template(request.messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors="pt")
        expected = encoded["input_ids"].shape[1]
        actual = item["metadata"]["usage"]["prompt_tokens"]
        record = {"case": item["case"], "reference_count": len(case["images"]), "hf_tokens": expected, "gguf_runtime_tokens": actual, "identical_length": expected == actual}
        records.append(record)
        if expected != actual:
            raise RuntimeError(f"Vision token budget mismatch: {record}")
    (ROOT / "artifacts/vision-token-parity.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
