import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux3_pe.protocol import prepare_request


def main():
    processor = AutoProcessor.from_pretrained(ROOT / "models/original", local_files_only=True)
    requests = [
        prepare_request('A cafe sign reading "OPEN 24 HOURS".'),
        prepare_request('海报写着“贞贞的AI世界”，三个红气球。'),
        prepare_request("A red balloon", boxes=[{"id": "balloon", "desc": "a red balloon", "bbox": [0.1, 0.1, 0.6, 0.6]}]),
        prepare_request("Move the red object to the right", [Image.new("RGB", (64, 32), "red")]),
        prepare_request("Combine the references", [Image.new("RGB", (64, 32), "red"), Image.new("RGB", (32, 64), "blue")], [{"id": "obj_1", "desc": "a green balloon", "from": "target", "src_bbox": None, "tgt_bbox": [0.1, 0.1, 0.6, 0.6]}]),
    ]
    records = []
    for index, request in enumerate(requests):
        rendered = processor.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        path = ROOT / f"artifacts/tokenizer-input-{index}.txt"
        path.write_bytes(rendered.encode("utf-8"))
        expected = processor.tokenizer.encode(rendered, add_special_tokens=False)
        completed = subprocess.run([str(ROOT / "vendor/llama.cpp/build-pe/bin/Release/test-pe-sampling.exe"), str(ROOT / "artifacts/pup-vocab.gguf"), str(path)], check=True, capture_output=True, text=True)
        actual = json.loads(completed.stdout)
        match = expected == actual
        records.append({"mode": request.mode, "tokens": len(expected), "identical": match})
        if not match:
            raise RuntimeError(f"Tokenizer mismatch in {request.mode}")
    (ROOT / "artifacts/tokenizer-parity.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
