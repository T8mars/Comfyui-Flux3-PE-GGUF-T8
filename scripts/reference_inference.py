"""Generate an independent BF16 Transformers reference with official PUP penalties."""
import argparse
import json
from pathlib import Path
import sys
import time
from PIL import Image

import torch
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux3_pe.protocol import prepare_request


class GeneratedPresencePenalty(LogitsProcessor):
    def __init__(self, prompt_length, penalty):
        self.length = prompt_length
        self.penalty = penalty
        self.seen = None

    def __call__(self, input_ids, scores):
        if self.seen is None:
            self.seen = torch.zeros_like(scores, dtype=torch.bool)
        if input_ids.shape[1] > self.length:
            self.seen.scatter_(1, input_ids[:, self.length:], True)
            self.length = input_ids.shape[1]
        return scores - self.seen.to(scores.dtype) * self.penalty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="A red bicycle leaning against a white wall, afternoon sunlight.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/transformers-reference.json")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--max-tokens", type=int, default=3584)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(ROOT / "models/original", local_files_only=True)
    images = []
    for path in args.image:
        with Image.open(path) as image:
            images.append(image.copy())
    prepared = prepare_request(args.prompt, images)
    inputs = processor.apply_chat_template(prepared.messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False)
    length = inputs["input_ids"].shape[1]
    started = time.monotonic()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(ROOT / "models/original", dtype=torch.bfloat16, device_map="cuda", local_files_only=True, attn_implementation="sdpa").eval()
    inputs = inputs.to(model.device)
    loaded = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False, logits_processor=LogitsProcessorList([GeneratedPresencePenalty(length, prepared.payload()["presence_penalty"])]))
    torch.cuda.synchronize()
    elapsed = time.monotonic() - loaded
    generated = result[0, length:]
    record = {
        "backend": "Transformers BF16, official generated-only presence penalty",
        "prompt": args.prompt, "input_tokens": length, "output_tokens": len(generated),
        "load_seconds": loaded - started, "generation_seconds": elapsed,
        "tokens_per_second": len(generated) / elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "caption": processor.decode(generated, skip_special_tokens=True).strip(),
        "token_ids": generated.tolist(),
        "finished": int(generated[-1]) in (248044, 248046),
    }
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items() if key not in ("caption", "token_ids")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
