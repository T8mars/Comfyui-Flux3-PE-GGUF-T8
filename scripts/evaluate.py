"""Run real PE tasks and record latency, memory, outputs and structural failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flux3_pe.engine import ModelSpec, enhance, release_model


def fixtures():
    a = Image.new("RGB", (640, 480), "white")
    draw = ImageDraw.Draw(a)
    draw.rectangle((80, 100, 260, 330), fill="red")
    draw.ellipse((350, 130, 550, 330), fill="blue")
    draw.text((90, 350), "RED BOX", fill="black")
    draw.text((370, 350), "BLUE BALL", fill="black")
    b = Image.new("RGB", (576, 320), "skyblue")
    draw = ImageDraw.Draw(b)
    draw.rectangle((0, 200, 575, 319), fill="green")
    draw.rectangle((220, 110, 360, 230), fill="yellow")
    draw.polygon([(200, 110), (290, 40), (380, 110)], fill="brown")
    return {"a": a, "b": b}


def cases():
    t2i = [
        'A red bicycle leaning against a white wall, afternoon sunlight.',
        '三个蓝色气球，左边一只红色气球，纯白背景。',
        'A cafe sign reading "OPEN 24 HOURS", warm evening light.',
        'A minimalist poster with the exact Chinese text "贞贞的AI世界".',
        'Two sisters, one in a green dress and one in a purple coat, standing side by side.',
        'An overhead photograph of a ceramic bowl of strawberries on a wooden table.',
        'An abstract watercolor about quiet anticipation, no people.',
        'A tiny orange robot under a giant fern, low-angle macro photograph.',
        'A wide mountain landscape, a small lighthouse at the far right.',
        'A black-and-white pen drawing of exactly four teacups, no extra objects.',
    ]
    edits = [
        ('Change the red rectangle to orange; keep everything else unchanged.', ["a"]),
        ('Remove the blue circle but retain the red rectangle and both labels.', ["a"]),
        ('Replace the text RED BOX with GREEN BOX, preserving its placement.', ["a"]),
        ('Turn the whole image into a watercolor illustration.', ["b"]),
        ('Move the blue circle to the left of the red rectangle.', ["a"]),
        ('Place the blue ball from the first image on the lawn in the second image.', ["a", "b"]),
        ('Use the house from the first image and the colors of the second image.', ["b", "a"]),
        ('把黄色房子的屋顶改成红色，保留其他元素。', ["b"]),
        ('Horizontally flip the image while retaining the original content.', ["b"]),
        ('Create a contact sheet faithfully showing the supplied references in their attachment order.', ["a", "b"] * 5),
    ]
    result = []
    for index, prompt in enumerate(t2i):
        result.append({"id": f"t2i_{index+1:02}", "prompt": prompt, "images": [], "boxes": None})
    for index, (prompt, images) in enumerate(edits):
        result.append({"id": f"i2i_{index+1:02}", "prompt": prompt, "images": images, "boxes": None})
    for index in range(10):
        row = {"id": "obj_1", "desc": "a red balloon", "bbox": [0.1, 0.1, 0.6, 0.4]}
        rows = [row]
        prompt = 'A red balloon on the left, a blue teacup on the right, white background.'
        if index % 2:
            rows.append({"id": "obj_2", "desc": "a blue teacup", "bbox": [0.5, 0.6, 0.9, 0.95]})
        if index == 2:
            prompt = '左边一个红气球，右边一只蓝色茶杯。'
        if index == 3:
            prompt = 'A red balloon with the exact text "SALE 50%", a blue teacup on the right.'
        if index >= 4:
            row = {**row, "bbox": [0.05 + index * 0.01, 0.05, 0.65, 0.45]}
            rows[0] = row
        result.append({"id": f"t2i_bbox_{index+1:02}", "prompt": prompt, "images": [], "boxes": rows})
    for index in range(10):
        rows = [{"id": "red_box", "desc": "the red rectangle", "from": "ref_image_0", "src_bbox": [0.208, 0.125, 0.69, 0.407], "tgt_bbox": [0.208, 0.15, 0.69, 0.432]}]
        prompt, images = 'Move the red rectangle slightly to the right, keeping all other elements unchanged.', ["a"]
        if index % 3 == 1:
            rows[0]["tgt_bbox"] = None
            prompt = 'Remove the red rectangle, keeping the blue circle and labels.'
        if index % 3 == 2:
            rows = [{"id": "green_ball", "desc": "a new green ball", "from": "target", "src_bbox": None, "tgt_bbox": [0.05, 0.4, 0.3, 0.6]}]
            prompt = 'Add a green ball above the two existing shapes without moving them.'
        if index == 6:
            images = ["a", "b"]
            prompt = 'Place the red rectangle from the first image on the lawn of the second image.'
        if index == 7:
            prompt = '移除红色矩形，保留蓝色圆形与文字。'
        result.append({"id": f"i2i_bbox_{index+1:02}", "prompt": prompt, "images": images, "boxes": rows})
    return result


def evaluation_input_identity():
    """Bind the corpus, actual reference pixels, and its execution script."""
    payload = {"cases": cases(), "fixtures": {name: {"mode": image.mode, "size": image.size, "sha256": hashlib.sha256(image.tobytes()).hexdigest()} for name, image in fixtures().items()}, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["Q4_K_M"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--gpu-layers", default="all")
    parser.add_argument("--strict", action="store_true", help="Disable bounded recovery to measure the original decoding policy")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluation.json")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 1 <= args.max_tokens <= 3584:
        parser.error("--max-tokens must be in 1..3584")
    samples = cases()
    if args.ids:
        unknown = sorted(set(args.ids) - {case["id"] for case in samples})
        if unknown:
            parser.error("Unknown case IDs: " + ", ".join(unknown))
        samples = [case for case in samples if case["id"] in args.ids]
    if args.limit is not None:
        samples = samples[:args.limit]
    if not samples:
        parser.error("No evaluation tasks were selected")
    images = fixtures()
    fixture_dir = ROOT / "artifacts/fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for name, image in images.items():
        image.save(fixture_dir / f"{name}.png")
    records = []
    report = {"source_revision": "8b069751eb5dcab23829d9c5eccb470eb8b8a4ed", "input_identity": evaluation_input_identity(), "recovery_enabled": not args.strict, "results": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        spec = ModelSpec.load(ROOT / f"models/gguf/flux3-klein-pup-{name}.gguf", gpu_layers=args.gpu_layers)
        for case in samples:
            memory, stopped = [], threading.Event()

            def monitor():
                while not stopped.wait(0.5):
                    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, creationflags=flags)
                    if result.returncode == 0:
                        memory.append([int(value.strip()) for value in result.stdout.strip().split(",")])

            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            started = time.monotonic()
            record = {"model": name, "case": case["id"], "request": case}
            try:
                caption, metadata = enhance(spec, case["prompt"], [images[key] for key in case["images"]], case["boxes"], max_tokens=args.max_tokens, keep_loaded=True, use_cache=False, recover_invalid=not args.strict)
                required_text = re.findall(r'"([^"\n]+)"', case["prompt"])
                literal_text_ok = all(text in caption for text in required_text)
                record.update({"passed": literal_text_ok, "caption": caption, "metadata": metadata, "checks": {"structure": True, "required_text": required_text, "literal_text_preserved": literal_text_ok}})
            except Exception as error:
                record.update({"passed": False, "error": f"{type(error).__name__}: {error}"})
                if getattr(error, "partial_text", None):
                    record["partial_caption"] = error.partial_text
            finally:
                stopped.set()
                thread.join(timeout=2)
            record["wall_seconds"] = round(time.monotonic() - started, 4)
            record["gpu_peak_mib_total"] = max((entry[0] for entry in memory), default=None)
            records.append(record)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{name} {case['id']} passed={record['passed']} seconds={record['wall_seconds']}", flush=True)
        release_model()
    report["summary"] = {}
    for name in args.models:
        entries = [entry for entry in records if entry["model"] == name]
        passed = [entry for entry in entries if entry["passed"]]
        durations = [entry["metadata"]["generation_seconds"] for entry in passed]
        report["summary"][name] = {"tasks": len(entries), "passed": len(passed), "median_generation_seconds": statistics.median(durations) if durations else None, "model_bytes": (ROOT / f"models/gguf/flux3-klein-pup-{name}.gguf").stat().st_size}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    if any(not record["passed"] for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        release_model()
