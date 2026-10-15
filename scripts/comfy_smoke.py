"""Execute real PE and optional image-generation workflows through ComfyUI."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import uuid

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import fixtures


def wait_for_history(session, url, prompt_id, name, started=None):
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            history = session.get(url + f"/history/{prompt_id}", timeout=30).json()
        except (requests.Timeout, requests.ConnectionError):
            continue
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry["status"]["status_str"]
            timestamps = [message[1]["timestamp"] for message in entry["status"].get("messages", []) if "timestamp" in message[1]]
            seconds = time.monotonic() - started if started is not None else ((max(timestamps) - min(timestamps)) / 1000 if timestamps else None)
            record = {"workflow": name, "prompt_id": prompt_id, "seconds": seconds, "status": status, "history": entry}
            if status != "success":
                raise RuntimeError(f"Comfy execution failed: {entry['status']}")
            if not entry["outputs"].get("3", {}).get("text"):
                raise RuntimeError("PE output was not previewed")
            return record
        time.sleep(0.5)
    raise TimeoutError(f"Comfy did not complete {prompt_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8267")
    parser.add_argument("--generate-image", action="store_true")
    parser.add_argument("--only", choices=["text", "layout", "two_references", "pe_to_image"])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/comfy-smoke.json")
    parser.add_argument("--resume-prompt", help="Observe an existing job without submitting it again")
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False
    if args.resume_prompt:
        records = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else []
        record = wait_for_history(session, args.url, args.resume_prompt, args.only or "pe_to_image")
        records = [item for item in records if item["prompt_id"] != args.resume_prompt] + [record]
        args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"PASS resumed {record['workflow']} {record['seconds']}s", flush=True)
        return
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            if session.get(args.url + "/system_stats", timeout=1).ok:
                break
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(0.5)
    else:
        raise TimeoutError("ComfyUI did not become ready")
    text = json.loads((ROOT / "examples/01_prompt_enhancement.api.json").read_text(encoding="utf-8"))
    layout = json.loads((ROOT / "examples/02_layout_enhancement.api.json").read_text(encoding="utf-8"))
    fixture_dir = ROOT / "artifacts/fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for name, image in fixtures().items():
        path = fixture_dir / f"{name}.png"
        image.save(path)
        with path.open("rb") as stream:
            response = session.post(args.url + "/upload/image", files={"image": (path.name, stream, "image/png")}, data={"overwrite": "true"}, timeout=20)
        response.raise_for_status()
        names.append(response.json()["name"])
    editing = copy.deepcopy(text)
    editing["2"]["inputs"]["prompt"] = "Place the blue circle from the first image on the lawn of the second image."
    for index, name in enumerate(names):
        node_id = str(10 + index)
        editing[node_id] = {"class_type": "LoadImage", "inputs": {"image": name}}
        editing["2"]["inputs"][f"images.image_{index+1}"] = [node_id, 0]
    workflows = [("text", text), ("layout", layout), ("two_references", editing)]
    if args.generate_image:
        generation = copy.deepcopy(text)
        generation["2"]["inputs"]["use_cache"] = False
        generation.update({
            "10": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux3_image_klein_sd_int8_convrot.safetensors", "weight_dtype": "default"}},
            "11": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_4b_int8_convrot.safetensors", "type": "flux3", "device": "default"}},
            "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["11", 0], "text": ["2", 0]}},
            "13": {"class_type": "FluxDisableGuidance", "inputs": {"conditioning": ["12", 0]}},
            "14": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["13", 0]}},
            "15": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "16": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "positive": ["13", 0], "negative": ["14", 0], "latent_image": ["15", 0], "seed": 7, "steps": 16, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "17": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
            "18": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["17", 0]}},
            "19": {"class_type": "SaveImage", "inputs": {"images": ["18", 0], "filename_prefix": "FLUX3_PE_Q4_smoke"}},
        })
        (ROOT / "examples/03_pe_to_flux3.api.json").write_text(json.dumps(generation, ensure_ascii=False, indent=2), encoding="utf-8")
        workflows.append(("pe_to_image", generation))
    records = []
    for name, graph in workflows:
        if args.only and name != args.only:
            continue
        started = time.monotonic()
        response = session.post(args.url + "/prompt", json={"prompt": graph, "client_id": str(uuid.uuid4())}, timeout=30)
        if not response.ok:
            raise RuntimeError(f"Comfy validation failed: {response.text}")
        prompt_id = response.json()["prompt_id"]
        print(f"Queued {name}: {prompt_id}", flush=True)
        record = wait_for_history(session, args.url, prompt_id, name, started)
        records.append(record)
        args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"PASS {name} {record['seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
