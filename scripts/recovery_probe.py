import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import fixtures
from flux3_pe.engine import LlamaEngine, ModelSpec
from flux3_pe.protocol import prepare_request


def main():
    engine = LlamaEngine(ModelSpec.load(ROOT / "models/gguf/flux3-klein-pup-Q4_K_M.gguf"))
    request = prepare_request("把黄色房子的屋顶改成红色，保留其他元素。", [fixtures()["b"]])
    records = []
    try:
        engine.start()
        for presence in (0.5, 1.0):
            payload = request.payload(1024)
            payload.update({"presence_penalty": presence, "repeat_penalty": 1.05})
            response = engine.complete(payload)
            choice = response["choices"][0]
            text = choice["message"].get("content") or ""
            records.append({"presence_penalty": presence, "finish_reason": choice["finish_reason"], "text": text, "usage": response.get("usage", {})})
            print(presence, choice["finish_reason"], response.get("usage", {}), flush=True)
    finally:
        engine.close()
    (ROOT / "artifacts/recovery-probe.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
