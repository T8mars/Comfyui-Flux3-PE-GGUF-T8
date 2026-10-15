from __future__ import annotations

import base64
import copy
from collections import Counter
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

from . import bbox

SOURCE_MODEL = "black-forest-labs/flux-3-image-klein-pup"
SOURCE_REVISION = "8b069751eb5dcab23829d9c5eccb470eb8b8a4ed"
MAX_REFERENCES = 10
SYSTEM_PROMPTS = json.loads((Path(__file__).parent / "resources/system_prompts.json").read_text(encoding="utf-8"))
I2I_SCAFFOLD = {
    False: "Reference images follow in order (the first image, the second image, ...).\nUser request: {instruction}\nOutput the caption.",
    True: "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\nUser editing instruction: {instruction}\nDecide the edit, imagine the target image, and output the annotation record.",
}


class PEError(RuntimeError):
    def __init__(self, message, partial_text=None, kind="backend"):
        super().__init__(message)
        self.partial_text = partial_text
        self.kind = kind


def reference_size(width: int, height: int, n_refs: int) -> tuple[int, int]:
    if not 1 <= n_refs <= MAX_REFERENCES:
        raise ValueError("PUP supports 1 to 10 reference images")
    if min(width, height) <= 0 or max(width, height) / min(width, height) > 200:
        raise ValueError("Invalid reference size or aspect ratio above 200:1")
    cap = 1024**2 if n_refs == 1 else 724**2 if n_refs == 2 else 512**2
    floor = lambda value: max(32, int(math.floor(value / 32)) * 32)
    w, h = max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor(width / beta), floor(height / beta)
    if w * h > cap:
        if w == 32:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    # Rounding a thin image down to a 32-pixel short side can otherwise
    # create a ratio rejected by the official Qwen reference processor.
    if w > 200 * h:
        w = 200 * h
    elif h > 200 * w:
        h = 200 * w
    return w, h


def prepare_reference(image: Image.Image, n_refs: int) -> tuple[bytes, tuple[int, int]]:
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = reference_size(image.width, image.height, n_refs)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue(), size


def parse_boxes(value: str | list | None, n_images: int) -> list | None:
    rows = (json.loads(value) if value.strip() else None) if isinstance(value, str) else value
    if rows in (None, "", []):
        return None
    if not isinstance(rows, list):
        raise ValueError("boxes_json must be a JSON array")
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Box {index} must be an object")
        row_id = row.get("id") or f"obj_{index + 1}"
        if not isinstance(row_id, str) or not bbox.ID.fullmatch(row_id) or row_id in seen:
            raise ValueError(f"Invalid or duplicate box id: {row_id}")
        seen.add(row_id)
        keys = [key for key in ("bbox", "src_bbox", "tgt_bbox") if row.get(key) is not None]
        if not keys or ("bbox" in keys and len(keys) > 1):
            raise ValueError(f"Box {row_id} needs bbox OR src_bbox/tgt_bbox")
        for key in keys:
            box = row[key]
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError(f"{row_id}.{key} must be [y_min,x_min,y_max,x_max]")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
                raise ValueError(f"{row_id}.{key} contains a non-finite coordinate")
            y1, x1, y2, x2 = box
            if not (0 <= y1 < y2 <= 1 and 0 <= x1 < x2 <= 1):
                raise ValueError(f"{row_id}.{key} must be ordered normalized coordinates in 0..1")
        if "bbox" not in keys:
            origin = row.get("from", "")
            if origin == "target":
                if row.get("src_bbox") is not None:
                    raise ValueError(f"New target object {row_id} cannot have src_bbox")
            elif not isinstance(origin, str) or origin not in {f"ref_image_{index}" for index in range(n_images)}:
                raise ValueError(f"Box {row_id} refers to a missing reference image")
            elif row.get("src_bbox") is None:
                raise ValueError(f"Reference object {row_id} needs a non-null src_bbox")
    return rows


def validate_canvas_rows(rows: list | None, n_images: int):
    """Validate inline 0–1000 tables without changing their supplied coordinates."""
    if not rows:
        return
    normalized = []
    for row in rows:
        converted = dict(row)
        for key in ("bbox", "src_bbox", "tgt_bbox"):
            box = row.get(key)
            if isinstance(box, (list, tuple)) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box):
                converted[key] = [v / bbox.SCALE for v in box]
        normalized.append(converted)
    parse_boxes(normalized, n_images)


@dataclass
class PreparedRequest:
    mode: str
    messages: list
    locked: list | None
    reference_sizes: list[tuple[int, int]]
    fingerprint: str

    def payload(self, max_tokens: int = 3584) -> dict:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 3584:
            raise ValueError("max_tokens must be in 1..3584")
        return {
            "messages": self.messages,
            "temperature": 0.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": max_tokens,
            "presence_penalty": 0.0 if self.mode.startswith("i2i") else 1.5,
            "frequency_penalty": 0.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 8192,
            "penalties_generated_only": True,
            "samplers": ["penalties", "temperature"],
            "chat_template_kwargs": {"enable_thinking": False},
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "cache_prompt": False,
            "stream": False,
        }


def prepare_request(prompt: str, images: list[Image.Image] | None = None, boxes=None, canvas: tuple[int, int] | None = None) -> PreparedRequest:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The image request cannot be empty")
    images = images or []
    if len(images) > MAX_REFERENCES:
        raise ValueError("PUP accepts at most 10 reference images")
    if canvas is not None:
        if not isinstance(canvas, (list, tuple)) or len(canvas) != 2 or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in canvas):
            raise ValueError("Both canvas dimensions must be positive integers")
        width, height = canvas
        prompt = f"TARGET CANVAS: {width}:{height}\n{prompt}"
    rows = parse_boxes(boxes, len(images))
    mode, user_text, locked = bbox.plan(prompt, rows, len(images))
    validate_canvas_rows(locked, len(images))
    content, sizes = [], []
    for image in images:
        data, size = prepare_reference(image, len(images))
        sizes.append(size)
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}})
    if images:
        content.append({"type": "text", "text": I2I_SCAFFOLD[locked is not None].format(instruction=user_text)})
        user = content
    else:
        user = user_text
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}, {"role": "user", "content": user}]
    fingerprint = hashlib.sha256(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return PreparedRequest(mode, messages, locked, sizes, fingerprint)


def finish_caption(text: str, request: PreparedRequest, finish_reason: str) -> str:
    if finish_reason != "stop":
        raise PEError(f"PE generation did not finish normally ({finish_reason}); increase context/output budget or simplify the request", partial_text=text, kind="truncated" if finish_reason == "length" else "incomplete")
    caption = text.strip()
    if not caption or caption.startswith("```") or "<think>" in caption:
        raise PEError("PUP returned an empty, fenced, or thinking response", partial_text=text, kind="invalid_output")
    if request.locked:
        _, rows = bbox.split(caption)
        if rows is None:
            raise PEError("PUP layout output is missing its box table", partial_text=text, kind="layout")
        if any(isinstance(row, dict) and row.get("id") is not None and not isinstance(row["id"], str) for row in rows):
            raise PEError("PUP layout output contains a malformed box id", partial_text=text, kind="layout")
        declared_ids = {row["id"] for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)}
        caption = bbox.reconcile(caption, request.locked)
        prose, rows = bbox.split(caption)
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)) or any(f"<{row_id}>" not in prose for row_id in ids):
            raise PEError("PUP layout caption does not cite every box id", partial_text=text, kind="layout")
        if any(row_id not in ids and f"<{row_id}>" in prose for row_id in declared_ids):
            raise PEError("PUP layout caption cites a discarded invalid box", partial_text=text, kind="layout")
        # Quoted on-image wording may itself contain angle brackets. Only
        # unquoted annotation citations bind object or reference identities.
        annotation_prose = re.sub(r'"(?:\\.|[^"\\])*"', "", prose)
        cited_ids = set(re.findall(r"<([A-Za-z][A-Za-z0-9_]*)>", annotation_prose))
        reference_ids = {f"ref_image_{index}" for index in range(len(request.reference_sizes))}
        if set(ids) - cited_ids:
            raise PEError("PUP layout caption does not cite every box id", partial_text=text, kind="layout")
        if cited_ids - set(ids) - reference_ids:
            raise PEError("PUP layout caption cites an undeclared object or missing reference image", partial_text=text, kind="layout")
        for row in rows:
            origin = row.get("from")
            if origin is not None and origin != "target" and (not isinstance(origin, str) or origin not in reference_ids):
                raise PEError("PUP layout output refers to a missing reference image", partial_text=text, kind="layout")
    return caption


def has_repetition(text: str | None) -> bool:
    words = (text or "").lower().split()
    if len(words) < 80:
        return False
    phrases = Counter(tuple(words[index:index + 8]) for index in range(len(words) - 7))
    return max(phrases.values(), default=0) >= 5


def recovery_payload(request: PreparedRequest, max_tokens: int) -> dict:
    payload = copy.deepcopy(request.payload(max_tokens))
    payload["presence_penalty"] = max(payload["presence_penalty"], 1.0)
    payload["repeat_penalty"] = 1.05
    if request.locked:
        identities = ", ".join(f"<{row['id']}>" for row in request.locked)
        payload["messages"][0]["content"] += (
            "\n\nRequired annotation constraints for this retry: the user's locked identities override the default "
            "class-noun-plus-index naming convention. Use each locked ID verbatim in both the paragraph and its "
            "table row; a transferred reference object retains that ID and source, with both its source and target "
            "boxes in the same row. Do not create a second target identity for it. Preserve all supplied coordinates "
            "and nulls. Only annotate visible objects or text, and make every non-null box have positive width and "
            "height. Required locked rows: " + json.dumps(request.locked, ensure_ascii=False)
        )
        cue = f"\nOutput only the final caption and box table. Cite each locked identity verbatim in the caption: {identities}. Preserve its supplied coordinates and provenance; do not replace or rename locked identities."
        bindings = [{key: row[key] for key in ("id", "from", "desc") if key in row} for row in request.locked]
        cue += "\nRequired object bindings: " + json.dumps(bindings, ensure_ascii=False)
        cue += "\nUse these same IDs in the caption text for the described objects, not only in the JSON table. An object copied from a reference keeps its required ID and original source; do not create a separate target-origin alias for that object. Before answering, check that each required <id> occurs in the caption text."
        content = payload["messages"][-1]["content"]
        if isinstance(content, str):
            payload["messages"][-1]["content"] += cue
        else:
            content[-1]["text"] += cue
    return payload
