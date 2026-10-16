"""In-process PUP inference using an installed llama-cpp-python wheel."""
from __future__ import annotations

import base64
import ctypes
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import time

from .protocol import PEError

MIN_WHEEL_VERSION = (0, 3, 36)


def wheel_identity() -> str:
    """Inspect the installed distribution without importing its native libraries."""
    try:
        package = metadata.distribution("llama-cpp-python")
    except metadata.PackageNotFoundError as error:
        raise PEError("Install llama-cpp-python>=0.3.36 into ComfyUI's Python; see README wheel downloads") from error
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[.+-].*)?", package.version)
    if match is None or tuple(map(int, match.groups())) < MIN_WHEEL_VERSION:
        raise PEError(f"llama-cpp-python {package.version} is too old; install >=0.3.36 with MTMD and Qwen3.5 support")
    root = Path(package.locate_file("llama_cpp")).resolve()
    bound = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and (path.suffix in (".py", ".dll", ".so", ".dylib") or ".so." in path.name):
            stat = path.stat()
            bound.append((str(path), stat.st_size, stat.st_mtime_ns))
    if not bound:
        raise PEError("The llama-cpp-python installation is incomplete; reinstall a compatible wheel")
    return hashlib.sha256(json.dumps([package.version, bound, os.environ.get("LLAMA_CPP_LIB_PATH")], sort_keys=True).encode()).hexdigest()


def render_prompt(template: str, messages: list, marker: str) -> tuple[str, list[bytes]]:
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    images = []
    converted = []
    for message in messages:
        message = dict(message)
        if isinstance(message["content"], list):
            parts = []
            for part in message["content"]:
                if part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:image/") or ";base64," not in url:
                        raise PEError("The wheel backend accepts only locally prepared reference images")
                    images.append(base64.b64decode(url.split(",", 1)[1], validate=True))
                    # MTMD inserts the model's vision boundary tokens itself.
                    parts.append({"type": "text", "text": marker})
                else:
                    parts.append(dict(part))
            message["content"] = parts
        converted.append(message)
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def fail(message):
        raise PEError(str(message))

    prompt = environment.from_string(template).render(messages=converted, add_generation_prompt=True, enable_thinking=False, add_vision_id=False, tools=None, bos_token="", eos_token="<|im_end|>", raise_exception=fail)
    return prompt, images


def generated_sampler(llm, payload):
    from llama_cpp._internals import LlamaSampler

    # A new native chain has no prompt history. llama_sampler_sample accepts
    # only each sampled output token, matching the server's PE sampling patch.
    sampler = LlamaSampler()
    try:
        sampler.add_penalties(n_vocab=llm.n_vocab(), penalty_last_n=payload["repeat_last_n"], penalty_repeat=payload["repeat_penalty"], penalty_freq=payload["frequency_penalty"], penalty_present=payload["presence_penalty"])
        sampler.add_greedy()
        return sampler
    except BaseException:
        sampler.close()
        raise


def token_piece(api, vocab, token) -> bytes:
    # The wheel's token_to_piece helper returns its entire padded buffer.
    # Use the C API's actual byte count, including tokens longer than 32 bytes.
    buffer = ctypes.create_string_buffer(64)
    count = api.llama_token_to_piece(vocab, token, buffer, len(buffer), 0, False)
    if count < 0:
        buffer = ctypes.create_string_buffer(-count)
        count = api.llama_token_to_piece(vocab, token, buffer, len(buffer), 0, False)
    if count < 0 or count > len(buffer):
        raise PEError("Failed to decode a generated token")
    return buffer.raw[:count]


class WheelEngine:
    process = None

    def __init__(self, spec):
        self.spec = spec
        self.llm = None
        self.handler = None
        self._cleanup_failed = False
        self._interrupt = lambda: None
        self._deadline = float("inf")
        self._abort_error = None
        self._abort_callback = None
        self.version = ""

    @property
    def is_loaded(self):
        return self.llm is not None

    def _check(self):
        if self._abort_error is not None:
            raise self._abort_error
        self._interrupt()
        if time.monotonic() > self._deadline:
            raise PEError("PE request exceeded 600 seconds")

    def start(self, interrupt=lambda: None):
        self._interrupt = interrupt
        self._check()
        if self.llm is not None:
            return
        wheel_identity()
        try:
            import llama_cpp
            from llama_cpp import mtmd_cpp
            from llama_cpp._logger import llama_log_callback
            from llama_cpp.llama_chat_format import MTMDChatHandler
        except (ImportError, OSError, RuntimeError, AttributeError) as error:
            raise PEError(f"Cannot load llama-cpp-python: {error}. Install a compatible wheel and its CUDA/OS runtime dependencies; see README") from error
        self.api, self.mtmd = llama_cpp, mtmd_cpp
        self.version = llama_cpp.__version__
        installed = metadata.version("llama-cpp-python")
        if self.version != installed:
            raise PEError("Another llama-cpp-python version is already imported; restart ComfyUI after updating its wheel")
        mtmd_cpp.mtmd_log_set(llama_log_callback, None)
        mtmd_cpp.mtmd_helper_log_set(llama_log_callback, None)
        try:
            gpu_layers = -1 if self.spec.gpu_layers == "all" else int(self.spec.gpu_layers)
            self.llm = llama_cpp.Llama(model_path=str(self.spec.model), n_ctx=self.spec.context, n_gpu_layers=gpu_layers, n_batch=512, n_ubatch=512, n_threads=min(8, os.cpu_count() or 1), n_threads_batch=min(8, os.cpu_count() or 1), flash_attn=True, verbose=False)

            @llama_cpp.ggml_abort_callback
            def abort(_):
                try:
                    self._check()
                    return False
                except BaseException as error:
                    self._abort_error = error
                    return True

            self._abort_callback = abort
            llama_cpp.llama_set_abort_callback(self.llm._ctx.ctx, abort, None)
            self.handler = MTMDChatHandler(str(self.spec.mmproj), verbose=False, use_gpu=gpu_layers != 0)
            self.handler._init_mtmd_context(self.llm)
            self._check()
        except BaseException:
            self.close()
            raise

    def complete(self, payload, interrupt=lambda: None):
        self._interrupt = interrupt
        self._deadline = time.monotonic() + 600
        self._abort_error = None
        self._check()
        if self.llm is None:
            raise PEError("The Python wheel backend is not loaded")
        if payload["temperature"] != 0.0:
            raise PEError("The PE wheel backend requires greedy decoding")
        llm, mtmd, api = self.llm, self.mtmd, self.api
        prompt, images = render_prompt(llm.metadata["tokenizer.chat_template"], payload["messages"], mtmd.mtmd_default_marker().decode())
        bitmaps, chunks, sampler = [], None, None
        started = time.monotonic()
        try:
            llm.reset()
            llm._ctx.kv_cache_clear()
            for data in images:
                self._check()
                bitmap = self.handler._create_bitmap_from_bytes(data)
                bitmaps.append(bitmap)
            text_bytes = prompt.encode("utf-8")
            text = mtmd.mtmd_input_text()
            text.text, text.text_len, text.add_special, text.parse_special = text_bytes, len(text_bytes), True, True
            chunks = mtmd.mtmd_input_chunks_init()
            if not chunks:
                raise PEError("Cannot allocate MTMD prompt chunks")
            bitmap_array = (mtmd.mtmd_bitmap_p_ctypes * len(bitmaps))(*bitmaps)
            result = mtmd.mtmd_tokenize(self.handler.mtmd_ctx, chunks, ctypes.byref(text), bitmap_array, len(bitmaps))
            self._check()
            if result:
                raise PEError(f"MTMD prompt tokenization failed: {result}")
            prompt_tokens = mtmd.mtmd_helper_get_n_tokens(chunks)
            if prompt_tokens + payload["max_tokens"] > self.spec.context:
                raise PEError(f"Prompt and output budget exceed context ({prompt_tokens} + {payload['max_tokens']} > {self.spec.context}); increase context_size or reduce references/max_tokens")
            position = api.llama_pos(0)
            count = mtmd.mtmd_input_chunks_size(chunks)
            for index in range(count):
                self._check()
                chunk = mtmd.mtmd_input_chunks_get(chunks, index)
                result = mtmd.mtmd_helper_eval_chunk_single(self.handler.mtmd_ctx, llm._ctx.ctx, chunk, position.value, 0, llm.n_batch, index == count - 1, ctypes.byref(position))
                self._check()
                if result:
                    raise PEError(f"MTMD prompt evaluation failed: {result}")
            prefetched = time.monotonic()
            sampler = generated_sampler(llm, payload)
            tokens, output_bytes, output, finish_reason = [], bytearray(), "", "length"
            for _ in range(payload["max_tokens"]):
                self._check()
                token = sampler.sample(llm._ctx, -1)
                if api.llama_vocab_is_eog(llm._model.vocab, token):
                    finish_reason = "stop"
                    break
                tokens.append(token)
                output_bytes.extend(token_piece(api, llm._model.vocab, token))
                output = output_bytes.decode("utf-8", errors="replace")
                stops = [output.find(stop) for stop in payload["stop"] if stop in output]
                if stops:
                    output, finish_reason = output[:min(stops)], "stop"
                    break
                if len(tokens) < payload["max_tokens"]:
                    llm._batch.set_batch([token], n_past=position.value, logits_all=False)
                    try:
                        llm._ctx.decode(llm._batch)
                    except RuntimeError:
                        self._check()
                        raise
                    position.value += 1
            self._check()
            elapsed = time.monotonic() - prefetched
            return {"choices": [{"message": {"content": output}, "finish_reason": finish_reason}], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(tokens), "total_tokens": prompt_tokens + len(tokens)}, "timings": {"prompt_ms": round((prefetched - started) * 1000, 3), "predicted_ms": round(elapsed * 1000, 3), "predicted_per_second": round(len(tokens) / elapsed, 3) if elapsed else 0}}
        finally:
            if sampler is not None:
                sampler.close()
            if chunks:
                mtmd.mtmd_input_chunks_free(chunks)
            for bitmap in bitmaps:
                mtmd.mtmd_bitmap_free(bitmap)
            self._deadline = float("inf")

    def close(self):
        if self.llm is not None:
            try:
                self.llm.close()
            except BaseException:
                self._cleanup_failed = True
                raise
        self.llm = self.handler = self._abort_callback = None
        self._cleanup_failed = False
        self._abort_error = None
        self._deadline = float("inf")
