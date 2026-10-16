"""Backend selection, multimodal ordering, sampling and native resource ownership."""
import base64
import ctypes
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from flux3_pe import engine, wheel_backend as wheel
from flux3_pe.protocol import PEError, prepare_request
from test_model_spec import model_files


def test_python_selection_does_not_read_any_server_manifest(model_files, monkeypatch):
    monkeypatch.setattr(wheel, "wheel_identity", lambda: "installed-wheel")
    model_files["server"].unlink()
    model_files["runtime_manifest"].unlink()
    spec = engine.ModelSpec.load(model_files["model"], backend="llama-cpp-python")
    assert spec.backend == "llama-cpp-python" and spec.server is None
    assert spec.mmproj == model_files["projector"]


def test_auto_prefers_compatible_wheel_and_explicit_server_still_works(model_files, monkeypatch):
    monkeypatch.delenv("FLUX3_PE_LLAMA_SERVER", raising=False)
    monkeypatch.setattr(wheel, "wheel_identity", lambda: "installed-wheel")
    assert engine.ModelSpec.load(model_files["model"]).backend == "llama-cpp-python"
    server = engine.ModelSpec.load(model_files["model"], server=model_files["server"])
    assert server.backend == "llama-server"


def test_explicit_wheel_does_not_silently_fall_back_to_server(model_files, monkeypatch):
    def unavailable():
        raise PEError("wheel is too old")
    monkeypatch.setattr(wheel, "wheel_identity", unavailable)
    with pytest.raises(PEError, match="too old"):
        engine.ModelSpec.load(model_files["model"], backend="llama-cpp-python")
    with pytest.raises(ValueError, match="server cannot"):
        engine.ModelSpec.load(model_files["model"], server=model_files["server"], backend="llama-cpp-python")


def test_auto_falls_back_only_when_packaged_runtime_exists(model_files, monkeypatch):
    root = model_files["server"].parent.parent
    monkeypatch.setattr(engine, "ROOT", root)
    monkeypatch.setenv("FLUX3_PE_LLAMA_SERVER", str(model_files["server"]))
    assert engine.ModelSpec.load(model_files["model"]).backend == "llama-server"


def test_wheel_revision_changes_when_wheel_changes(model_files, monkeypatch):
    monkeypatch.setattr(wheel, "wheel_identity", lambda: "wheel-one")
    first = engine.ModelSpec.load(model_files["model"], backend="llama-cpp-python")
    monkeypatch.setattr(wheel, "wheel_identity", lambda: "wheel-two")
    second = engine.ModelSpec.load(model_files["model"], backend="llama-cpp-python")
    assert first.revision != second.revision


@pytest.mark.parametrize("version", ["0.2.89", "0.3.35", "invalid"])
def test_old_wheels_fail_before_native_import(monkeypatch, version):
    monkeypatch.setattr(wheel.metadata, "distribution", lambda _: SimpleNamespace(version=version))
    with pytest.raises(PEError, match="too old"):
        wheel.wheel_identity()


def test_wheel_identity_binds_package_location_and_native_file_changes(tmp_path, monkeypatch):
    package = tmp_path / "llama_cpp"
    package.mkdir()
    (package / "__init__.py").write_text("fixture")
    library = package / "libllama.so"
    library.write_bytes(b"native")
    monkeypatch.setattr(wheel.metadata, "distribution", lambda _: SimpleNamespace(version="0.3.36", locate_file=lambda name: tmp_path / name))
    first = wheel.wheel_identity()
    library.write_bytes(b"new native code")
    assert wheel.wheel_identity() != first


def test_template_keeps_image_order_avoids_duplicate_vision_tokens_and_disables_thinking():
    template = "{% for m in messages %}{% for p in m.content %}{{ p.text }}{% endfor %}{% endfor %}|{{ enable_thinking }}"
    data = lambda value: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(value).decode()}}
    messages = [{"role": "user", "content": [data(b"first"), {"type": "text", "text": "middle"}, data(b"second")]}]
    before = json.dumps(messages)
    text, images = wheel.render_prompt(template, messages, "<media>")
    assert text == "<media>middle<media>|False"
    assert images == [b"first", b"second"]
    assert json.dumps(messages) == before


def test_wheel_never_fetches_external_reference_urls():
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.invalid/image"}}]}]
    with pytest.raises(PEError, match="locally prepared"):
        wheel.render_prompt("unused", messages, "<media>")


def test_generated_sampler_has_no_prompt_history_and_closes_failed_initialization(monkeypatch):
    calls = []
    class Sampler:
        def __init__(self): calls.append("new")
        def add_penalties(self, **kwargs): calls.append(kwargs)
        def add_greedy(self): calls.append("greedy")
        def close(self): calls.append("close")
    monkeypatch.setitem(sys.modules, "llama_cpp._internals", SimpleNamespace(LlamaSampler=Sampler))
    payload = prepare_request("A red balloon.").payload()
    sampler = wheel.generated_sampler(SimpleNamespace(n_vocab=lambda: 100), payload)
    assert calls == ["new", {"n_vocab": 100, "penalty_last_n": 8192, "penalty_repeat": 1.0, "penalty_freq": 0.0, "penalty_present": 1.5}, "greedy"]
    sampler.close()
    monkeypatch.setattr(Sampler, "add_greedy", lambda self: (_ for _ in ()).throw(ValueError("broken sampler")))
    with pytest.raises(ValueError):
        wheel.generated_sampler(SimpleNamespace(n_vocab=lambda: 100), payload)
    assert calls[-1] == "close"


@pytest.mark.parametrize("piece", [b"short", b"long" * 40, "提示词".encode("utf-8"), b"\xe8\xaf"])
def test_token_decoder_preserves_exact_bytes_without_buffer_padding(piece):
    def decode(vocab, token, buffer, size, lstrip, special):
        if len(piece) > size:
            return -len(piece)
        ctypes.memmove(buffer, piece, len(piece))
        return len(piece)
    assert wheel.token_piece(SimpleNamespace(llama_token_to_piece=decode), "vocab", 10) == piece


@pytest.fixture
def native_engine(monkeypatch):
    effects = []
    spec = engine.ModelSpec(Path("model"), Path("vision"), None, "wheel", backend="llama-cpp-python")
    backend = wheel.WheelEngine(spec)
    positions = []
    pieces = {10: b"A ", 11: b"red ball.", 0: b""}
    llm = SimpleNamespace(metadata={"tokenizer.chat_template": "prompt"}, reset=lambda: effects.append("reset"), n_batch=512,
        _ctx=SimpleNamespace(ctx="ctx", kv_cache_clear=lambda: effects.append("clear"), decode=lambda batch: effects.append("decode")),
        _model=SimpleNamespace(vocab="vocab", token_to_piece=lambda token: pieces[token]),
        _batch=SimpleNamespace(set_batch=lambda tokens, **kwargs: positions.append(kwargs["n_past"])), close=lambda: effects.append("close_model"))
    def evaluate(ctx, llama_ctx, chunk, position, seq, batch, last, output):
        effects.append(("chunk", chunk, position, last))
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int32)).contents.value = position + (3 if chunk == 0 else 7)
        return 0
    backend.llm = llm
    backend.handler = SimpleNamespace(mtmd_ctx="vision", _create_bitmap_from_bytes=lambda data: effects.append(("bitmap", data)) or 1)
    def decode_piece(vocab, token, buffer, size, lstrip, special):
        ctypes.memmove(buffer, pieces[token], len(pieces[token]))
        return len(pieces[token])
    backend.api = SimpleNamespace(llama_pos=ctypes.c_int32, llama_vocab_is_eog=lambda vocab, token: token == 0, llama_token_to_piece=decode_piece)
    backend.mtmd = SimpleNamespace(mtmd_default_marker=lambda: b"<media>", mtmd_input_text=lambda: SimpleNamespace(),
        mtmd_input_chunks_init=lambda: 1, mtmd_bitmap_p_ctypes=ctypes.c_void_p,
        mtmd_tokenize=lambda *args: 0, mtmd_helper_get_n_tokens=lambda _: 10,
        mtmd_input_chunks_size=lambda _: 2, mtmd_input_chunks_get=lambda _, index: index,
        mtmd_helper_eval_chunk_single=evaluate, mtmd_input_chunks_free=lambda _: effects.append("free_chunks"), mtmd_bitmap_free=lambda _: effects.append("free_bitmap"))
    # The text structure is a real ctypes value as required by byref().
    class Text(ctypes.Structure):
        _fields_ = [("text", ctypes.c_char_p), ("text_len", ctypes.c_size_t), ("add_special", ctypes.c_bool), ("parse_special", ctypes.c_bool)]
    backend.mtmd.mtmd_input_text = Text
    tokens = iter([10, 11, 0])
    monkeypatch.setattr(wheel, "generated_sampler", lambda *args: SimpleNamespace(sample=lambda *args: next(tokens), close=lambda: effects.append("free_sampler")))
    return backend, effects, positions


def test_native_positions_usage_and_all_temporary_resources(native_engine):
    backend, effects, positions = native_engine
    response = backend.complete(prepare_request("A ball").payload(16))
    assert response["choices"][0] == {"message": {"content": "A red ball."}, "finish_reason": "stop"}
    assert ("chunk", 0, 0, False) in effects and ("chunk", 1, 3, True) in effects
    assert positions == [10, 11]
    assert response["usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
    assert effects[-2:] == ["free_sampler", "free_chunks"]


def test_over_context_budget_fails_before_evaluation_and_frees_chunks(native_engine):
    backend, effects, _ = native_engine
    backend.mtmd.mtmd_helper_get_n_tokens = lambda _: 8190
    with pytest.raises(PEError, match="exceed context"):
        backend.complete(prepare_request("A ball").payload(16))
    assert not any(isinstance(item, tuple) and item[0] == "chunk" for item in effects)
    assert effects[-1] == "free_chunks"


def test_cancel_during_prefill_frees_native_temporary_resources(native_engine):
    backend, effects, _ = native_engine
    def interrupt():
        if any(isinstance(item, tuple) and item[0] == "chunk" for item in effects):
            raise InterruptedError("cancelled prefill")
    with pytest.raises(InterruptedError):
        backend.complete(prepare_request("A ball").payload(16), interrupt)
    assert effects[-1] == "free_chunks"


def test_failed_native_close_preserves_handle_for_retry(native_engine):
    backend, effects, _ = native_engine
    llm = backend.llm
    llm.close = lambda: (_ for _ in ()).throw(RuntimeError("native close failed"))
    with pytest.raises(RuntimeError): backend.close()
    assert backend.llm is llm and backend._cleanup_failed
    llm.close = lambda: effects.append("retry_close")
    backend.close()
    assert backend.llm is None and not backend._cleanup_failed


def test_cached_wheel_model_avoids_repeated_comfy_unload_and_releases(monkeypatch):
    engine.release_model()
    engine._results.clear()
    effects = []
    class FakeWheel:
        process = None
        version = "0.3.36"
        def __init__(self, spec): self.spec, self.is_loaded = spec, False
        def start(self, interrupt): self.is_loaded = True
        def complete(self, payload, interrupt): return {"choices": [{"message": {"content": "A ball."}, "finish_reason": "stop"}]}
        def close(self): effects.append("close"); self.is_loaded = False
    monkeypatch.setattr(wheel, "WheelEngine", FakeWheel)
    spec = engine.ModelSpec(Path("model"), Path("vision"), None, "wheel", backend="llama-cpp-python")
    try:
        for prompt in ("First ball", "Second ball"):
            _, details = engine.enhance(spec, prompt, keep_loaded=True, before_load=lambda: effects.append("unload"))
            assert details["backend"] == "llama-cpp-python" and details["wheel_version"] == "0.3.36"
        assert effects == ["unload"]
        engine.enhance(spec, "Second ball", keep_loaded=False)
        assert effects == ["unload", "close"] and engine._engine is None
    finally:
        engine.release_model()
        engine._results.clear()


def test_canvas_loader_forwards_explicit_wheel_choice_and_keeps_old_workflow_default():
    from test_nodes import run_node_script
    run_node_script("""
        calls = []
        def load(path, context, backend='auto'):
            calls.append(backend)
            return engine.ModelSpec(Path(path), Path('vision'), None, backend, context, backend=backend)
        nodes.ModelSpec.load = load
        for selected in ('auto', 'llama-cpp-python', 'llama-server'):
            result = nodes.Flux3PELoader.execute('model.gguf', backend=selected)
            assert result[0].backend == selected
            assert nodes.Flux3PELoader.fingerprint_inputs('model.gguf', backend=selected) == selected
        assert nodes.Flux3PELoader.execute('model.gguf')[0].backend == 'auto'
        assert calls == ['auto', 'auto', 'llama-cpp-python', 'llama-cpp-python', 'llama-server', 'llama-server', 'auto']
    """)
