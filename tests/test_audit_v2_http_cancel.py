"""Cancellation stays bounded even if its owned server cannot be killed."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading

import pytest

import flux3_pe.engine as backend
from flux3_pe.engine import LlamaEngine, ModelSpec
from flux3_pe.protocol import PEError


@pytest.fixture
def pending_http():
    entered, respond = threading.Event(), threading.Event()
    received = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            entered.set()
            assert respond.wait(3), "Mock response was never released"
            body = json.dumps({"choices": [{"message": {"content": "A ball."}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", entered, respond, received
    finally:
        respond.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class StubbornOwnedProcess:
    def __init__(self):
        self.exited = False
        self.wait_timeouts = []
    def poll(self):
        return 0 if self.exited else None
    def terminate(self):
        raise PermissionError("simulated permission failure")
    def kill(self):
        raise PermissionError("simulated permission failure")
    def wait(self, timeout):
        self.wait_timeouts.append(timeout)
        if not self.exited:
            raise subprocess.TimeoutExpired("owned fixture", timeout)
        return 0


def test_cancel_does_not_join_a_stuck_http_worker_unboundedly(pending_http):
    url, entered, respond, received = pending_http
    engine = LlamaEngine(ModelSpec(Path("model"), Path("vision"), Path("server"), "cancel-test"))
    engine.url = url
    process = StubbornOwnedProcess()
    engine.process = process
    cancel, returned = threading.Event(), threading.Event()
    outcome = {}
    before = set(thread.ident for thread in threading.enumerate())
    def interrupt():
        if cancel.is_set():
            raise InterruptedError("cancelled")
    def call():
        try:
            engine.complete({"messages": []}, interrupt)
        except BaseException as error:
            outcome["error"] = error
        finally:
            returned.set()
    caller = threading.Thread(target=call, name="audit-cancel-caller")
    caller.start()
    timely = False
    try:
        assert entered.wait(2)
        cancel.set()
        timely = returned.wait(0.8)
        if timely:
            assert isinstance(outcome.get("error"), PEError), outcome
            assert engine.process is process
            assert process.wait_timeouts and max(process.wait_timeouts) <= 0.25
            pending = [thread for thread in threading.enumerate() if thread.ident not in before and thread is not caller]
            assert all(thread.daemon for thread in pending), "Unfinished non-daemon HTTP worker can block Python shutdown"
            assert len(received) == 1, "Cancellation resent the POST"
            with pytest.raises(PEError, match="refusing another POST"):
                engine.complete({"messages": []})
            with pytest.raises(PEError, match="awaiting cleanup"):
                engine.start()
            assert len(received) == 1
    finally:
        respond.set()
        caller.join(timeout=2)
        process.exited = True
        engine.close()
    assert timely, "Cancellation kept waiting for HTTP response after cleanup failed"
    assert not caller.is_alive()


def test_global_cancel_retains_handle_and_blocks_post_until_release(pending_http, monkeypatch):
    url, entered, respond, received = pending_http
    backend.release_model()
    backend._results.clear()
    spec = ModelSpec(Path("model"), Path("vision"), Path("server"), "global-cancel")
    value = LlamaEngine(spec)
    value.url = url
    process = StubbornOwnedProcess()
    value.process = process
    backend._engine = value
    cancel, returned = threading.Event(), threading.Event()
    outcome = {}
    def interrupt():
        if cancel.is_set():
            raise InterruptedError("cancelled global request")
    def call():
        try:
            backend.enhance(spec, "A ball", keep_loaded=True, use_cache=False, interrupt=interrupt)
        except BaseException as error:
            outcome["error"] = error
        finally:
            returned.set()
    caller = threading.Thread(target=call, name="audit-global-cancel-caller")
    caller.start()
    replacement = None
    try:
        assert entered.wait(2)
        cancel.set()
        assert returned.wait(0.8), "enhance repeated a long cleanup after cancellation"
        caller.join(1)
        assert isinstance(outcome.get("error"), PEError)
        assert backend._engine is value and value.process is process
        with pytest.raises(PEError, match="awaiting cleanup"):
            backend.enhance(spec, "Another ball", use_cache=False)
        assert len(received) == 1 and not backend._lock.locked()

        respond.set()
        value._request_worker.join(timeout=2)
        assert not value._request_worker.is_alive()
        process.exited = True
        backend.release_model()
        assert backend._engine is None

        replacement = LlamaEngine(spec)
        replacement.url = url
        replacement.process = StubbornOwnedProcess()
        monkeypatch.setattr(backend, "LlamaEngine", lambda spec: replacement)
        caption, details = backend.enhance(spec, "A recovered ball", keep_loaded=True, use_cache=False)
        assert caption == "A ball." and details["cache_hit"] is False
        assert len(received) == 2 and replacement._request_worker is None
    finally:
        respond.set()
        caller.join(2)
        process.exited = True
        if replacement is not None:
            replacement.process.exited = True
        backend.release_model()
        backend._results.clear()
