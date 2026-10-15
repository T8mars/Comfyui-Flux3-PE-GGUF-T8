from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import threading
import time

import pytest
import requests

from flux3_pe.engine import LlamaEngine, ModelSpec
from flux3_pe.protocol import PEError


@contextmanager
def mock_backend(mode):
    entered, stopped = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            entered.set()
            if mode == "cancel":
                assert stopped.wait(timeout=3), "Test process was not terminated"
            if mode in ("disconnect", "cancel"):
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            payload = b"backend failed" if mode == "http500" else b"not JSON"
            self.send_response(500 if mode == "http500" else 200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", entered, stopped
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class FakeProcess:
    def __init__(self, stopped):
        self.stopped = stopped
        self.terminated = False
    def poll(self):
        return 0 if self.terminated else None
    def terminate(self):
        self.terminated = True
        self.stopped.set()
    def wait(self, timeout=None):
        return 0


def engine(url, stopped):
    backend = LlamaEngine(ModelSpec(Path("model"), Path("vision"), Path("server"), "test"))
    backend.url = url
    process = FakeProcess(stopped)
    backend.process = process
    return backend, process


@pytest.mark.parametrize("mode,exception", [("http500", PEError), ("badjson", requests.exceptions.JSONDecodeError), ("disconnect", requests.ConnectionError)])
def test_transport_failure_closes_process_and_joins_worker(mode, exception):
    with mock_backend(mode) as (url, _, stopped):
        backend, process = engine(url, stopped)
        before = {thread.ident for thread in threading.enumerate()}
        with pytest.raises(exception):
            backend.complete({"messages": []})
        assert process.terminated and backend.process is None
        assert not [thread for thread in threading.enumerate() if thread.ident not in before and thread.name.startswith(("ThreadPoolExecutor", "Flux3PEHTTP"))]
        backend.session.close()


def test_interrupt_terminates_owned_process_and_joins_active_http_worker():
    with mock_backend("cancel") as (url, entered, stopped):
        backend, process = engine(url, stopped)
        before = {thread.ident for thread in threading.enumerate()}
        def interrupt():
            if entered.is_set():
                raise InterruptedError("user cancelled")
        started = time.monotonic()
        with pytest.raises(InterruptedError, match="user cancelled"):
            backend.complete({"messages": []}, interrupt=interrupt)
        assert time.monotonic() - started < 2
        assert entered.is_set() and process.terminated and backend.process is None
        assert not [thread for thread in threading.enumerate() if thread.ident not in before and thread.name.startswith(("ThreadPoolExecutor", "Flux3PEHTTP"))]
        backend.session.close()
