import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from test_audit_v2_build_inputs import download


@pytest.mark.parametrize("failure", ["status", "header", "short"])
def test_concurrent_resume_and_retry_over_real_loopback_http(download, monkeypatch, failure):
    module, destination, data, _ = download
    (destination / "model.safetensors.download").write_bytes(data[:4] + b"?" * 6)
    (destination / "model.safetensors.ranges.json").write_text(json.dumps({
        "0": hashlib.sha256(data[:4]).hexdigest(), "1": hashlib.sha256(data[4:8]).hexdigest()}))
    seen = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            value = self.headers["Range"]
            with lock:
                attempt = seen.count(value)
                seen.append(value)
            start, end = map(int, value.removeprefix("bytes=").split("-"))
            retry = value == "bytes=4-7" and attempt == 0
            body = data[start:end + 1]
            if retry and failure == "short": body = body[:-1]
            self.send_response(503 if retry and failure == "status" else 206)
            self.send_header("Content-Range", "bytes 0-0/10" if retry and failure == "header" else f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    real_get = __import__("requests").api.get
    monkeypatch.setattr(module.requests, "get", lambda url, **kwargs: real_get(f"http://127.0.0.1:{server.server_port}/model", **kwargs))
    import sys
    monkeypatch.setattr(sys, "argv", ["download_model.py", "--destination", str(destination), "--workers", "3"])
    worker.start()
    try:
        module.main()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
    assert not worker.is_alive()
    assert (destination / "model.safetensors").read_bytes() == data
    assert "bytes=0-3" not in seen
    assert seen.count("bytes=4-7") == 2 and seen.count("bytes=8-9") == 1
    assert not (destination / "model.safetensors.download").exists()
    journal = json.loads((destination / "model.safetensors.ranges.json").read_text())
    assert journal == {str(i): hashlib.sha256(data[i * 4:(i + 1) * 4]).hexdigest() for i in range(3)}
