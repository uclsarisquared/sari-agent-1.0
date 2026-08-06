from __future__ import annotations

import os
import sys
import threading
import time
from io import BytesIO

import pytest
import requests
from PIL import Image

_ROOT = os.path.dirname(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.vision.ocr_server import OcrApplication, create_server, parse_paddle_lines


def png_bytes(size=(8, 6)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, "PNG")
    return buffer.getvalue()


def test_both_paddle_result_shapes_and_empty():
    assert parse_paddle_lines([{"rec_texts": [" A ", "", "B"]}]) == ["A", "B"]
    assert parse_paddle_lines([[(None, (" A ", 0.9)), (None, ("B", 0.8))]]) == ["A", "B"]
    assert parse_paddle_lines([]) == []
    assert parse_paddle_lines([[]]) == []


def test_invalid_images_and_request_limit():
    app = OcrApplication(lambda: object(), max_request_bytes=64)
    with pytest.raises(ValueError, match="valid PNG"):
        app.infer_png(b"not an image")
    with pytest.raises(OverflowError):
        app.infer_png(b"x" * 65)


def test_slow_engine_is_constructed_once_and_inference_is_serialized():
    state = {"constructed": 0, "active": 0, "peak": 0, "calls": 0}
    state_lock = threading.Lock()

    class Engine:
        def predict(self, _source):
            with state_lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.03)
            with state_lock:
                state["active"] -= 1
                state["calls"] += 1
            return [{"rec_texts": ["OK"]}]

    def factory():
        state["constructed"] += 1
        return Engine()

    app = OcrApplication(factory, max_in_flight=24)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(app.infer_png(png_bytes())[0]))
        for _ in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state == {"constructed": 1, "active": 0, "peak": 1, "calls": 20}
    assert results == [["OK"]] * 20


def test_http_health_structured_errors_size_limit_and_overload():
    entered = threading.Event()
    release = threading.Event()

    class Engine:
        def predict(self, _source):
            entered.set()
            release.wait(2)
            return [{"rec_texts": ["OK"]}]

    server = create_server(
        "127.0.0.1",
        0,
        engine_factory=lambda: Engine(),
        model="fake-model",
        max_in_flight=1,
        max_request_bytes=1024,
    )
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        health = requests.get(f"{base}/health", timeout=2)
        assert health.json() == {"ready": True, "api_version": "v1", "model": "fake-model"}

        invalid = requests.post(f"{base}/v1/ocr", data=b"bad", timeout=2)
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "invalid_image"

        large = requests.post(f"{base}/v1/ocr", data=b"x" * 1025, timeout=2)
        assert large.status_code == 413
        assert large.json()["error"]["code"] == "request_too_large"

        first = threading.Thread(
            target=lambda: requests.post(f"{base}/v1/ocr", data=png_bytes(), timeout=3)
        )
        first.start()
        assert entered.wait(1)
        overloaded = requests.post(f"{base}/v1/ocr", data=png_bytes(), timeout=2)
        assert overloaded.status_code == 503
        assert overloaded.json()["error"]["code"] == "overloaded"
        release.set()
        first.join()
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        serving.join()
