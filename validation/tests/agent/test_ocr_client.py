from __future__ import annotations

import os
import sys
from io import BytesIO

import pytest
import requests
from PIL import Image

_ROOT = os.path.dirname(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.vision import ocr_client


class Response:
    def __init__(self, payload=None, *, status=200, json_error=None):
        self.payload = payload
        self.status_code = status
        self.json_error = json_error

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        if self.json_error:
            raise self.json_error
        return self.payload


def test_url_precedence(monkeypatch):
    monkeypatch.delenv(ocr_client.OCR_URL_ENV, raising=False)
    assert ocr_client.resolve_ocr_url() == ocr_client.DEFAULT_OCR_URL
    monkeypatch.setenv(ocr_client.OCR_URL_ENV, "http://env.test:9200/")
    assert ocr_client.resolve_ocr_url() == "http://env.test:9200"
    assert ocr_client.resolve_ocr_url("http://flag.test:9300/") == "http://flag.test:9300"


def test_png_upload_and_valid_or_empty_response():
    calls = []

    class Session:
        @staticmethod
        def post(url, **kwargs):
            calls.append((url, kwargs))
            return Response({"lines": [], "model": "fake", "elapsed_ms": 1.25})

    assert ocr_client.ocr_lines(Image.new("RGB", (4, 3), "red"), session=Session) == []
    url, kwargs = calls[0]
    assert url.endswith("/v1/ocr")
    assert kwargs["headers"]["Content-Type"] == "image/png"
    assert kwargs["data"].startswith(b"\x89PNG\r\n\x1a\n")
    assert kwargs["timeout"] == 60.0
    with Image.open(BytesIO(kwargs["data"])) as uploaded:
        assert uploaded.size == (4, 3)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"lines": "not-a-list", "model": "fake", "elapsed_ms": 1},
        {"lines": [1], "model": "fake", "elapsed_ms": 1},
        {"lines": [], "model": "", "elapsed_ms": 1},
        {"lines": [], "model": "fake", "elapsed_ms": "fast"},
    ],
)
def test_malformed_response_raises(payload):
    class Session:
        @staticmethod
        def post(*_args, **_kwargs):
            return Response(payload)

    with pytest.raises(ocr_client.OcrUnavailable, match="invalid schema"):
        ocr_client.ocr_lines(Image.new("RGB", (2, 2)), session=Session)


@pytest.mark.parametrize("failure", [requests.Timeout("slow"), requests.ConnectionError("down")])
def test_transport_failures_raise(failure):
    class Session:
        @staticmethod
        def post(*_args, **_kwargs):
            raise failure

    with pytest.raises(ocr_client.OcrUnavailable, match="ocr-server"):
        ocr_client.ocr_lines(Image.new("RGB", (2, 2)), session=Session)


def test_http_and_malformed_json_raise():
    class HttpSession:
        @staticmethod
        def post(*_args, **_kwargs):
            return Response(status=503)

    with pytest.raises(ocr_client.OcrUnavailable):
        ocr_client.ocr_lines(Image.new("RGB", (2, 2)), session=HttpSession)

    class JsonSession:
        @staticmethod
        def post(*_args, **_kwargs):
            return Response(json_error=ValueError("bad json"))

    with pytest.raises(ocr_client.OcrUnavailable):
        ocr_client.ocr_lines(Image.new("RGB", (2, 2)), session=JsonSession)


def test_health_schema_and_url():
    seen = {}

    class Session:
        @staticmethod
        def get(url, **kwargs):
            seen.update(url=url, kwargs=kwargs)
            return Response({"ready": True, "api_version": "v1", "model": "fake"})

    assert ocr_client.check_ocr_health("http://ocr.test/", session=Session)["model"] == "fake"
    assert seen["url"] == "http://ocr.test/health"
