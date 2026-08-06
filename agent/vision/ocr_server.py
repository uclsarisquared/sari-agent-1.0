"""One-process, serialized PaddleOCR HTTP service for all local agent workers."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError

from .ocr_client import API_VERSION

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9100
DEFAULT_MAX_REQUEST_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_IN_FLIGHT = 32
MODEL_IDENTITY = "paddleocr:en:text-line-orientation"


def build_paddle_engine() -> Any:
    """Construct the measured five-component pipeline exactly once, in the server process."""
    from paddleocr import PaddleOCR

    return PaddleOCR(lang="en", use_textline_orientation=True, enable_mkldnn=False)


def parse_paddle_lines(result: Any) -> list[str]:
    """Normalize PaddleOCR 3.x dict pages and legacy 2.x tuple pages."""
    if not result or not result[0]:
        return []
    page = result[0]
    if isinstance(page, dict):
        texts = page.get("rec_texts") or []
    else:
        texts = [line[1][0] for line in page]
    return [text.strip() for text in texts if isinstance(text, str) and text.strip()]


class OcrApplication:
    """Owns one engine, one inference lock, and a bounded set of admitted requests."""

    def __init__(
        self,
        engine_factory: Callable[[], Any] = build_paddle_engine,
        *,
        model: str = MODEL_IDENTITY,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        if max_request_bytes < 1:
            raise ValueError("max_request_bytes must be at least 1")
        self.model = model
        self.max_request_bytes = max_request_bytes
        self._engine = engine_factory()
        self._admission = threading.BoundedSemaphore(max_in_flight)
        self._inference_lock = threading.Lock()

    def infer_png(self, body: bytes) -> tuple[list[str], float]:
        if not body:
            raise ValueError("request body is empty")
        if len(body) > self.max_request_bytes:
            raise OverflowError(
                f"request body exceeds {self.max_request_bytes} byte limit"
            )
        try:
            with Image.open(BytesIO(body)) as opened:
                opened.load()
                if opened.format != "PNG":
                    raise ValueError("request body is not a PNG image")
                image = opened.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("request body is not a valid PNG image") from error

        import numpy as np

        source = np.asarray(image)
        started = time.perf_counter()
        with self._inference_lock:
            predict = getattr(self._engine, "predict", None)
            result = predict(source) if callable(predict) else self._engine.ocr(source)
        return parse_paddle_lines(result), round((time.perf_counter() - started) * 1000.0, 2)

    def try_admit(self) -> bool:
        return self._admission.acquire(blocking=False)

    def release(self) -> None:
        self._admission.release()


def make_handler(application: OcrApplication) -> type[BaseHTTPRequestHandler]:
    class OcrHandler(BaseHTTPRequestHandler):
        server_version = "SariOCR/1"

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, code: str, message: str) -> None:
            self._json(status, {"error": {"code": code, "message": message}})

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/health":
                self._error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
                return
            self._json(
                HTTPStatus.OK,
                {"ready": True, "api_version": API_VERSION, "model": application.model},
            )

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path != "/v1/ocr":
                self._error(HTTPStatus.NOT_FOUND, "not_found", "unknown endpoint")
                return
            if not application.try_admit():
                self._error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "overloaded",
                    "OCR request queue is full",
                )
                return
            try:
                length_header = self.headers.get("Content-Length")
                if length_header is None:
                    self._error(
                        HTTPStatus.LENGTH_REQUIRED,
                        "length_required",
                        "Content-Length is required",
                    )
                    return
                try:
                    length = int(length_header)
                except ValueError:
                    self._error(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_length",
                        "Content-Length must be an integer",
                    )
                    return
                if length < 0:
                    self._error(
                        HTTPStatus.BAD_REQUEST,
                        "invalid_length",
                        "Content-Length cannot be negative",
                    )
                    return
                if length > application.max_request_bytes:
                    self._error(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        "request_too_large",
                        f"PNG exceeds {application.max_request_bytes} byte limit",
                    )
                    return
                body = self.rfile.read(length)
                try:
                    lines, elapsed_ms = application.infer_png(body)
                except OverflowError as error:
                    self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large", str(error))
                    return
                except ValueError as error:
                    self._error(HTTPStatus.BAD_REQUEST, "invalid_image", str(error))
                    return
                except Exception as error:  # Paddle failures are server errors, never fake empties.
                    self._error(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "inference_failed",
                        f"{type(error).__name__}: {error}",
                    )
                    return
                self._json(
                    HTTPStatus.OK,
                    {"lines": lines, "model": application.model, "elapsed_ms": elapsed_ms},
                )
            finally:
                application.release()

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[ocr-server] {self.address_string()} {fmt % args}", flush=True)

    return OcrHandler


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    engine_factory: Callable[[], Any] = build_paddle_engine,
    model: str = MODEL_IDENTITY,
    max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> ThreadingHTTPServer:
    application = OcrApplication(
        engine_factory,
        model=model,
        max_in_flight=max_in_flight,
        max_request_bytes=max_request_bytes,
    )
    server = ThreadingHTTPServer((host, port), make_handler(application))
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the central runner-local PaddleOCR service.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--max-in-flight", type=int, default=DEFAULT_MAX_IN_FLIGHT)
    parser.add_argument("--max-request-bytes", type=int, default=DEFAULT_MAX_REQUEST_BYTES)
    args = parser.parse_args(argv)

    print("[ocr-server] loading PaddleOCR pipeline...", flush=True)
    server = create_server(
        args.host,
        args.port,
        max_in_flight=args.max_in_flight,
        max_request_bytes=args.max_request_bytes,
    )
    host, port = server.server_address[:2]
    print(
        f"[ocr-server] ready on http://{host}:{port} "
        f"(model={MODEL_IDENTITY}, max_in_flight={args.max_in_flight})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
