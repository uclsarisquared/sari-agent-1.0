"""HTTP client for the runner-local PaddleOCR service.

This module deliberately contains no Paddle imports.  Agent processes only crop/encode images and
call the daemon; the daemon is the sole owner of the model and result-shape compatibility code.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

DEFAULT_OCR_URL = "http://127.0.0.1:9100"
DEFAULT_OCR_TIMEOUT = 60.0
DEFAULT_OCR_HEALTH_TIMEOUT = 5.0
OCR_URL_ENV = "SARI_OCR_URL"
API_VERSION = "v1"


class OcrUnavailable(RuntimeError):
    """The OCR service could not produce a trustworthy reading."""


def resolve_ocr_url(explicit: str | None = None) -> str:
    """Resolve explicit flag, environment, then runner-local default, in that order."""
    value = explicit or os.environ.get(OCR_URL_ENV) or DEFAULT_OCR_URL
    value = value.strip().rstrip("/")
    if not value:
        return DEFAULT_OCR_URL
    return value


def _request_error(action: str, url: str, error: BaseException) -> OcrUnavailable:
    return OcrUnavailable(
        f"OCR service {action} failed at {url} ({type(error).__name__}: {error}). "
        "Start it with `uv run poe ocr-server` and verify its /health endpoint."
    )


def check_ocr_health(
    ocr_url: str | None = None,
    *,
    timeout: float = DEFAULT_OCR_HEALTH_TIMEOUT,
    session: Any = requests,
) -> dict[str, Any]:
    """Require a ready, version-compatible OCR daemon and return its identity payload."""
    base_url = resolve_ocr_url(ocr_url)
    url = f"{base_url}/health"
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError, TypeError) as error:
        raise _request_error("health check", url, error) from error

    if (
        not isinstance(payload, dict)
        or payload.get("ready") is not True
        or payload.get("api_version") != API_VERSION
        or not isinstance(payload.get("model"), str)
        or not payload["model"]
    ):
        raise OcrUnavailable(f"OCR service health response at {url} has an invalid schema: {payload!r}")
    return payload


def _png_bytes(source: Image.Image | str | os.PathLike[str] | bytes | bytearray) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, Image.Image):
        image = source
        close = False
    else:
        image = Image.open(Path(source))
        close = True
    try:
        image.load()
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    finally:
        if close:
            image.close()


def ocr_lines(
    source: Image.Image | str | os.PathLike[str] | bytes | bytearray,
    ocr_url: str | None = None,
    *,
    timeout: float = DEFAULT_OCR_TIMEOUT,
    session: Any = requests,
) -> list[str]:
    """Encode ``source`` as PNG and return validated line strings from the OCR API."""
    base_url = resolve_ocr_url(ocr_url)
    url = f"{base_url}/v1/ocr"
    try:
        body = _png_bytes(source)
        response = session.post(
            url,
            data=body,
            headers={"Content-Type": "image/png"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, OSError, ValueError, TypeError) as error:
        raise _request_error("request", url, error) from error

    valid = (
        isinstance(payload, dict)
        and isinstance(payload.get("lines"), list)
        and all(isinstance(line, str) for line in payload["lines"])
        and isinstance(payload.get("model"), str)
        and payload["model"]
        and isinstance(payload.get("elapsed_ms"), (int, float))
        and not isinstance(payload.get("elapsed_ms"), bool)
    )
    if not valid:
        raise OcrUnavailable(f"OCR service response at {url} has an invalid schema: {payload!r}")
    return payload["lines"]
