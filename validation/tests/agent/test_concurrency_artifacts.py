"""Offline regression tests for attempt-local runtime artifacts."""

from io import BytesIO
import os
from pathlib import Path
import sys

from PIL import Image
import pytest

_ROOT = Path(__file__).resolve().parents[3] / "agent"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent_core.agent import EmbodiedAgent
from orchestrator import orchestration
from sim import env
from vision import annotation_tools, perception


def _png_bytes(color):
    buf = BytesIO()
    Image.new("RGB", (8, 6), color).save(buf, format="PNG")
    return buf.getvalue()


def test_run_artifact_paths_are_attempt_local_and_legacy_fallback_is_unchanged(
        monkeypatch, tmp_path):
    monkeypatch.delenv(env.RUN_DIR_ENV, raising=False)
    assert env.screenshot_dir() == "screenshots"
    assert env.artifact_path("annotations", legacy_base="") == "annotations"

    run_dir = tmp_path / "attempt"
    monkeypatch.setenv(env.RUN_DIR_ENV, str(run_dir))
    assert env.screenshot_dir() == str(run_dir / "screenshots")
    assert env.artifact_path("annotations", legacy_base="") == str(run_dir / "annotations")


def test_screenshot_default_folder_resolves_at_call_time(monkeypatch, tmp_path):
    captured = {}

    async def fake_send(command, uri=None):
        captured.update(command)
        return {"image": b"frame"}

    monkeypatch.setenv(env.RUN_DIR_ENV, str(tmp_path / "attempt"))
    monkeypatch.setattr(env, "SendCommand", fake_send)
    assert env.RequestScreenshot(save_image=True)["image"] == b"frame"
    assert captured["folder_name"] == str(tmp_path / "attempt" / "screenshots")


def test_fallback_run_dirs_are_unique_even_in_the_same_second(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestration, "_AGENT_DIR", str(tmp_path))
    first = orchestration._resolve_run_dir(None, "graph")
    second = orchestration._resolve_run_dir(None, "graph")
    assert first != second
    assert Path(first).is_dir()
    assert Path(second).is_dir()


def test_explicit_run_dir_is_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    resolved = orchestration._resolve_run_dir(Path("runs") / "try01", "graph")
    assert Path(resolved).is_absolute()
    assert Path(resolved).is_dir()


def test_memory_snapshots_publish_inside_the_agent_run(tmp_path):
    agent = object.__new__(EmbodiedAgent)
    agent._run_dir = str(tmp_path / "try01")
    semantic = agent._run_artifact("semantic_memory.txt")
    episodic = agent._run_artifact("episodic_memory.txt")

    agent._write_text_atomic(semantic, "semantic-a")
    agent._write_text_atomic(episodic, "episodic-a")
    agent._write_text_atomic(semantic, "semantic-b")

    assert Path(semantic).read_text(encoding="utf-8") == "semantic-b"
    assert Path(episodic).read_text(encoding="utf-8") == "episodic-a"
    assert not list((tmp_path / "try01").glob("*.tmp"))


def test_depth_upload_uses_returned_frame_not_a_saved_screenshot(monkeypatch, tmp_path):
    captured = {}
    response_png = _png_bytes("black")

    class Response:
        content = response_png

        @staticmethod
        def raise_for_status():
            return None

    monkeypatch.setenv(env.RUN_DIR_ENV, str(tmp_path / "try01"))
    monkeypatch.setattr(perception, "RequestScreenshot",
                        lambda **kwargs: {"image": _png_bytes("red")})

    def fake_post(url, *, files, timeout):
        captured.update(url=url, files=files, timeout=timeout)
        return Response()

    monkeypatch.setattr(perception.requests, "post", fake_post)
    perception.request_rgbd_image("http://depth.test/estimate", timeout=7.0)

    assert captured["url"] == "http://depth.test/estimate"
    assert captured["timeout"] == 7.0
    assert captured["files"]["file"][1] == _png_bytes("red")
    assert (tmp_path / "try01" / "depth_image.png").is_file()


def test_region_ocr_uses_in_memory_crop_and_creates_no_shared_crop(monkeypatch, tmp_path):
    seen = {}

    def fake_ocr(source):
        seen["source"] = source
        return ["TOTAL 12.34"]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(perception, "ocr_lines", fake_ocr)
    frame = Image.new("RGB", (1920, 1080), "white")
    lines = perception.read_text_in_box(
        {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 250},
        source_image=frame,
    )

    assert lines == ["TOTAL 12.34"]
    assert isinstance(seen["source"], Image.Image)
    assert seen["source"].size == (216, 162)  # exact 4% padding around the 200x150 box
    assert not (tmp_path / "screenshots" / "_ocr_crop.png").exists()


def test_region_ocr_preserves_service_lines(monkeypatch):
    monkeypatch.setattr(
        perception,
        "ocr_lines",
        lambda _source: ["TOTAL 12.34", "CHANGE 0.66"],
    )
    lines = perception.read_text_in_box(
        {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 250},
        source_image=Image.new("RGB", (1920, 1080), "white"),
    )

    assert lines == ["TOTAL 12.34", "CHANGE 0.66"]


def test_region_ocr_raises_instead_of_faking_an_empty_read(monkeypatch):
    """A broken OCR service must NOT return [] — the checkout gates would read that as
    'nothing new scanned'. A missing box is still a legitimate empty answer, not an error."""
    def explode(_source):
        raise perception.OcrUnavailable("connection refused")

    monkeypatch.setattr(perception, "ocr_lines", explode)

    with pytest.raises(perception.OcrUnavailable) as excinfo:
        perception.read_text_in_box(
            {"xmin": 100, "ymin": 100, "xmax": 300, "ymax": 250},
            source_image=Image.new("RGB", (1920, 1080), "white"),
        )
    assert isinstance(excinfo.value.__cause__, perception.OcrUnavailable)
    assert "ocr-server" in str(excinfo.value)
    assert perception.read_text_in_box(None) == []


def test_annotations_default_to_the_attempt_directory(monkeypatch, tmp_path):
    run_dir = tmp_path / "try01"
    monkeypatch.setenv(env.RUN_DIR_ENV, str(run_dir))
    annotation_tools.annotate_boxes(
        {"box": {"xmin": 1, "ymin": 1, "xmax": 6, "ymax": 5}},
        source_image=Image.new("RGB", (8, 6), "white"),
    )
    assert (run_dir / "annotations" / "0.png").is_file()
