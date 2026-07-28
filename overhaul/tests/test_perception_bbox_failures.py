"""Regression tests for bbox parse failures being retried and surfaced distinctly."""

import os
import sys
from io import BytesIO
from unittest.mock import patch

from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vision import perception as p


def _png_bytes():
    buf = BytesIO()
    Image.new("RGB", (2, 2)).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_empty_bbox_is_a_real_negative_without_retry():
    replies = iter(["[]"])
    calls = []

    def request(*args):
        calls.append(args)
        return next(replies)

    with patch.object(p, "_bbox_request", request):
        assert p._detect_boxes_px(object(), "target") == []
    assert len(calls) == 1


def test_malformed_bbox_is_retried_then_succeeds():
    replies = iter([
        "[{'box_2d': [1, 2, 3",  # truncated
        "[{'box_2d': [100, 200, 300, 400], 'label': 'item'}]",
    ])
    calls = []

    def request(*args):
        calls.append(args)
        return next(replies)

    with patch.object(p, "_bbox_request", request):
        boxes = p._detect_boxes_px(object(), "target")
    assert len(calls) == 2
    assert boxes[0]["label"] == "item"


def test_invalid_bbox_shape_is_retried_and_eventually_raises():
    calls = []

    def request(*args):
        calls.append(args)
        return "[{'box_2d': [100, 200, 300]}]"

    with patch.object(p, "_bbox_request", request):
        try:
            p._detect_boxes_px(object(), "target")
        except p.BBoxResponseParseError as error:
            assert "after 3 attempts" in str(error)
        else:
            raise AssertionError("malformed bbox response did not raise")
    assert len(calls) == p.DETECTION_PARSE_MAX_ATTEMPTS


def test_center_surfaces_parse_exhaustion_as_detection_error():
    parse_error = p.BBoxResponseParseError("truncated response after retries")
    with (
        patch.object(p, "TransformAgent", return_value={}),
        patch.object(p, "RequestScreenshot", return_value={"image": _png_bytes()}),
        patch.object(p, "_detect_boxes_px", side_effect=parse_error),
    ):
        result = p.center_object_on_screen("target", max_iters=0)

    assert result["outcome"] == "detection_error"
    assert result["detected"] is False
    assert "not evidence that the target is absent" in result["center_message"]
    assert result["detection_error"] == str(parse_error)
