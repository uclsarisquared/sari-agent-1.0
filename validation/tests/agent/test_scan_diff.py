"""Offline unit tests for perception._fuzzy_new_lines - the receipt-delta the Phase 6.2 scan tool
uses to decide `scanned_measured` from POS-screen OCR.

The one thing it MUST get right (the exact-string bug the probe exposed live, 2026-07-22): OCR
character jitter across frames (`JIN_RAKEN` one look, `JIN_RAMEN` the next) must NOT re-count an
already-present receipt line as a new scan, while a GENUINELY new item still must. Pure function,
no sim - importing perception is offline-safe (the OpenAI client init makes no network call).

    uv run pytest validation/tests/agent/test_scan_diff.py   # or: pytest validation/tests/agent/test_scan_diff.py
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")  # agent/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vision.perception import _fuzzy_new_lines


def test_empty_baseline_all_new():
    # First scan of a fresh checkout: everything on the screen is new.
    assert _fuzzy_new_lines([], ["1x JIN_RAMEN_MILD_120"]) == ["1x JIN_RAMEN_MILD_120"]


def test_ocr_jitter_not_counted_new():
    # Same line, different OCR read - must be absorbed, not flagged.
    before = ["1x JIN_RAKEN_MILD_120"]
    after = ["1x JIN_RAMEN_MILD_120"]
    assert _fuzzy_new_lines(before, after) == []


def test_real_run_delta():
    # The actual scan_probe.csv rows: run 1's receipt -> run 2's receipt. Only LESLIES is new;
    # the ramen line (jittered) and the changed subtotal are NOT new items.
    run1 = ["Scanned Items", "Subtotal:64.50php", "1x JIN_RAKEN_MILD_120", "-64.50php/pc"]
    run2 = ["Subtotal:97.00php", "1x JIN_RAMEN_MILD_120", "1x LESLIES_CLOVERCHIP", "-64.58php/pc"]
    new = _fuzzy_new_lines(run1, run2)
    assert "1x LESLIES_CLOVERCHIP" in new
    assert not any("JIN_RA" in l for l in new)   # the ramen line, however spelled, is not "new"


def test_genuine_duplicate_is_new():
    # Two identical items really scanned twice: the second copy IS new (multiset, not set).
    assert _fuzzy_new_lines(["1x COKE"], ["1x COKE", "1x COKE"]) == ["1x COKE"]


def test_no_change_no_new():
    lines = ["1x JIN_RAMEN_MILD_120", "Subtotal:64.50php", "Tax:7.74php"]
    assert _fuzzy_new_lines(lines, list(lines)) == []


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
