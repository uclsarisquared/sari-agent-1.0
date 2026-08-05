"""test_plan_place.py - offline unit table for manipulation.plan_place (Phase 6.2 place geometry).

No sim: feeds synthetic RequestLidarCenter samples of the bagging tray and checks verdict / move_steps
/ surface height for the measured slant-distance model (place iff place_min <= slant <= place_max; the
held item extends along the gaze). Runs against a FROZEN reference envelope (REF_ENVELOPE), NOT the
live manipulation.PLACE_ENVELOPE, so recalibrating the live place_max never breaks this logic test.

Two envelopes are exercised: FAR-ONLY (place_min=None, the shipped shape) and a WITH-NEAR variant, so
the dormant 'back' verdict is still covered against the day a near bound gets measured.

    python test_plan_place.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent"))  # agent/ — manipulation lives there

from manip.manipulation import plan_place

REF_ENVELOPE = {"place_max": 1.28, "place_min": None, "move_unit": 0.10, "move_cap": 10}
NEAR_ENVELOPE = {"place_max": 1.28, "place_min": 0.90, "move_unit": 0.10, "move_cap": 10}


def sample(distance, pitch_deg, camera_height, hit=True):
    return {"distance": distance, "pitch_deg": pitch_deg, "camera_height": camera_height,
            "hit": hit, "min_range": 0.05, "max_range": 10.0}


def approx(a, b, tol=0.02):
    return a is not None and abs(a - b) <= tol


# (name, sample, envelope, expected_verdict, expected_surface_height)
CASES = [
    ("placeable: within range",        sample(1.20, 38.0, 1.46), REF_ENVELOPE,  "placeable", 0.721),
    ("move: too far, surface reachable", sample(1.45, 37.0, 1.46), REF_ENVELOPE, "move",      0.588),
    ("recenter: miss (hit=False)",     sample(2.00, 38.0, 1.46, hit=False), REF_ENVELOPE, "recenter", None),
    ("back: too close (near bound)",   sample(0.80, 40.0, 1.46), NEAR_ENVELOPE, "back",      0.946),
    ("placeable also under near env",  sample(1.10, 38.0, 1.46), NEAR_ENVELOPE, "placeable", 0.783),
]


def main():
    fails = 0
    for name, s, env, want_verdict, want_h in CASES:
        p = plan_place(s, env)
        sh = (s["camera_height"] - s["distance"] * math.sin(math.radians(s["pitch_deg"]))
              if s["hit"] else None)
        ok_v = p["verdict"] == want_verdict
        ok_h = (want_h is None) or approx(p["surface_height"], want_h)
        ok_trig = (sh is None) or approx(p["surface_height"], sh)   # branch-independent trig
        ok = ok_v and ok_h and ok_trig
        fails += 0 if ok else 1
        extra = f" move_steps={p['move_steps']}" if p["verdict"] in ("move", "back") else ""
        sh_str = "None" if p["surface_height"] is None else f"{p['surface_height']:.3f}"
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: verdict={p['verdict']} (want {want_verdict}) "
              f"surface_h={sh_str} (want {want_h}){extra}")

    pm, unit = REF_ENVELOPE["place_max"], REF_ENVELOPE["move_unit"]

    # the place boundary itself: slant just under place_max places, just over -> move
    assert plan_place(sample(pm - 0.01, 0.0, 1.46), REF_ENVELOPE)["verdict"] == "placeable"
    assert plan_place(sample(pm + 0.01, 0.0, 1.46), REF_ENVELOPE)["verdict"] == "move"

    # KEY INVARIANT: after moving move_steps forward, the slant must actually be <= place_max
    # (the move brings the tray into place range in one shot, not just "closer").
    mv_s = sample(1.45, 37.0, 1.46)
    mv = plan_place(mv_s, REF_ENVELOPE)
    vo = mv_s["distance"] * math.sin(math.radians(mv_s["pitch_deg"]))
    new_gap = mv["horizontal_gap"] - mv["move_steps"] * unit
    new_dist = math.hypot(max(0.0, new_gap), vo)
    assert new_dist <= pm + 1e-9, f"move undershoots place range: new slant {new_dist:.3f} > {pm}"

    # the far-only shipped envelope must NEVER emit 'back' - the near verdict is dormant until measured
    for d in (0.10, 0.40, 0.80, 1.0):
        assert plan_place(sample(d, 40.0, 1.46), REF_ENVELOPE)["verdict"] != "back", \
            f"far-only envelope emitted 'back' at slant {d} (place_min is None)"

    # near env: after moving BACK, slant must be >= place_min (clears the lip in one shot)
    bk_s = sample(0.80, 40.0, 1.46)
    bk = plan_place(bk_s, NEAR_ENVELOPE)
    vo_b = bk_s["distance"] * math.sin(math.radians(bk_s["pitch_deg"]))
    new_gap_b = bk["horizontal_gap"] + bk["move_steps"] * unit
    new_dist_b = math.hypot(new_gap_b, vo_b)
    assert new_dist_b >= NEAR_ENVELOPE["place_min"] - 1e-9, \
        f"back undershoots near bound: new slant {new_dist_b:.3f} < {NEAR_ENVELOPE['place_min']}"

    # unavailable: a pre-Phase-D sample (no pose) must NOT crash and must ask the caller to fall back
    un = plan_place({"distance": 1.2, "hit": True, "min_range": 0.05, "max_range": 10.0}, REF_ENVELOPE)
    assert un["verdict"] == "unavailable", f"missing pose should be 'unavailable', got {un['verdict']}"

    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}  "
          f"(REF_ENVELOPE frozen: place_max={pm}, place_min={REF_ENVELOPE['place_min']})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
