"""test_plan_reach.py - offline unit table for manipulation.plan_reach (Phase D geometry brain).

No sim: feeds synthetic RequestLidarCenter samples and checks verdict / move_steps / target height for
the MEASURED slant-distance model (reachable iff distance <= max_reach; the hand extends along the
gaze). Runs against a FROZEN reference envelope (REF_ENVELOPE), NOT the live manipulation.REACH_ENVELOPE,
so recalibrating the live max_reach never breaks this logic test.

    python test_plan_reach.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent"))  # agent/ — manipulation lives there

from manip.manipulation import plan_reach

REF_ENVELOPE = {"max_reach": 0.85, "move_unit": 0.10, "move_cap": 10}


def sample(distance, pitch_deg, camera_height, hit=True):
    return {"distance": distance, "pitch_deg": pitch_deg, "camera_height": camera_height,
            "hit": hit, "min_range": 0.05, "max_range": 10.0}


def approx(a, b, tol=0.02):
    return a is not None and abs(a - b) <= tol


# (name, sample, expected_verdict, expected_target_height) - verdicts under REF_ENVELOPE (max_reach 0.85)
CASES = [
    ("reachable: within slant reach",  sample(0.60,  20.0, 1.30), "reachable", 1.095),
    ("move: too far, small vert gap",  sample(1.00,  10.0, 1.30), "move",      1.126),
    ("crouch: too low (below reach)",  sample(1.20,  60.0, 1.30), "crouch",    0.261),
    ("bail: too high (above reach)",   sample(1.20, -60.0, 1.30), "bail",      2.339),
    ("recenter: miss (hit=False)",     sample(2.00,  20.0, 1.30, hit=False), "recenter", None),
]


def main():
    fails = 0
    for name, s, want_verdict, want_h in CASES:
        p = plan_reach(s, REF_ENVELOPE)
        th = (s["camera_height"] - s["distance"] * math.sin(math.radians(s["pitch_deg"]))
              if s["hit"] else None)
        ok_v = p["verdict"] == want_verdict
        ok_h = (want_h is None) or approx(p["target_height"], want_h)
        ok_trig = (th is None) or approx(p["target_height"], th)   # branch-independent trig
        ok = ok_v and ok_h and ok_trig
        fails += 0 if ok else 1
        extra = f" move_steps={p['move_steps']}" if p["verdict"] == "move" else ""
        th_str = "None" if p["target_height"] is None else f"{p['target_height']:.3f}"
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: verdict={p['verdict']} (want {want_verdict}) "
              f"target_h={th_str} (want {want_h}){extra}")

    mr, unit = REF_ENVELOPE["max_reach"], REF_ENVELOPE["move_unit"]

    # the reach boundary itself: distance just under max_reach grabs, just over does not
    assert plan_reach(sample(mr - 0.01, 0.0, 1.30), REF_ENVELOPE)["verdict"] == "reachable"
    assert plan_reach(sample(mr + 0.01, 0.0, 1.30), REF_ENVELOPE)["verdict"] == "move"

    # KEY INVARIANT: after moving move_steps forward, the slant distance must actually be <= max_reach
    # (the move brings the target into reach in one shot, not just "closer").
    mv_s = sample(1.00, 10.0, 1.30)
    mv = plan_reach(mv_s, REF_ENVELOPE)
    vo = mv_s["distance"] * math.sin(math.radians(mv_s["pitch_deg"]))
    new_gap = mv["horizontal_gap"] - mv["move_steps"] * unit
    new_dist = math.hypot(max(0.0, new_gap), vo)
    assert new_dist <= mr + 1e-9, f"move undershoots reach: new slant {new_dist:.3f} > {mr}"

    # crouch vs bail sign: same steep angle, opposite pitch -> below=crouch, above=bail
    assert plan_reach(sample(1.20, 60.0, 1.30), REF_ENVELOPE)["verdict"] == "crouch"
    assert plan_reach(sample(1.20, -60.0, 1.30), REF_ENVELOPE)["verdict"] == "bail"

    # unavailable: a pre-Phase-D sample (no pose) must NOT crash and must ask the caller to fall back
    un = plan_reach({"distance": 0.6, "hit": True, "min_range": 0.05, "max_range": 10.0}, REF_ENVELOPE)
    assert un["verdict"] == "unavailable", f"missing pose should be 'unavailable', got {un['verdict']}"

    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}  "
          f"(REF_ENVELOPE frozen: max_reach={mr})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
