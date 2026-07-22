"""test_plan_reach.py - offline unit table for manipulation.plan_reach (Phase D geometry brain).

No sim: feeds synthetic RequestLidarCenter samples and checks verdict / move_steps / target height.
This validates the GEOMETRY and the 4-way branch. It runs against a FROZEN reference envelope
(REF_ENVELOPE below), NOT the live manipulation.REACH_ENVELOPE - so recalibrating the live constants
after a reach_probe run never breaks this logic test. (The live constants are validated separately by
the calibration + the task-level A/B, not here.)

    python test_plan_reach.py
"""
import math

from manipulation import plan_reach

# Frozen first-guess envelope - the test asserts branch LOGIC against these fixed values, so it is
# independent of whatever the live REACH_ENVELOPE is calibrated to.
REF_ENVELOPE = {
    "hand_drop": 0.25, "r_eff": 0.40, "standoff": 0.35,
    "reach_tol": 0.05, "move_unit": 0.10, "move_cap": 10,
}


def sample(distance, pitch_deg, camera_height, hit=True):
    return {"distance": distance, "pitch_deg": pitch_deg, "camera_height": camera_height,
            "hit": hit, "min_range": 0.05, "max_range": 10.0}


def approx(a, b, tol=0.02):
    return a is not None and abs(a - b) <= tol


# (name, sample, expected_verdict, expected_target_height) - verdicts under REF_ENVELOPE
CASES = [
    ("reachable: close, mid shelf",   sample(0.40,  25.0, 1.30), "reachable", 1.131),
    ("move: far, mid shelf",          sample(0.80,  20.0, 1.30), "move",      1.026),
    ("crouch: bottom shelf",          sample(0.70,  70.0, 1.30), "crouch",    0.642),
    ("bail: top shelf (gaze up)",     sample(0.60, -30.0, 1.30), "bail",      1.600),
    ("recenter: miss (hit=False)",    sample(10.0,  20.0, 1.30, hit=False), "recenter", None),
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

    # pure-trig sanity (constant-independent): a level gaze puts the target at eye height, straight ahead
    lvl = plan_reach(sample(0.50, 0.0, 1.30), REF_ENVELOPE)
    assert approx(lvl["target_height"], 1.30), "level gaze: target_height must equal camera_height"
    assert approx(lvl["horizontal_gap"], 0.50), "level gaze: horizontal_gap must equal distance"

    # move_steps must equal round((gap - standoff)/move_unit): gap 0.752, standoff 0.35, unit 0.10 -> 4
    mv = plan_reach(sample(0.80, 20.0, 1.30), REF_ENVELOPE)
    want_steps = round((mv["horizontal_gap"] - REF_ENVELOPE["standoff"]) / REF_ENVELOPE["move_unit"])
    assert mv["move_steps"] == want_steps == 4, f"move_steps {mv['move_steps']} != {want_steps}"

    # unavailable: a pre-Phase-D sample (no pose) must NOT crash and must ask the caller to fall back
    un = plan_reach({"distance": 0.6, "hit": True, "min_range": 0.05, "max_range": 10.0}, REF_ENVELOPE)
    assert un["verdict"] == "unavailable", f"missing pose should be 'unavailable', got {un['verdict']}"

    print(f"\n{'ALL PASS' if fails == 0 else str(fails) + ' FAILED'}  "
          f"(REF_ENVELOPE frozen: standoff={REF_ENVELOPE['standoff']} r_eff={REF_ENVELOPE['r_eff']} "
          f"hand_drop={REF_ENVELOPE['hand_drop']})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
