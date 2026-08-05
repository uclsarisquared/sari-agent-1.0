"""fit_place_envelope.py - derive the Phase 6.2 PLACE envelope from a place_probe.py CSV.

The place sibling of fit_envelope.py, with one difference that CHANGES the fit. A grab is a single
UPPER bound (reachable iff slant distance <= max_reach). A release onto a horizontal surface can have
BOTH a far limit (too far -> the item falls short of the counter) AND a near limit (too close -> the
hand can't clear the counter's front edge). So this reports a RANGE, and it FLAGS a near bound when the
data shows one - because plan_place, as designed, has a "too far -> move closer" verdict but NO
"too close" verdict. A real near bound is therefore a DESIGN decision (add a verdict, or make the
approach stop short of it), not just a constant to paste.

It splits the fit two ways, each a measured answer the phase6 plan asks for:
  - by POSTURE: if standing and crouched land at different distances, release HEIGHT matters - the
    signal that a distinct PLACE pose earns its existence over reusing GRAB.
  - by RELEASE_MODE: the headline uses the 'extend' rows (what place_held_item will do); 'rest' rows
    (release without extending) are reported alongside for the A/B.

Rows where the LiDAR did not hit a surface (hit=False) are dropped from the fit - the release aimed at
nothing (floor / past the counter), so their slant distance is max_range, not an envelope point.

    python fit_place_envelope.py [path/to/place_envelope.csv]
"""
import csv
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "agent")  # agent/
DEFAULT_CSV = os.path.join(
    os.path.dirname(_ROOT), "validation", "artifacts", "calibration", "place", "place_envelope.csv")
MARGIN = 0.04   # place INSIDE each bound, not at its edge (max below the far bound, min above the near)


def _truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def _accuracy(rows, lo, hi):
    """How many rows a `lo <= distance <= hi` rule classifies correctly (lo None => no near bound)."""
    n = 0
    for r in rows:
        if r["_d"] is None:
            continue
        pred = (lo is None or r["_d"] >= lo) and r["_d"] <= hi
        n += int(pred == r["_land"])
    return n


def _fit(rows, tag):
    """Fit + print the envelope for one subset. Returns (place_min, place_max) in metres, either of
    which may be None (no bound observed on that side / not enough data)."""
    usable = [r for r in rows if r["_d"] is not None]
    land = sorted(r["_d"] for r in usable if r["_land"])
    miss = sorted(r["_d"] for r in usable if not r["_land"])
    print(f"\n--- {tag}: {len(land)} landed / {len(usable)} usable ---")
    if not land:
        print("  no landed rows - nothing to bound. Place some items that LAND first.")
        return None, None
    lo_land, hi_land = land[0], land[-1]
    print(f"  landed slant distance: {lo_land:.3f} .. {hi_land:.3f} m")
    if miss:
        print(f"  missed slant distance: {miss[0]:.3f} .. {miss[-1]:.3f} m")

    near_miss = [d for d in miss if d < lo_land]      # closer than any landing -> a near bound
    far_miss = [d for d in miss if d > hi_land]       # farther than any landing -> a far bound
    mid_miss = [d for d in miss if lo_land <= d <= hi_land]   # inside the landed span -> overlap

    far_bound = (hi_land + min(far_miss)) / 2.0 if far_miss else None
    near_bound = (max(near_miss) + lo_land) / 2.0 if near_miss else None

    if far_bound is None:
        place_max = hi_land
        print(f"  no far miss logged - place_max {place_max:.3f} m is only a LOWER bound; back off "
              f"(s) until releases fall short to find the true far limit.")
    else:
        place_max = round(far_bound - MARGIN, 2)
        print(f"  far bound ~= {far_bound:.3f} m -> place_max {place_max:.3f} m ({far_bound:.3f} - {MARGIN})")

    place_min = None
    if near_bound is not None:
        place_min = round(near_bound + MARGIN, 2)
        print(f"  ** NEAR bound ~= {near_bound:.3f} m -> place_min {place_min:.3f} m: releases CLOSER "
              f"than this missed. plan_place has NO 'too close' verdict by design - either add one, or "
              f"make the counter approach STOP at >= place_min. This is a design call, flag it. **")

    if mid_miss:
        print(f"  ** OVERLAP: {len(mid_miss)} miss(es) INSIDE the landed span "
              f"({min(mid_miss):.3f}..{max(mid_miss):.3f} m) - distance alone does not decide landing "
              f"here. Inspect those rows (posture? release_mode? centring?) before trusting a bound. **")

    acc = _accuracy(usable, place_min, place_max)
    lo_str = "-inf" if place_min is None else f"{place_min:.2f}"
    print(f"  rule: place iff {lo_str} <= slant <= {place_max:.2f} m  "
          f"({acc}/{len(usable)} rows classified correctly)")
    return place_min, place_max


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(path):
        print(f"No CSV at {path}.\nRun place_probe.py and log some places (`p <label>`) first.")
        return 1
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_d"] = _f(r, "slant_distance")
        r["_land"] = _truthy(r.get("landed"))
        r["_hit"] = _truthy(r.get("hit"))
        r["_posture"] = (r.get("posture") or "").strip().lower()
        r["_mode"] = (r.get("release_mode") or "").strip().lower()

    dropped = [r for r in rows if not r["_hit"]]
    hit_rows = [r for r in rows if r["_hit"]]
    print(f"=== place envelope fit from {os.path.basename(path)} "
          f"({len(hit_rows)} surface-hit rows / {len(rows)} total"
          + (f", {len(dropped)} dropped for hit=False" if dropped else "") + ") ===")

    extend = [r for r in hit_rows if r["_mode"] == "extend"]
    rest = [r for r in hit_rows if r["_mode"] == "rest"]
    headline = extend or hit_rows          # fall back to all rows if nobody tagged release_mode
    if not extend:
        print("  (no release_mode='extend' rows - fitting all surface-hit rows as the headline)")

    pmin, pmax = _fit(headline, "HEADLINE (extend)")

    # Posture split - does release height matter?
    standing = [r for r in headline if r["_posture"] == "standing"]
    crouched = [r for r in headline if r["_posture"] == "crouched"]
    if standing and crouched:
        _, smax = _fit(standing, "standing")
        _, cmax = _fit(crouched, "crouched")
        if smax is not None and cmax is not None and abs(smax - cmax) >= 0.10:
            print(f"\n  ** HEIGHT MATTERS: standing place_max {smax:.2f} m vs crouched {cmax:.2f} m "
                  f"(>= 0.10 m apart). A distinct PLACE pose is justified - do NOT reuse GRAB blindly. **")
        else:
            print(f"\n  posture split: standing/crouched far bounds agree within 0.10 m - height does "
                  f"not decide landing; reusing the GRAB pose for placing is defensible.")
    if rest:
        _fit(rest, "rest-release (A/B only, not the headline)")

    if pmax is not None:
        lo_str = "None" if pmin is None else f"{pmin:.2f}"
        print(f"\n  paste into manipulation.PLACE_ENVELOPE (keep move_unit/move_cap like REACH_ENVELOPE):")
        print(f'    "place_max": {pmax:.2f},   # place if slant distance <= this (m)')
        print(f'    "place_min": {lo_str},   # None = no near bound; else the too-close limit (design call above)')
        print(f'    "move_unit": 0.10, "move_cap": 10,')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
