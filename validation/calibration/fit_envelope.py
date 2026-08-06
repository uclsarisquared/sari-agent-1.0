"""fit_envelope.py - derive manipulation.REACH_ENVELOPE (max_reach) from a reach_probe.py CSV.

MEASURED MODEL (2026-07-22): the hand grabs along the gaze, so reachability is decided by the SLANT
distance alone - reachable iff distance <= max_reach. This reads the calibration CSV, finds the
distance boundary that separates grabs from misses, and prints max_reach (boundary minus a safety
margin) to paste into manipulation.REACH_ENVELOPE. It also reports how CLEANLY distance separates the
two: a big overlap (misses closer than grabs) means the single-distance model isn't holding and you
should inspect the crossing rows before trusting it.

    python fit_envelope.py [path/to/envelope.csv]
"""
import csv
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "agent")  # agent/
DEFAULT_CSV = os.path.join(
    os.path.dirname(_ROOT), "validation", "artifacts", "calibration", "reach", "envelope.csv")
MARGIN = 0.04   # set max_reach this far BELOW the boundary so you grab INSIDE the limit, not at its edge


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def _accuracy(rows, thresh):
    return sum(1 for r in rows if r["_d"] is not None and (r["_d"] <= thresh) == r["_g"])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(path):
        print(f"No CSV at {path}.\nRun reach_probe.py and log some grabs (`g <label>`) first.")
        return 1
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_g"] = str(r.get("grabbed", "")).strip().lower() in ("true", "1", "yes")
        r["_d"] = _f(r, "distance")
    usable = [r for r in rows if r["_d"] is not None]
    grab = [r["_d"] for r in usable if r["_g"]]
    miss = [r["_d"] for r in usable if not r["_g"]]
    if not grab:
        print(f"{len(rows)} row(s) but no grabbed==True with a distance. Log successful grabs first.")
        return 1

    max_grab = max(grab)
    min_miss = min(miss) if miss else None
    print(f"\n=== max_reach fit from {os.path.basename(path)} "
          f"({len(grab)} grabbed / {len(usable)} usable rows) ===\n")
    print(f"  grabbed slant distance: {min(grab):.3f} .. {max_grab:.3f} m")
    if miss:
        print(f"  missed  slant distance: {min_miss:.3f} .. {max(miss):.3f} m")

    if min_miss is None:
        boundary = max_grab
        print(f"  no misses logged - {boundary:.3f} m is only a LOWER bound; grab farther until some "
              f"fail to find the true limit.")
    elif max_grab < min_miss:
        boundary = (max_grab + min_miss) / 2.0
        print(f"  clean split: every grab is closer than every miss. boundary ~= {boundary:.3f} m "
              f"({_accuracy(usable, boundary)}/{len(usable)} rows predicted by distance alone)")
    else:
        # overlap: pick the threshold that classifies the most rows correctly, and flag it
        best_t = max(sorted(set(grab + miss)), key=lambda t: _accuracy(usable, t))
        boundary = best_t
        print(f"  OVERLAP of {max_grab - min_miss:.3f} m: some misses are CLOSER than some grabs, so "
              f"distance alone doesn't fully decide it here. Best threshold {boundary:.3f} m gets "
              f"{_accuracy(usable, boundary)}/{len(usable)} rows - inspect the crossing rows before trusting.")

    max_reach = round(boundary - MARGIN, 2)
    print(f"\n  paste into manipulation.REACH_ENVELOPE (keep move_unit/move_cap):")
    print(f'    "max_reach": {max_reach:.2f},   # boundary {boundary:.3f} m - {MARGIN} margin '
          f'(agrees with {_accuracy(usable, max_reach)}/{len(usable)} grabs; conservative near the edge)')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
