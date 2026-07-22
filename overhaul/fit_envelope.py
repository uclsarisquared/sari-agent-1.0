"""fit_envelope.py - derive manipulation.REACH_ENVELOPE from a reach_probe.py calibration CSV.

reach_probe.py logs one row per grab attempt to slamtest/output/reachtests/envelope.csv. This reads
that CSV and prints the measured envelope constants (hand_drop, r_eff, standoff) ready to paste into
manipulation.REACH_ENVELOPE, plus two sanity reports:
  - the POSTURE-INVARIANCE check: does the crouched hand keep the same drop below the camera as
    standing? (plan_reach assumes one hand_drop and lets camera_height carry the posture.)
  - the per-posture REACHABLE HEIGHT BAND, so you can see which of your shelf levels each posture
    actually covers - standing reaches the upper levels, crouch shifts the band down to the lower ones.

    python fit_envelope.py [path/to/envelope.csv]

Postures are split by MEASURED camera_height, NOT the CSV 'posture' label - if you drive/crouch with
the Unity built-in controller the label can be stale, but camera_height is read live at each grab.

The model (see manipulation.plan_reach): a grab lands iff the item height is within r_eff of
h_reach = camera_height - hand_drop, AND the horizontal gap is <= standoff. So for every GRABBED row,
delta = camera_height - target_height lies in [hand_drop - r_eff, hand_drop + r_eff]; we fit that band
to the grabbed rows and read standoff off the largest grabbed horizontal_gap.
"""
import csv
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_THIS, "slamtest", "output", "reachtests", "envelope.csv")


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_grabbed"] = str(r.get("grabbed", "")).strip().lower() in ("true", "1", "yes")
        ch, th = _f(r, "camera_height"), _f(r, "target_height")
        r["_ch"] = ch
        r["_th"] = th
        r["_delta"] = (ch - th) if (ch is not None and th is not None) else None
        r["_hgap"] = _f(r, "horizontal_gap")
    return rows


def band(deltas):
    """hand_drop (midpoint), r_eff (half-spread) of a set of grabbed deltas."""
    lo, hi = min(deltas), max(deltas)
    return (lo + hi) / 2.0, (hi - lo) / 2.0


def split_by_camera_height(grabbed):
    """Group grabbed rows into postures by the LARGEST gap in measured camera_height (standing sits
    ~2x higher than crouched). No clear gap -> a single posture cluster. Returns [(name, rows), ...]."""
    chs = sorted(r["_ch"] for r in grabbed if r["_ch"] is not None)
    if not chs:
        return []
    gap, gi = max(((chs[i + 1] - chs[i], i) for i in range(len(chs) - 1)), default=(0.0, -1))
    if gap > 0.15:                      # bimodal -> standing (high cam) vs crouched (low cam)
        split = (chs[gi] + chs[gi + 1]) / 2.0
        return [(f"standing (cam>={split:.2f})", [r for r in grabbed if r["_ch"] is not None and r["_ch"] >= split]),
                (f"crouched (cam<{split:.2f})",  [r for r in grabbed if r["_ch"] is not None and r["_ch"] < split])]
    return [("one posture cluster", [r for r in grabbed if r["_ch"] is not None])]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(path):
        print(f"No CSV at {path}.\nRun reach_probe.py and log some grabs (`g <label>`) first.")
        return 1
    rows = load(path)
    grabbed = [r for r in rows if r["_grabbed"] and r["_delta"] is not None]
    if not grabbed:
        print(f"{len(rows)} row(s) but none grabbed==True with usable pose. Log successful grabs first.")
        return 1

    deltas = [r["_delta"] for r in grabbed]
    hand_drop, r_eff = band(deltas)
    hgaps = sorted(r["_hgap"] for r in grabbed if r["_hgap"] is not None)
    standoff = hgaps[-1] if hgaps else None
    standoff_p90 = hgaps[int(0.9 * (len(hgaps) - 1))] if hgaps else None

    print(f"\n=== envelope fit from {os.path.basename(path)} "
          f"({len(grabbed)} grabbed / {len(rows)} rows) ===\n")

    # POSTURE-INVARIANCE + per-posture coverage, split by MEASURED camera_height (label-independent).
    drops = []
    for name, grp in split_by_camera_height(grabbed):
        ds = [r["_delta"] for r in grp]
        ths = [r["_th"] for r in grp if r["_th"] is not None]
        chg = [r["_ch"] for r in grp if r["_ch"] is not None]
        hd, re = band(ds)
        drops.append(hd)
        print(f"  {name}: {len(grp):2d} grabs | cam_h {min(chg):.2f}..{max(chg):.2f} | "
              f"hand_drop~{hd:+.2f} r_eff~{re:.2f} | reachable height band {min(ths):.2f}..{max(ths):.2f} m")
    if len(drops) == 2:
        d = abs(drops[0] - drops[1])
        verdict = "OK - pool into one hand_drop" if d <= 0.10 else \
                  "DIFFERS >0.10 m - crouch changes the arm drop; plan_reach may need a per-posture hand_drop"
        print(f"  posture-invariance: hand_drop differs by {d:.2f} m across postures -> {verdict}")
    print()

    # contradictions: a MISS whose delta is INSIDE the fitted vertical band, so height was not the
    # reason it missed - likely too far (check horizontal_gap vs standoff) or the band is too wide.
    inside = [r for r in rows if not r["_grabbed"] and r["_delta"] is not None
              and abs(r["_delta"] - hand_drop) <= r_eff]
    if inside:
        print(f"  note: {len(inside)} MISS row(s) fall inside the fitted vertical band - check their "
              f"horizontal_gap vs standoff, or tighten r_eff.\n")

    print("  paste into manipulation.REACH_ENVELOPE (keep reach_tol/move_unit/move_cap as-is):")
    print(f'    "hand_drop": {hand_drop:.2f},')
    print(f'    "r_eff":     {r_eff:.2f},')
    if standoff is not None:
        print(f'    "standoff":  {standoff:.2f},   # largest grabbed gap; conservative p90 = {standoff_p90:.2f}')
    else:
        print('    "standoff":  <none - no horizontal_gap logged>')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
