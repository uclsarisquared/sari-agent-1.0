#!/usr/bin/env python3
"""Per-leg context growth model for the context-window ablation.

The actor's message history is rebuilt at each leg boundary, so growth has to be
fitted within a leg. If the k-th call of a leg is handed base + slope*(k-1)
input tokens, a leg of n calls spends

    tokens_in(n) = base*n + slope*n*(n-1)/2

Fitting that through the origin over an arm's legs recovers (base, slope) without
per-call token counts, which the harness does not record. `base` is the flat cost
an arm pays every call; `slope` is the compounding term each seam claims to cut.

Usage:  python3 analysis/context-ablation/growth.py [--csv PATH]
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import random
import statistics

from collections import defaultdict

ARMS = ["baseline", "a1", "a2c", "a3", "a4", "a5", "a6-2", "a6-4"]
# Legs shorter than this cannot separate base from slope; a 1-call leg puts zero
# weight on the quadratic term and just inflates the base estimate.
MIN_CALLS = 3


def fit(points):
    """Least squares for tokens = base*n + slope*n(n-1)/2 through the origin."""
    sxx = sxy = syy = sxz = syz = 0.0
    for n, total in points:
        x, y = n, n * (n - 1) / 2
        sxx += x * x
        sxy += x * y
        syy += y * y
        sxz += x * total
        syz += y * total
    det = sxx * syy - sxy * sxy
    if abs(det) < 1e-9 or len(points) < 3:
        return None
    base = (sxz * syy - syz * sxy) / det
    slope = (syz * sxx - sxz * sxy) / det
    ybar = statistics.fmean(t for _, t in points)
    ss_tot = sum((t - ybar) ** 2 for _, t in points)
    ss_res = sum((t - (base * n + slope * n * (n - 1) / 2)) ** 2 for n, t in points)
    return base, slope, (1 - ss_res / ss_tot if ss_tot else float("nan"))


def boot_ci(points, index, iters=2000, seed=0):
    """Percentile bootstrap over legs for one fitted parameter."""
    rng = random.Random(seed)
    draws = []
    for _ in range(iters):
        sample = [points[rng.randrange(len(points))] for _ in points]
        result = fit(sample)
        if result:
            draws.append(result[index])
    if len(draws) < iters // 2:
        return (float("nan"), float("nan"))
    draws.sort()
    return (draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))])


def table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=pathlib.Path, default=here / "legs.csv")
    parser.add_argument("--min-calls", type=int, default=MIN_CALLS)
    args = parser.parse_args()

    by_arm = defaultdict(list)
    with args.csv.open() as handle:
        for row in csv.DictReader(handle):
            if row["arm"] in ARMS:
                by_arm[row["arm"]].append(row)

    print(f"Per-leg growth fit, legs with >= {args.min_calls} calls of the role.\n")
    for role, label in (("actor", "A5 / A6 act here"),
                        ("semantic", "A1 / A2c act here"),
                        ("episodic", "A5 moves this indirectly")):
        print(f"\nrole = {role}   ({label})")
        ref = None
        body = []
        for arm in ARMS:
            points = [(int(r[f"{role}_calls"]), float(r[f"{role}_in"]))
                      for r in by_arm[arm]
                      if int(r[f"{role}_calls"] or 0) >= args.min_calls]
            result = fit(points)
            if not result:
                body.append([arm, len(points), "-", "-", "-", "-"])
                continue
            base, slope, r2 = result
            lo, hi = boot_ci(points, 1)
            if arm == "baseline":
                ref = slope
            body.append([arm, len(points), f"{base:,.0f}", f"{slope:,.0f}",
                         f"[{lo:,.0f}, {hi:,.0f}]", f"{r2:.2f}",
                         "-" if arm == "baseline" or ref in (None, 0)
                         else f"{slope / ref - 1:+.0%}"])
        table(["arm", "legs", "base tok/call", "slope tok/call", "slope 95% CI",
               "R2", "vs baseline"], body)

    print("\n\nPredicted actor input tokens at matched leg lengths")
    print("(the growth fit evaluated at a fixed n, which is what a step of a long")
    print("leg actually costs -- immune to arms whose legs ran longer)")
    body = []
    ref_fit = None
    for arm in ARMS:
        points = [(int(r["actor_calls"]), float(r["actor_in"]))
                  for r in by_arm[arm] if int(r["actor_calls"] or 0) >= args.min_calls]
        result = fit(points)
        if not result:
            continue
        base, slope, _ = result
        if arm == "baseline":
            ref_fit = (base, slope)
        cells = [arm]
        for n in (5, 10, 20, 40):
            value = base + slope * (n - 1)
            reference = ref_fit[0] + ref_fit[1] * (n - 1)
            cells.append(f"{value:,.0f}" + ("" if arm == "baseline"
                                            else f" ({value / reference - 1:+.0%})"))
        body.append(cells)
    table(["arm", "call 5", "call 10", "call 20", "call 40"], body)
    print("\nRead down a column: that is what one actor call costs at that point in a")
    print("leg. The rightmost column is where the compounding arms are supposed to pay")
    print("off, and it is the only place they clearly do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
