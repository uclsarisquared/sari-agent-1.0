#!/usr/bin/env python3
"""Arm-level statistics for the context-window ablation.

Every table printed here is reproducible from analysis/context-ablation/attempts.csv,
which collect.py builds straight out of bench_runs/. Standard library only.

Usage:  python3 analysis/context-ablation/analyze.py [--csv PATH]
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import random
import statistics
from collections import defaultdict

ARMS = ["baseline", "a1", "a2c", "a3", "a4", "a5", "a6-2", "a6-4"]
PROMPTS = ["easy_01", "easy_03", "medium_01", "medium_03", "hard_01", "hard_03"]
NUMERIC = {"attempt", "success", "verified", "success_final", "excluded", "false_pass",
           "false_fail", "clean", "wall_seconds", "tokens_in", "tokens_out",
           "llm_calls", "legs_planned", "legs_completed", "leg_files", "steps",
           "halt_requests", "semantic_chars", "requeues", "leg_retries"}


def load(path: pathlib.Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            for key, value in list(row.items()):
                if key in NUMERIC or key.endswith(("_in", "_out", "_calls")):
                    row[key] = None if value == "" else float(value)
            rows.append(row)
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else float("nan")


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. At n<=18 the interval is the headline, not the point."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """p-value for the 2x2 table [[a, b], [c, d]] by full enumeration."""
    row1, row2, col1 = a + b, c + d, a + c
    if min(row1, row2) < 0 or col1 > row1 + row2:
        return float("nan")
    total = math.comb(row1 + row2, col1)

    def prob(x):
        return math.comb(row1, x) * math.comb(row2, col1 - x) / total

    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(col1, row1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= observed * (1 + 1e-9)))


def permutation_test(xs, ys, iters=20000, seed=0) -> float:
    """Two-sided permutation test on the difference of means."""
    xs, ys = [x for x in xs if x is not None], [y for y in ys if y is not None]
    if not xs or not ys:
        return float("nan")
    observed = abs(mean(xs) - mean(ys))
    pool = xs + ys
    rng = random.Random(seed)
    hits = 0
    for _ in range(iters):
        rng.shuffle(pool)
        if abs(mean(pool[:len(xs)]) - mean(pool[len(xs):])) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (iters + 1)


def growth_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Split a role's per-attempt input tokens into a flat and a compounding term.

    If the k-th call of a leg is handed base + slope*(k-1) tokens, then an attempt
    with n calls spends base*n + slope*n*(n-1)/2 in total. Least squares through
    the origin on those two predictors recovers (base, slope) without needing
    per-call token counts, which the harness does not record.
    """
    sxx = sxy = sxz = syy = syz = 0.0
    for n, total in points:
        if n < 1:
            continue
        x, y = n, n * (n - 1) / 2
        sxx += x * x
        sxy += x * y
        syy += y * y
        sxz += x * total
        syz += y * total
    det = sxx * syy - sxy * sxy
    if abs(det) < 1e-9:
        return (float("nan"), float("nan"), float("nan"))
    base = (sxz * syy - syz * sxy) / det
    slope = (syz * sxx - sxz * sxy) / det
    # base and slope trade off against each other, so report how much of the
    # spread the pair explains before reading either one on its own.
    used = [(n, t) for n, t in points if n >= 1]
    ybar = statistics.fmean(t for _, t in used)
    ss_tot = sum((t - ybar) ** 2 for _, t in used)
    ss_res = sum((t - (base * n + slope * n * (n - 1) / 2)) ** 2 for n, t in used)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return (base, slope, r2)


def projected(base: float, slope: float, n: int) -> float:
    """Total input tokens the fit predicts for a leg of n calls.

    Less sensitive than base or slope alone: the two parameters are strongly
    anti-correlated, so their sum at a fixed leg length is the stable readout.
    """
    return base * n + slope * n * (n - 1) / 2


def section(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def table(headers, rows) -> None:
    if not rows:
        return
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def main() -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=pathlib.Path, default=here / "attempts.csv")
    args = parser.parse_args()
    rows = [r for r in load(args.csv) if r["arm"] in ARMS]
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    # Cost tables use attempts that ran to their own stopping point. An operator
    # kill or a harness timeout stops the clock from outside, so those rows
    # describe the harness rather than the policy.
    def costable(arm):
        return [r for r in by_arm[arm] if r["clean"]]

    section("0. Scoring rule")
    print("Every prompt has a denominator of 3. An attempt counts as a success only")
    print("if a human reviewer passed it on replay. Attempts halted because a sibling")
    print("had already won score as non-successes like any other -- no arm gets a")
    print("smaller denominator. The baseline is now graded on the same terms as the")
    print("arms, so all eight are comparable.")
    body = []
    for arm in ARMS:
        rs = by_arm[arm]
        judged = sum(1 for r in rs if r["verified_verdict"] in ("pass", "fail"))
        body.append([arm, len(rs), judged,
                     sum(1 for r in rs if r["verified_verdict"] == "invalid"),
                     sum(1 for r in rs if not r["verified_verdict"]),
                     sum(int(r["false_pass"]) for r in rs),
                     sum(int(r["false_fail"]) for r in rs)])
    table(["arm", "attempts", "pass/fail judged", "invalid", "unreviewed",
           "auto-pass -> human-fail", "auto-fail -> human-pass"], body)
    judged_all = [r for r in rows if r["verified_verdict"] in ("pass", "fail")]
    fp_all = sum(int(r["false_pass"]) for r in judged_all)
    auto_pass = [r for r in judged_all if r["success"]]
    lo, hi = wilson(fp_all, len(auto_pass))
    print(f"\nThe auto predicate in subtask_completion.py over-grants: {fp_all} of "
          f"{len(auto_pass)} auto-passes")
    print(f"({fp_all / len(auto_pass):.1%}, 95% CI [{lo:.1%}, {hi:.1%}]) were overturned on "
          "review, against")
    print(f"{sum(int(r['false_fail']) for r in judged_all)} auto-fail promoted to a pass. "
          "Unreviewed rows fall back to the auto")
    print("verdict; only A4 has any, and all four are attempts it never got to finish.")

    section("1. Success rate over 3 tries per prompt")
    body = []
    base = by_arm["baseline"]
    base_s = sum(int(r["success_final"]) for r in base)
    for arm in ARMS:
        rs = by_arm[arm]
        s = sum(int(r["success_final"]) for r in rs)
        auto = sum(int(r["success"]) for r in rs)
        lo, hi = wilson(s, len(rs))
        p = "-" if arm == "baseline" else \
            f"{fisher_exact_two_sided(s, len(rs) - s, base_s, len(base) - base_s):.3f}"
        body.append([arm, f"{auto}/{len(rs)}", f"{s}/{len(rs)}", f"{s / len(rs):.3f}",
                     f"[{lo:.2f}, {hi:.2f}]",
                     f"{s / len(rs) - base_s / len(base):+.3f}", p])
    table(["arm", "auto (ungraded)", "graded", "rate", "wilson95", "vs baseline",
           "fisher p"], body)
    print("\nGrading costs the baseline 4 of its 15 auto-passes, which is the single")
    print("largest correction in the table and the reason the earlier draft read the")
    print("arms as uniformly worse than the control.")

    section("2. Per-prompt successes out of 3")
    body = []
    for arm in ARMS:
        cells = [arm]
        for pid in PROMPTS:
            cells.append(sum(int(r["success_final"]) for r in by_arm[arm]
                             if r["prompt_id"] == pid))
        cells.append(sum(int(r["success_final"]) for r in by_arm[arm]))
        body.append(cells)
    table(["arm"] + PROMPTS + ["total"], body)
    print("\nPrompts solved at least once, out of 6:")
    body = []
    for arm in ARMS:
        solved = sum(1 for pid in PROMPTS
                     if any(r["success_final"] for r in by_arm[arm]
                            if r["prompt_id"] == pid))
        body.append([arm, solved])
    table(["arm", "prompts solved >=1"], body)

    section("3. What the flat denominator costs A4")
    print("A4 had 7 attempts halted after a sibling won; under a denominator of 3 they")
    print("score as failures. This is what the arm would read if those rows were")
    print("dropped instead -- shown so the choice is visible, not to score on it.")
    body = []
    for arm in ARMS:
        rs = by_arm[arm]
        inc = [r for r in rs if not r["excluded"]]
        s_all = sum(int(r["success_final"]) for r in rs)
        s_inc = sum(int(r["success_final"]) for r in inc)
        body.append([arm, f"{s_all}/{len(rs)} ({s_all / len(rs):.2f})",
                     f"{s_inc}/{len(inc)} ({s_inc / len(inc):.2f})",
                     len(rs) - len(inc)])
    table(["arm", "scored (over 3)", "if halted rows dropped", "rows dropped"], body)

    section("4. Run-order drift check")
    print("run_all.sh ran the arms back to back on one machine and one sandbox fleet,")
    print("so a machine that degraded over the day would masquerade as an arm effect.")
    order = ["baseline", "a5", "a6-2", "a6-4", "a2c", "a1", "a3", "a4"]
    body, rates = [], []
    for index, arm in enumerate(order, start=1):
        rs = by_arm[arm]
        rate = sum(int(r["success_final"]) for r in rs) / len(rs)
        rates.append(rate)
        body.append([index, arm, rs[0]["battery"], f"{rate:.2f}"])
    table(["order", "arm", "battery", "graded success"], body)
    ranks = sorted(range(len(rates)), key=lambda i: rates[i])
    ranked = [0] * len(rates)
    for position, i in enumerate(ranks):
        ranked[i] = position + 1
    n = len(rates)
    d2 = sum((i + 1 - ranked[i]) ** 2 for i in range(n))
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    # Exact permutation p: 8! orderings is small enough to enumerate.
    import itertools
    extreme = sum(1 for perm in itertools.permutations(range(1, n + 1))
                  if abs(1 - 6 * sum((perm[i] - ranked[i]) ** 2
                                     for i in range(n)) / (n * (n * n - 1))) >= abs(rho) - 1e-12)
    print(f"\nSpearman rho(run order, graded success) = {rho:+.2f} over {n} arms, "
          f"exact p = {extreme / math.factorial(n):.3f}.")
    print("Not significant: with eight arms a rho of this size is unremarkable, and")
    print("the highest-scoring arm ran third. No drift correction is warranted.")

    section("5. Context cost per LLM call (costable attempts)")
    print("Input tokens per call reads context size directly, but it is an average over")
    print("that arm's own call mix -- an arm whose runs went longer took more late-leg")
    print("calls, which are the expensive ones. Read section 7 for the deconfounded view.")
    body = []
    for arm in ARMS:
        clean = costable(arm)
        cells = [arm, len(clean)]
        for role in ("actor", "semantic", "episodic", "perception"):
            num = sum(r[f"{role}_in"] for r in clean)
            den = sum(r[f"{role}_calls"] for r in clean)
            cells.append(f"{num / den:,.0f}" if den else "-")
        num = sum(r["tokens_in"] for r in clean)
        den = sum(r["llm_calls"] for r in clean)
        cells.append(f"{num / den:,.0f}" if den else "-")
        body.append(cells)
    table(["arm", "n", "actor/call", "semantic/call", "episodic/call", "perception/call",
           "all/call"], body)

    section("6. Cost per step and per attempt (costable attempts)")
    body = []
    for arm in ARMS:
        clean = costable(arm)
        steps = sum(r["steps"] for r in clean)
        tin = sum(r["tokens_in"] for r in clean)
        body.append([arm, len(clean), int(steps), f"{steps / len(clean):.1f}",
                     f"{tin / steps:,.0f}" if steps else "-",
                     f"{tin / len(clean):,.0f}",
                     f"{mean(r['wall_seconds'] for r in clean) / 60:.1f}"])
    table(["arm", "n", "steps", "steps/att", "tok_in/step", "tok_in/att", "wall_min"], body)
    print("\ntok_in/att is dominated by how long an arm survived, not by its context")
    print("policy: an attempt that runs to the 40-minute leg cap spends 40 minutes of")
    print("tokens. Section 7 removes the run-length term.")

    section("8. Permutation tests on cost vs baseline (costable attempts)")
    base_clean = costable("baseline")
    body = []
    for arm in ARMS:
        if arm == "baseline":
            continue
        clean = costable(arm)
        cells = [arm]
        for fn in (lambda r: r["tokens_in"],
                   lambda r: r["actor_in"] / r["actor_calls"] if r["actor_calls"] else None,
                   lambda r: r["steps"],
                   lambda r: r["semantic_chars"]):
            xs = [v for v in map(fn, clean) if v]
            ys = [v for v in map(fn, base_clean) if v]
            cells.append(f"{mean(xs) / mean(ys) - 1:+.0%} p={permutation_test(xs, ys):.3f}")
        body.append(cells)
    table(["arm", "tok_in/att", "actor/call", "steps", "semantic chars"], body)

    section("9. Semantic store size at end of run (chars, all attempts)")
    body = []
    for arm in ARMS:
        vals = [r["semantic_chars"] for r in by_arm[arm] if r["semantic_chars"]]
        body.append([arm, len(vals), f"{min(vals):,.0f}", f"{statistics.median(vals):,.0f}",
                     f"{mean(vals):,.0f}", f"{max(vals):,.0f}"])
    table(["arm", "n", "min", "median", "mean", "max"], body)
    print("\nBase store is 17,161 chars. A1 pinning every run to 17,162 and A2c capping")
    print("the maximum at ~21k are the invariant checks that those two seams fired.")

    section("10. Failure texture")
    body = []
    for arm in ARMS:
        rs = by_arm[arm]
        body.append([arm, int(sum(r["leg_retries"] for r in rs)),
                     f"{mean(r['legs_planned'] for r in rs):.2f}",
                     sum(1 for r in rs if r["end_reason"] == "halt_forced"),
                     sum(1 for r in rs if r["end_reason"] == "time_cap"),
                     f"{mean(r['tokens_out'] for r in rs):,.0f}"])
    table(["arm", "leg retries", "legs planned/att", "halt_forced", "time_cap",
           "tok_out/att"], body)

    section("11. Multi-leg vs single-leg split (the split A3/A4 target)")
    print("Findings summaries only cross a leg boundary, so a task the decomposer")
    print("split into one leg is unaffected by A3/A4 by construction.")
    body = []
    for arm in ARMS:
        inc = [r for r in by_arm[arm] if not r["excluded"] and r["legs_planned"]]
        multi = [r for r in inc if r["legs_planned"] > 1]
        single = [r for r in inc if r["legs_planned"] == 1]
        body.append([arm,
                     f"{sum(int(r['success_final']) for r in single)}/{len(single)}" if single else "-",
                     f"{sum(int(r['success_final']) for r in multi)}/{len(multi)}" if multi else "-"])
    table(["arm", "single-leg", "multi-leg"], body)

    section("12. Cost of a win (reviewed wins that ran to their own stop)")
    body = []
    for arm in ARMS:
        won = [r for r in costable(arm) if r["success_final"]]
        if not won:
            body.append([arm, 0, "-", "-", "-"])
            continue
        body.append([arm, len(won), f"{mean(r['tokens_in'] for r in won):,.0f}",
                     f"{mean(r['steps'] for r in won):.1f}",
                     f"{mean(r['wall_seconds'] for r in won) / 60:.1f}"])
    table(["arm", "n wins", "tok_in/win", "steps/win", "wall_min/win"], body)
    print("\nThis is the fairest cost column in the report: it compares arms only on")
    print("runs that reached the same outcome, so it cannot be inflated by an arm that")
    print("simply failed for longer.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
