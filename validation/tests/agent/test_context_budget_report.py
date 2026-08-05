"""Offline unit tests for tools/context_budget_report.py - the context-window budget analysis.

The report exists to make an ablation decision on measured numbers, so its arithmetic has to be
checkable independently of the corpus it happens to read. These drive the estimators against
SYNTHETIC inputs whose right answer is known by construction: a token series generated from chosen
coefficients must fit back to those coefficients, a retention policy applied to entries with known
overlap must report the coverage that overlap implies, and the per-arm integrals must equal the
closed forms they claim to be.

Also pins the step-dump section parser against a real dump's shape. That parser failed silently
once - an all-caps character class did not match "MODE ROUTER (semantic)", so every recall-derived
statistic read zero and the retention replay ran on empty input without raising.

    uv run pytest validation/tests/agent/test_context_budget_report.py   # or: pytest ...
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "agent")  # agent/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tools import context_budget_report as cbr


# --- the growth model -------------------------------------------------------------------------

def _attempt(steps_per_leg, tokens_in):
    a = cbr.Attempt(path="", battery="b", prompt_id="p",
                    summary={"tokens": {"tokens_in": tokens_in, "calls": 1}})
    a.steps_per_leg = list(steps_per_leg)
    return a


def test_fit_recovers_known_coefficients():
    """A series built from A=30000, B=2000 must fit back to exactly those, with R^2 = 1."""
    flat, growth = 30000.0, 2000.0
    attempts = []
    for legs in ([5], [12], [30], [7, 7], [40, 3], [60], [2, 9, 21]):
        depth = sum(n * (n + 1) / 2 for n in legs)
        attempts.append(_attempt(legs, flat * sum(legs) + growth * depth))

    fit = cbr.fit_growth(attempts)
    assert abs(fit["flat_per_step"] - flat) < 1e-6, fit
    assert abs(fit["growth_per_depth"] - growth) < 1e-6, fit
    assert fit["r2"] > 0.999999, fit
    assert fit["n"] == len(attempts)


def test_fit_needs_enough_rows():
    assert cbr.fit_growth([_attempt([3], 1000)]) == {}


def test_depth_is_per_leg_not_per_run():
    """History resets at each leg boundary (run_leg calls reset_history), so two 10-step legs must
    cost far less accumulated depth than one 20-step leg. If this ever inverts, the model is
    charging cross-leg growth that the code does not actually incur."""
    assert _attempt([10, 10], 0).depth == 110
    assert _attempt([20], 0).depth == 210


# --- section parsing --------------------------------------------------------------------------

_DUMP = """=== STEP 5 | mode=perception | halt=False ===

--- NAV NOTE ---
## MOVED 5 STEP(S) FORWARD

--- MODE ROUTER (semantic) ---
{
  "new_semantic_memory": "Pepero is on the middle shelf at Checkpoint 31.",
  "recall": "Navigate to Checkpoint 31 and grab the Pepero.",
  "mode": "navigation"
}

--- VLM ACTOR OUTPUT ---
```json
{'reasoning': 'centering', 'actions': ['center_object_on_screen'], 'times': [1]}
```

--- EPISODIC REFLECTION ---
```json
{'dense_summary': 'walked to the shelf', 'what_worked': 'centering', 'what_to_avoid': 'panning'}
```
"""


def test_sections_capture_the_mixed_case_router_header():
    sections = cbr._sections(_DUMP)
    assert "MODE ROUTER (semantic)" in sections, sorted(sections)
    assert {"NAV NOTE", "VLM ACTOR OUTPUT", "EPISODIC REFLECTION"} <= set(sections)


def test_router_json_parses_out_of_a_dump():
    router = cbr._loads_loose(cbr._sections(_DUMP)["MODE ROUTER (semantic)"])
    assert router["mode"] == "navigation"
    assert router["recall"].startswith("Navigate to Checkpoint 31")


def test_loads_loose_strips_a_code_fence():
    assert cbr._loads_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert cbr._loads_loose('prose {"a": 2} trailing') == {"a": 2}
    assert cbr._loads_loose("not json at all") is None


# --- retention ---------------------------------------------------------------------------------

def test_dedupe_drops_restatements_and_keeps_new_facts():
    entries = [
        "The Hello Panda boxes are on the middle shelf, left side.",
        "The Hello Panda boxes are on the middle shelf, left side of the fixture.",
        "The checkout counter is at Checkpoint 54.",
    ]
    kept = cbr._dedupe(entries, threshold=0.80)
    assert len(kept) == 2, kept
    assert kept[0] == entries[0] and kept[1] == entries[2]


def test_dedupe_lookback_window_is_bounded():
    """A restatement is only compared against the last `window` survivors, so an old duplicate that
    has scrolled out of the window is kept. That is deliberate - it bounds the cost - and this
    pins it so the behaviour is not mistaken for a bug later."""
    entries = ["fact A"] + [f"fact {i}" for i in range(10)] + ["fact A"]
    assert cbr._dedupe(entries, threshold=0.80, window=2)[-1] == "fact A"


def test_retention_replay_reports_full_coverage_when_nothing_is_lost():
    """Recall drawn entirely from the most recent entry must survive any last-K policy intact."""
    leg = [
        cbr.StepDump(router={"new_semantic_memory": "pepero sits on the middle shelf",
                             "recall": ""}, actor_chars=0, episodic_chars=0),
        cbr.StepDump(router={"new_semantic_memory": "counter stands at checkpoint fifty",
                             "recall": "pepero sits on the middle shelf"},
                     actor_chars=0, episodic_chars=0),
    ]
    out = cbr.retention_replay([leg], threshold=0.80, keep_last=12)
    assert out["mean"] == 1.0, out


def test_retention_replay_detects_content_lost_to_truncation():
    """Recall that draws on an entry evicted by a tight keep_last must score below 1.0 - otherwise
    the coverage metric cannot distinguish a safe policy from a lossy one."""
    # The distractors must be distinct from EACH OTHER as well as from the target: near-identical
    # distractors get deduped down to one survivor, the cap then never binds, and the test would
    # pass for the wrong reason.
    distractors = [
        "the juice fridge stands beside the entrance doors",
        "canned goods fill three bays of aisle four",
        "cleaning supplies occupy the rear wall bays",
        "bread racks block half of the western walkway",
        "frozen cabinets hum along the northern corridor",
        "the pharmacy counter closes off aisle seven",
    ]
    leg = [cbr.StepDump(router={"new_semantic_memory": "pepero occupies the middle shelf"},
                        actor_chars=0, episodic_chars=0)]
    leg += [cbr.StepDump(router={"new_semantic_memory": text},
                         actor_chars=0, episodic_chars=0) for text in distractors]
    leg.append(cbr.StepDump(router={"new_semantic_memory": "",
                                    "recall": "pepero occupies the middle shelf"},
                            actor_chars=0, episodic_chars=0))
    assert cbr.retention_replay([leg], threshold=0.80, keep_last=2)["mean"] < 1.0


# --- per-arm sizing ----------------------------------------------------------------------------

def _sizes(**means):
    return {k: {"n": 1, "mean": v, "median": v, "max": v} for k, v in means.items()}


def test_arm_savings_match_their_closed_forms():
    """The table's whole claim is that some components are flat and others integrate. Pin both:
    a findings cut must scale with N, an episodic-history cut with N(N+1)/2."""
    fit = {"flat_per_step": 30000.0, "growth_per_depth": 2000.0, "r2": 1.0, "n": 9}
    sizes = _sizes(**{
        "semantic entry (appended per step)": 240.0,      # 60 tok
        "episodic reflection (per step)": 1600.0,         # 400 tok
        "findings summary (per leg)": 3200.0,             # 800 tok
    })
    decomposition = {"residual": 1200.0}
    rows = cbr.arm_savings(fit, sizes, decomposition, {"kept_entries": 12},
                           chars_per_token=4.0, leg_lengths=(40,))
    by_arm = {r["arm"].split()[0]: r["by_leg_length"][40][0] for r in rows}

    assert abs(by_arm["A3"] - 800.0 * 40) < 1e-6            # flat: paid once per step
    assert abs(by_arm["A5"] - 400.0 * 40 * 41 / 2) < 1e-6   # growth: integrates
    assert abs(by_arm["A1"] - 60.0 * 40 * 41 / 2) < 1e-6
    assert abs(by_arm["A6"] - 1200.0 * 38 * 39 / 2) < 1e-6


def test_a2_saving_is_bounded_by_the_cap_and_never_exceeds_a1():
    """A2 keeps the last K entries, so it can only ever save what A1 saves, and its per-step saving
    must stop growing once the cap binds."""
    fit = {"flat_per_step": 30000.0, "growth_per_depth": 2000.0, "r2": 1.0, "n": 9}
    sizes = _sizes(**{"semantic entry (appended per step)": 240.0,
                      "episodic reflection (per step)": 1600.0,
                      "findings summary (per leg)": 3200.0})
    rows = cbr.arm_savings(fit, sizes, {"residual": 1200.0}, {"kept_entries": 12},
                           chars_per_token=4.0, leg_lengths=(10, 40, 100))
    by_arm = {r["arm"].split()[0]: r["by_leg_length"] for r in rows}
    for n in (10, 40, 100):
        assert by_arm["A2c"][n][0] <= by_arm["A1"][n][0] + 1e-9

    # Below the cap nothing is evicted, so A2 saves nothing at all.
    assert by_arm["A2c"][10][0] == 0.0


def test_chars_per_token_cancels_out_of_the_shares():
    """Absolute token counts scale with the conversion constant; the RANKING of arms must not.
    That is what licenses using this table to choose what to test."""
    fit = {"flat_per_step": 30000.0, "growth_per_depth": 2000.0, "r2": 1.0, "n": 9}
    sizes = _sizes(**{"semantic entry (appended per step)": 240.0,
                      "episodic reflection (per step)": 1600.0,
                      "findings summary (per leg)": 3200.0})
    order = []
    for cpt in (3.0, 4.0, 5.0):
        rows = cbr.arm_savings(fit, sizes, {"residual": 1200.0}, {"kept_entries": 12},
                               chars_per_token=cpt, leg_lengths=(40,))
        order.append([r["arm"] for r in sorted(rows, key=lambda r: -r["by_leg_length"][40][0])])
    assert order[0] == order[1] == order[2], order


# --- aggregates --------------------------------------------------------------------------------

def _outcome_attempt(prompt_id, tokens_in, success, manifest=None, steps=4, end_reason=None):
    a = _attempt([steps], tokens_in)
    a.prompt_id = prompt_id
    a.summary["success"] = success
    a.summary["legs"] = [{"end_reason": end_reason or ("halt_granted" if success else "time_cap")}]
    a.manifest = manifest if manifest is not None else {}
    return a


def test_outcomes_counts_wins_and_flags_the_confound_inputs():
    out = cbr.outcomes([_outcome_attempt("p1", 200_000, True),
                        _outcome_attempt("p2", 3_000_000, False)])
    row = out["per_battery"]["b"]
    assert (row["attempts"], row["predicate_wins"]) == (2, 1)
    assert out["end_reasons"] == {"halt_granted": 1, "time_cap": 1}
    assert out["median_tokens_success"] == 200_000
    assert out["median_tokens_failure"] == 3_000_000


def test_unmetered_attempts_count_as_outcomes_but_not_as_cost():
    """An attempt that crashed before the meter dumped is a real outcome and must not vanish from
    the success rate - but its tokens_in of 0 must not enter the cost split as a spurious win."""
    crashed = _outcome_attempt("p2", 0, False)
    crashed.summary["tokens"]["calls"] = 0
    out = cbr.outcomes([_outcome_attempt("p1", 200_000, True), crashed])
    assert out["per_battery"]["b"]["attempts"] == 2
    assert out["median_tokens_failure"] is None    # no metered failures, not a median of zeros


def test_excluded_verdicts_are_dropped_not_failed():
    """`already_successful` and `invalid` leave the totals entirely. Counting them as failures is
    exactly how this report first read 70% on a battery the dashboard scores at 96%."""
    if cbr._verdict_source() is None:
        return   # sari_bench not importable here; the report degrades to predicate-only and says so
    attempts = [
        _outcome_attempt("p1", 100, True, {"verified_success": True}),
        _outcome_attempt("p1", 100, False, {"verified_verdict": "already_successful"}),
        _outcome_attempt("p2", 100, True, {"verified_verdict": "invalid"}),
    ]
    row = cbr.outcomes(attempts)["per_battery"]["b"]
    assert row["attempts"] == 3
    assert row["included"] == 1 and row["attempt_wins"] == 1
    assert row["n_prompts"] == 1 and row["prompt_wins"] == 1


def test_reviewer_verdict_overrides_an_optimistic_predicate():
    """The predicate can grant on state it cannot ground, so a reviewer's `fail` must win."""
    if cbr._verdict_source() is None:
        return
    attempts = [_outcome_attempt("p1", 100, True, {"verified_success": False})]
    row = cbr.outcomes(attempts)["per_battery"]["b"]
    assert row["predicate_wins"] == 1 and row["attempt_wins"] == 0


def test_per_prompt_saturates_above_per_attempt():
    """One win out of five tries scores 20% per-attempt and 100% per-prompt. That gap is why the
    dashboard headline is the wrong metric to power an ablation on."""
    if cbr._verdict_source() is None:
        return
    attempts = [_outcome_attempt("p1", 100, i == 0) for i in range(5)]
    row = cbr.outcomes(attempts)["per_battery"]["b"]
    assert row["attempt_wins"] / row["included"] == 0.2
    assert row["prompt_wins"] / row["n_prompts"] == 1.0


def test_role_attribution_is_empty_without_metered_runs():
    """Runs predating per-role accounting must report zero metered attempts rather than inventing
    an attribution - the derived decomposition is only trustworthy while it says it is derived."""
    out = cbr.role_attribution([_attempt([3], 1000)])
    assert out["metered_attempts"] == 0 and out["by_role"] == {}


def test_role_attribution_sums_metered_runs():
    a = _attempt([3], 1000)
    a.summary["tokens"]["by_role"] = {"actor": {"tokens_in": 700, "calls": 3},
                                      "semantic": {"tokens_in": 300, "calls": 3}}
    b = _attempt([3], 1000)
    b.summary["tokens"]["by_role"] = {"actor": {"tokens_in": 100, "calls": 1}}
    out = cbr.role_attribution([a, b])
    assert out["metered_attempts"] == 2
    assert out["by_role"]["actor"] == {"tokens_in": 800, "calls": 4}


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL {name}: {exc}")
    print(f"\n{'FAILED' if failures else 'all passed'} ({failures} failure(s))")
    raise SystemExit(1 if failures else 0)
