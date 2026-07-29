"""Where the agent's context window goes, measured off saved bench runs.

Reproduces the context-bloat analysis end to end and sizes each ablation arm, so the decision to
cut (or keep) a context component is made against numbers rather than intuition. Offline: reads
`bench_runs/` only - no sim, no sandbox lease, no model call.

WHAT IT MEASURES, AND WHY THE MODEL EXISTS
    The per-step prompt is not one blob: the actor, the semantic learner and the episodic learner
    each assemble their own, and only some parts grow with the step index. Total input tokens fit

        tokens_in  ~=  A * steps  +  B * sum_k(k)

    where `sum_k(k)` is the accumulated conversation depth over a leg (step 1 pays for 0 prior
    turns, step 30 pays for 29). A is the flat per-step cost - system prompts, the rendered store
    map, the findings context, this step's frames. B is the COMPOUNDING term, and it is the one
    that decides whether a 100-step leg is affordable. The fit runs over every attempt in the
    corpus and reports R^2, because the whole argument rests on B being real.

    Attribution of B to individual components is DERIVED, not metered: the runs in `bench_runs/`
    predate `token_meter`'s per-role accounting, so the script measures every text component it
    can read from the step dumps and attributes the residual to the retained screenshots. When a
    run DOES carry `by_role` (anything metered after roles landed), the script reports it and
    checks the model against it instead of guessing - see `role_attribution`.

CHARS-PER-TOKEN
    Text sizes are measured in characters and converted with CHARS_PER_TOKEN, a rough constant for
    English prose + JSON. It is a scaling factor on every text-derived number here, so it cancels
    out of the SHARES (which arm is worth testing) and only matters for the absolute token counts.
    `--chars-per-token` overrides it; `by_role` supersedes it entirely once available.

USAGE
    python tools/context_budget_report.py                  # summary, sampled deep analysis
    python tools/context_budget_report.py --deep           # full redundancy + retention replay (slow)
    python tools/context_budget_report.py --json out.json  # machine-readable, for diffing arms
"""

import argparse
import collections
import difflib
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass, field

OVERHAUL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if OVERHAUL_DIR not in sys.path:
    sys.path.insert(0, OVERHAUL_DIR)

REPO_ROOT = os.path.dirname(OVERHAUL_DIR)

# Rough English-prose + JSON ratio. See the module docstring: it cancels out of every share and
# only scales the absolute counts.
CHARS_PER_TOKEN = 4.0

# Leg lengths the per-arm table is reported at. 40 is around the longest leg the corpus actually
# contains; 100 is inside the configured --max-steps 150, i.e. the region not yet stressed.
REPORT_LEG_LENGTHS = (20, 40, 60, 100)

# Qwen-family vision encoders emit one token per 2x2-merged 14px patch, i.e. one per 28x28 pixel
# block. Used only as a cross-check on the residual the fit attributes to retained frames.
VISION_PIXELS_PER_TOKEN = 28

_STEP_EVENT = '"event": "step"'
_ENTRY_RE = re.compile(r"^@ (?:leg \d+ step|timestep) \d+: (.*)$", re.M)
# Section headers are upper-case EXCEPT for the parenthetical in "MODE ROUTER (semantic)" - an
# all-caps class silently matched every section but that one, which zeroed the recall stats and
# emptied the retention replay without erroring.
_SECTION_RE = re.compile(r"^--- ([A-Z][A-Za-z ()/]+) ---$", re.M)
_WORD_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------------------------

def _verdict_source():
    """The bench harness's own verdict logic, imported rather than reimplemented.

    Duplicating it is how this report first reported 70% for a battery the dashboard scores at
    96%: `summary.json["success"]` is the COMPLETION PREDICATE's answer, and sari_bench's own
    comments note several predicates grant on state they cannot ground. The dashboard groups by
    `success_final`, which prefers a reviewer's `verified_success` and drops `invalid` /
    `already_successful` rows from the totals entirely. One definition, one place.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        from sari_bench.watch import scan
        return scan
    except Exception:  # noqa: BLE001 - the token analysis stands without the outcome section
        return None


@dataclass
class Attempt:
    """One `bench_runs/<battery>/<prompt>/<try>/` directory."""
    path: str
    battery: str
    prompt_id: str
    summary: dict
    manifest: dict = field(default_factory=dict)   # attempt.json - outcome + reviewer verdict
    steps_per_leg: list = field(default_factory=list)
    semantic_chars: int = 0
    semantic_base_chars: int = 0
    semantic_entries: list = field(default_factory=list)
    episodic_chars: int = 0
    findings_chars: list = field(default_factory=list)

    @property
    def steps(self) -> int:
        return sum(self.steps_per_leg)

    @property
    def longest_leg(self) -> int:
        return max(self.steps_per_leg) if self.steps_per_leg else 0

    @property
    def depth(self) -> float:
        """sum over legs of sum_k(k) - the accumulated conversation depth the fit regresses on."""
        return sum(n * (n + 1) / 2 for n in self.steps_per_leg)

    @property
    def tokens(self) -> dict:
        return self.summary.get("tokens") or {}

    @property
    def tokens_in(self) -> int:
        return int(self.tokens.get("tokens_in") or 0)

    @property
    def calls(self) -> int:
        return int(self.tokens.get("calls") or 0)


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _read_text(path):
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    except OSError:
        return ""


def _count_steps(jsonl_path) -> int:
    """Step events in one leg's JSONL. Substring test rather than a json.loads per line: these
    files run to thousands of lines across the corpus and only the event name is needed."""
    n = 0
    try:
        with open(jsonl_path, encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if _STEP_EVENT in line:
                    n += 1
    except OSError:
        pass
    return n


def _findings_sizes(agent_log_text: str) -> list:
    """Sizes of the between-leg findings summaries, read off the orchestrator's own printout. The
    summaries are not persisted as files, so agent.log is the only record of what was generated."""
    sizes = []
    for match in re.finditer(r"^\[FINDINGS SUMMARY\]\n", agent_log_text, re.M):
        rest = agent_log_text[match.end():]
        end = re.search(r"^\[[A-Z]", rest, re.M)
        sizes.append(len(rest[:end.start()] if end else rest).__int__())
    return sizes


def load_attempts(bench_root: str) -> list:
    """Every attempt directory under bench_root that has a summary.json with token counts."""
    attempts = []
    if not os.path.isdir(bench_root):
        return attempts
    for battery in sorted(os.listdir(bench_root)):
        battery_dir = os.path.join(bench_root, battery)
        if not os.path.isdir(battery_dir):
            continue
        for prompt_id in sorted(os.listdir(battery_dir)):
            prompt_dir = os.path.join(battery_dir, prompt_id)
            if not os.path.isdir(prompt_dir):
                continue
            for try_id in sorted(os.listdir(prompt_dir)):
                run_dir = os.path.join(prompt_dir, try_id)
                summary = _read_json(os.path.join(run_dir, "summary.json"))
                manifest_path = os.path.join(run_dir, "attempt.json")
                if summary is None and not os.path.isfile(manifest_path):
                    continue
                # Attempts with NO token counts (crashed before the meter dumped) are kept: they
                # are real outcomes and dropping them silently moves every success rate. The token
                # estimators filter on `calls`/`steps` themselves, so they never see these rows.
                summary = summary or {}
                attempt = Attempt(path=run_dir, battery=battery, prompt_id=prompt_id,
                                  summary=summary,
                                  manifest=_read_json(os.path.join(run_dir, "attempt.json")) or {})
                for name in sorted(os.listdir(run_dir)):
                    if name.startswith("leg") and name.endswith(".jsonl"):
                        attempt.steps_per_leg.append(_count_steps(os.path.join(run_dir, name)))
                sem = _read_text(os.path.join(run_dir, "semantic_memory.txt"))
                if sem:
                    attempt.semantic_chars = len(sem)
                    first = _ENTRY_RE.search(sem)
                    attempt.semantic_base_chars = first.start() if first else len(sem)
                    attempt.semantic_entries = _ENTRY_RE.findall(sem)
                attempt.episodic_chars = len(_read_text(os.path.join(run_dir,
                                                                    "episodic_memory.txt")))
                attempt.findings_chars = _findings_sizes(
                    _read_text(os.path.join(run_dir, "agent.log")))
                attempts.append(attempt)
    return attempts


# ---------------------------------------------------------------------------------------------
# Step dumps - the per-step record of what each reasoner emitted
# ---------------------------------------------------------------------------------------------

def _sections(text: str) -> dict:
    """Split one stepNN.txt into its `--- NAME ---` sections."""
    out, marks = {}, list(_SECTION_RE.finditer(text))
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[mark.group(1).strip()] = text[mark.end():end].strip()
    return out


def _loads_loose(blob: str):
    """The dumps carry the reasoner's raw reply, which may be fenced or have trailing prose."""
    blob = blob.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```[a-z]*\n|\n```$", "", blob)
    for candidate in (blob, blob[blob.find("{"):blob.rfind("}") + 1]):
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    return None


@dataclass
class StepDump:
    router: dict          # semantic learner reply: new_semantic_memory / recall / mode / ...
    actor_chars: int      # the actor's returned block, retained in its history forever
    episodic_chars: int   # the reflection, re-sent to the actor every step


def load_leg_dumps(bench_root: str, limit=None) -> list:
    """Per-leg ordered lists of StepDump, for every leg that dumped its steps.

    `limit` caps how many legs are read - the redundancy and retention passes are O(entries^2) in
    difflib and the full corpus takes minutes.
    """
    legs, seen = [], 0
    for root, dirs, files in os.walk(bench_root):
        dirs.sort()
        names = sorted(f for f in files if re.fullmatch(r"step\d+\.txt", f))
        if not names:
            continue
        dumps = []
        for name in names:
            sections = _sections(_read_text(os.path.join(root, name)))
            router = _loads_loose(sections.get("MODE ROUTER (semantic)", "")) or {}
            dumps.append(StepDump(
                router=router,
                actor_chars=len(sections.get("VLM ACTOR OUTPUT", "")),
                episodic_chars=len(sections.get("EPISODIC REFLECTION", "")),
            ))
        legs.append(dumps)
        seen += 1
        if limit and seen >= limit:
            break
    return legs


# ---------------------------------------------------------------------------------------------
# The growth model
# ---------------------------------------------------------------------------------------------

def fit_growth(attempts: list) -> dict:
    """Least squares for tokens_in = A*steps + B*depth, no intercept (a zero-step run costs zero).

    Two parameters, solved off the normal equations - the repo has no numpy and this does not
    warrant one.
    """
    # `a.calls` is the metered guard: an attempt that crashed before the meter dumped still has
    # steps in its leg JSONL but tokens_in == 0, and regressing those against real step counts
    # drags both coefficients down (measured: R^2 0.997 -> 0.760 when they leak in).
    rows = [(a.steps, a.depth, a.tokens_in) for a in attempts if a.steps and a.calls]
    if len(rows) < 3:
        return {}
    s11 = sum(x * x for x, _, _ in rows)
    s12 = sum(x * y for x, y, _ in rows)
    s22 = sum(y * y for _, y, _ in rows)
    b1 = sum(x * t for x, _, t in rows)
    b2 = sum(y * t for _, y, t in rows)
    det = s11 * s22 - s12 * s12
    if not det:
        return {}
    flat = (b1 * s22 - b2 * s12) / det
    growth = (s11 * b2 - s12 * b1) / det
    mean = statistics.mean(t for _, _, t in rows)
    resid = sum((t - (flat * x + growth * y)) ** 2 for x, y, t in rows)
    total = sum((t - mean) ** 2 for _, _, t in rows)
    return {"flat_per_step": flat, "growth_per_depth": growth,
            "r2": 1 - resid / total if total else float("nan"), "n": len(rows)}


def tokens_per_call_by_leg_length(attempts: list, width: int = 5) -> list:
    """Mean tokens per call, bucketed by the attempt's LONGEST leg - the direct, model-free view
    of the same compounding the fit describes."""
    buckets = collections.defaultdict(list)
    for a in attempts:
        if a.calls and a.longest_leg:
            buckets[min(a.longest_leg // width * width, 60)].append(a.tokens_in / a.calls)
    return [(low, statistics.mean(vals), len(vals)) for low, vals in sorted(buckets.items())]


def role_attribution(attempts: list) -> dict:
    """Metered per-role totals, when the corpus has any. Supersedes every derived estimate below:
    once runs carry `by_role`, the ablation reads its answer straight off this rather than off
    CHARS_PER_TOKEN and a residual."""
    totals = collections.defaultdict(lambda: {"tokens_in": 0, "calls": 0})
    metered = 0
    for a in attempts:
        by_role = a.tokens.get("by_role") or {}
        if not by_role:
            continue
        metered += 1
        for role, row in by_role.items():
            totals[role]["tokens_in"] += int(row.get("tokens_in") or 0)
            totals[role]["calls"] += int(row.get("calls") or 0)
    return {"metered_attempts": metered, "attempts": len(attempts), "by_role": dict(totals)}


# ---------------------------------------------------------------------------------------------
# Component sizes
# ---------------------------------------------------------------------------------------------

def static_prompt_costs() -> dict:
    """Sizes of the fixed prompt scaffolding. Imported from the live modules, not hardcoded, so
    the report tracks prompt edits instead of going stale against them."""
    out = {}
    try:
        from agent_core.sys_inst import (AGENT_STATE_DOC, SYS_INST_ASSOCIATIVE_EPISODIC,
                                         SYS_INST_ASSOCIATIVE_SEMANTIC, SYS_INST_VLM_LEAN)
        out["semantic learner sys"] = len(SYS_INST_ASSOCIATIVE_SEMANTIC)
        out["actor sys (VLM_LEAN)"] = len(SYS_INST_VLM_LEAN)
        out["episodic learner sys"] = len(SYS_INST_ASSOCIATIVE_EPISODIC)
        out["  (of which AGENT_STATE_DOC, in both)"] = len(AGENT_STATE_DOC)
    except Exception as exc:  # noqa: BLE001 - a prompt-module import failure must not kill the report
        out["<sys_inst import failed>"] = f"{type(exc).__name__}: {exc}"
    return out


def component_sizes(attempts: list, legs: list) -> dict:
    """Per-step / per-leg sizes of every context component, in characters."""
    entries = [len(e) for a in attempts for e in a.semantic_entries]
    bases = [a.semantic_base_chars for a in attempts if a.semantic_base_chars]
    finals = [a.semantic_chars for a in attempts if a.semantic_chars]
    findings = [c for a in attempts for c in a.findings_chars]
    episodic = [d.episodic_chars for leg in legs for d in leg if d.episodic_chars]
    actor = [d.actor_chars for leg in legs for d in leg if d.actor_chars]
    recalls = [len(d.router.get("recall") or "") for leg in legs for d in leg
               if d.router.get("recall")]

    def stats(values):
        if not values:
            return None
        ordered = sorted(values)
        return {"n": len(ordered), "mean": statistics.mean(ordered),
                "median": ordered[len(ordered) // 2], "max": ordered[-1]}

    return {
        "semantic base (rendered map)": stats(bases),
        "semantic entry (appended per step)": stats(entries),
        "semantic blob at run end": stats(finals),
        "episodic reflection (per step)": stats(episodic),
        "actor reply (per step)": stats(actor),
        "recall (per step)": stats(recalls),
        "findings summary (per leg)": stats(findings),
    }


def decompose_growth(fit: dict, sizes: dict, chars_per_token: float) -> dict:
    """Split the fitted compounding term B into the components that actually grow with step index.

    MEASURED here means "read off the step dumps and converted at chars-per-token". The residual
    is what the fit says is compounding but no text component accounts for - the retained
    screenshots, which are the only other thing the actor's untrimmed history holds. Reported as a
    residual, and cross-checked against the encoder's own geometry, precisely because it is the
    one number here that is not read directly.
    """
    if not fit:
        return {}
    per_token = lambda key: (sizes[key]["mean"] / chars_per_token) if sizes.get(key) else 0.0

    # Read once per step by the learner, which re-reads the whole accumulated blob.
    semantic = per_token("semantic entry (appended per step)")
    # Injected into the actor's user message every step AND retained in its history, so step k
    # carries k copies of a reflection that was overwritten k-1 times.
    episodic = per_token("episodic reflection (per step)")
    # The actor's own replies stay in history as assistant turns.
    actor_reply = per_token("actor reply (per step)")
    recall = per_token("recall (per step)")

    measured = semantic + episodic + actor_reply + recall
    residual = fit["growth_per_depth"] - measured
    return {
        "growth_per_depth": fit["growth_per_depth"],
        "components": [
            ("semantic log entry (learner re-reads all)", semantic),
            ("episodic reflection retained in actor history", episodic),
            ("actor reply retained in actor history", actor_reply),
            ("recall line retained in actor history", recall),
        ],
        "measured": measured,
        "residual": residual,
        "residual_note": "retained screenshots + per-turn state/action blocks not in the dumps",
    }


def vision_tokens(width: int = 1280, height: int = 720) -> int:
    """Encoder-geometry estimate for one retained frame, as a sanity check on the residual."""
    return (width // VISION_PIXELS_PER_TOKEN) * (height // VISION_PIXELS_PER_TOKEN)


# ---------------------------------------------------------------------------------------------
# Per-arm sizing - what each ablation would actually remove
# ---------------------------------------------------------------------------------------------

def arm_savings(fit: dict, sizes: dict, decomposition: dict, retention: dict,
                chars_per_token: float, leg_lengths=REPORT_LEG_LENGTHS) -> list:
    """Tokens each arm removes from one leg of N steps, and what fraction of that leg it is.

    A `flat` component is paid once per step (a constant tax); a `growth` component is paid once
    per step PER PRIOR STEP, so it integrates to N(N+1)/2. The distinction is the whole point of
    the table: a big flat number and a small growth number describe very different risks as legs
    get longer.
    """
    if not fit:
        return []
    per_token = lambda key: (sizes[key]["mean"] / chars_per_token) if sizes.get(key) else 0.0
    entry = per_token("semantic entry (appended per step)")
    episodic = per_token("episodic reflection (per step)")
    findings = per_token("findings summary (per leg)")
    image = max(decomposition.get("residual", 0.0), 0.0)

    kept = (retention or {}).get("kept_entries", 12)
    # A2 caps the blob at `kept` entries, so past that depth the saving stops growing.
    def a2_growth_equivalent(n):
        full = entry * n * (n + 1) / 2
        capped = sum(entry * min(k, kept) for k in range(n + 1))
        return full - capped

    arms = [
        ("A1  drop semantic log entirely",
         lambda n: entry * n * (n + 1) / 2,
         "learner re-reads every entry every step; O(n^2) over a leg"),
        ("A2c dedupe(0.80)+last-12",
         a2_growth_equivalent,
         f"bounds the blob at ~{kept:.0f} entries; growth becomes flat"),
        ("A3  drop findings summaries",
         lambda n: findings * n,
         "rides the learner prompt EVERY step, not once per leg"),
        ("A4  cap findings at 600 chars",
         lambda n: max(findings - 600 / chars_per_token, 0.0) * n,
         "same seam as A3, keeps the content that fits"),
        ("A5  episodic out of actor history",
         lambda n: episodic * n * (n + 1) / 2,
         "step k holds k copies of a blob overwritten k-1 times"),
        ("A6  actor image history <= 2 turns",
         lambda n: image * max(n - 2, 0) * max(n - 1, 0) / 2,
         "every turn retains its full frame for the rest of the leg"),
    ]

    rows = []
    for name, saving, rationale in arms:
        totals = {}
        for n in leg_lengths:
            leg_total = fit["flat_per_step"] * n + fit["growth_per_depth"] * n * (n + 1) / 2
            saved = saving(n)
            totals[n] = (saved, saved / leg_total if leg_total else 0.0)
        rows.append({"arm": name, "rationale": rationale, "by_leg_length": totals})
    return rows


# ---------------------------------------------------------------------------------------------
# Redundancy + retention replay (the A2 evidence)
# ---------------------------------------------------------------------------------------------

def _words(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 3}


def _dedupe(entries: list, threshold: float, window: int = 8) -> list:
    kept = []
    for entry in entries:
        if any(difflib.SequenceMatcher(None, entry, other).ratio() > threshold
               for other in kept[-window:]):
            continue
        kept.append(entry)
    return kept


def redundancy(attempts: list, threshold: float = 0.72, window: int = 8) -> dict:
    """Share of semantic entries that near-duplicate one of the previous `window`. This is the
    measurement that says the appended log is mostly restating itself, not accumulating."""
    shares, counts = [], []
    for a in attempts:
        entries = a.semantic_entries
        if len(entries) < 8:
            continue
        dup = sum(1 for i, e in enumerate(entries)
                  if any(difflib.SequenceMatcher(None, e, f).ratio() > threshold
                         for f in entries[max(0, i - window):i]))
        shares.append(dup / len(entries))
        counts.append(len(entries))
    if not shares:
        return {}
    return {"runs": len(shares), "mean_entries": statistics.mean(counts),
            "mean_share": statistics.mean(shares), "median_share": statistics.median(shares)}


def retention_replay(legs: list, threshold: float = 0.80, keep_last: int = 12) -> dict:
    """Replay a retention policy against real traces and ask what it would have cost.

    For each step, `coverage` is the fraction of the recall the learner ACTUALLY emitted whose
    content is still present in the retained subset, normalised by what the full blob covered.
    1.0 means the policy would have lost the learner nothing it demonstrably used. The 5th
    percentile matters more than the mean: that is where truncation changes behaviour.
    """
    coverage, kept_fraction, kept_counts = [], [], []
    for leg in legs:
        entries = []
        for dump in leg:
            recall = _words(dump.router.get("recall") or "")
            if entries and recall:
                full = set().union(*[_words(e) for e in entries])
                base = len(recall & full) / len(recall)
                if base > 0:
                    subset = _dedupe(entries, threshold)[-keep_last:] if keep_last else \
                        _dedupe(entries, threshold)
                    got = set().union(*[_words(e) for e in subset]) if subset else set()
                    coverage.append((len(recall & got) / len(recall)) / base)
            new = dump.router.get("new_semantic_memory") or ""
            if new:
                entries.append(new)
        if len(entries) >= 15:
            surviving = _dedupe(entries, threshold)
            kept_fraction.append(len(surviving) / len(entries))
            kept_counts.append(len(surviving[-keep_last:] if keep_last else surviving))
    if not coverage:
        return {}
    coverage.sort()
    return {"threshold": threshold, "keep_last": keep_last, "n": len(coverage),
            "mean": statistics.mean(coverage), "median": statistics.median(coverage),
            "p5": coverage[len(coverage) // 20],
            "kept_fraction": statistics.mean(kept_fraction) if kept_fraction else float("nan"),
            "kept_entries": statistics.mean(kept_counts) if kept_counts else keep_last}


def retention_sweep(legs: list, thresholds=(0.55, 0.62, 0.68, 0.72, 0.80, 0.88),
                    keep_last: int = 12) -> list:
    return [retention_replay(legs, t, keep_last) for t in thresholds]


# ---------------------------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------------------------

def outcomes(attempts: list) -> dict:
    """Success rates under all three definitions in play, plus the cost profile of wins vs losses.

    THE THREE ARE NOT INTERCHANGEABLE, and picking the wrong one silently changes the answer:

      predicate      `summary.json["success"]` - what subtask_completion decided. Optimistic:
                     sari_bench's own notes say several predicates grant on state they cannot
                     ground. Cheap, always available, never reviewed.
      attempt        `success_final` per ATTEMPT - the reviewer's call where one exists, else the
                     predicate, with invalid / already_successful rows EXCLUDED (not failed).
      prompt         `success_final` per PROMPT, any try passing. This is the dashboard headline.

    For an ablation, use `attempt`: `prompt` saturates (a 5-try any-pass rate hits 96% on easy and
    has only ~20 units of resolution), and `predicate` measures the guard, not the agent.

    The success/failure token split carries a warning on purpose: failing runs are long BECAUSE
    they fail, so total tokens is a confounded outcome measure. It is here to make the confound
    visible, not to license it.
    """
    scan = _verdict_source()
    per_battery = collections.defaultdict(
        lambda: {"attempts": 0, "predicate_wins": 0, "included": 0, "attempt_wins": 0,
                 "prompts": collections.defaultdict(list), "reviewed": 0})
    end_reasons = collections.Counter()
    won, lost = [], []

    for a in attempts:
        row = per_battery[a.battery]
        row["attempts"] += 1
        predicate = bool(a.summary.get("success"))
        row["predicate_wins"] += predicate
        if a.calls:   # unmetered attempts would enter the cost split as spurious zeros
            (won if predicate else lost).append(a.tokens_in)

        if scan is not None:
            verdict = scan.verdict_of(a.manifest)
            row["reviewed"] += bool(verdict)
            if verdict in scan.EXCLUDED_VERDICTS:
                continue          # excluded from every total rather than counted as a failure
            reviewed = a.manifest.get("verified_success")
            final = bool(reviewed) if reviewed is not None else predicate
            row["included"] += 1
            row["attempt_wins"] += final
            row["prompts"][a.prompt_id].append(final)

        for leg in a.summary.get("legs") or []:
            end_reasons[leg.get("end_reason")] += 1

    per_battery_out = {}
    for battery, row in sorted(per_battery.items()):
        prompts = row.pop("prompts")
        row["n_prompts"] = len(prompts)
        row["prompt_wins"] = sum(1 for tries in prompts.values() if any(tries))
        per_battery_out[battery] = row

    return {
        "have_verdicts": scan is not None,
        "per_battery": per_battery_out,
        "end_reasons": dict(end_reasons.most_common()),
        "median_tokens_success": statistics.median(won) if won else None,
        "median_tokens_failure": statistics.median(lost) if lost else None,
    }


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------

def _rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def render(report: dict, chars_per_token: float) -> None:
    fit = report["fit"]
    sizes = report["component_sizes"]

    _rule("CORPUS")
    print(f"  bench root      {report['bench_root']}")
    print(f"  attempts        {report['n_attempts']} total, "
          f"{report['n_metered']} with token counts")
    print(f"                  (outcomes use all of them; token estimates use the metered subset)")
    print(f"  legs dumped     {report['n_legs']} ({report['n_step_dumps']} step dumps"
          f"{', sampled' if report['sampled'] else ''})")
    print(f"  total input     {report['total_tokens_in']:,} tokens")

    _rule("STATIC PROMPT COSTS (per call that carries them)")
    for name, value in report["static"].items():
        if isinstance(value, int):
            print(f"  {name:44} {value:7,} chars  ~{value / chars_per_token:6,.0f} tok")
        else:
            print(f"  {name:44} {value}")

    _rule("COMPONENT SIZES (characters)")
    print(f"  {'component':44} {'n':>6} {'mean':>8} {'median':>8} {'max':>8}")
    for name, s in sizes.items():
        if s:
            print(f"  {name:44} {s['n']:6,} {s['mean']:8,.0f} {s['median']:8,} {s['max']:8,}")

    _rule("TOKEN GROWTH MODEL")
    if fit:
        print(f"  tokens_in ~= {fit['flat_per_step']:,.0f} * steps "
              f"+ {fit['growth_per_depth']:,.0f} * sum_k(k)")
        print(f"  R^2 = {fit['r2']:.3f}   (N = {fit['n']} attempts)")
        print(f"\n  {'leg length':>11} {'flat':>13} {'growth':>14} {'total':>14} {'growth share':>13}")
        for n in REPORT_LEG_LENGTHS:
            flat = fit["flat_per_step"] * n
            grow = fit["growth_per_depth"] * n * (n + 1) / 2
            print(f"  {n:>11} {flat:>13,.0f} {grow:>14,.0f} {flat + grow:>14,.0f} "
                  f"{grow / (flat + grow):>12.0%}")

    _rule("TOKENS PER CALL, BY LONGEST LEG (model-free view of the same effect)")
    for low, mean, n in report["per_call"]:
        label = f"{low}-{low + 4}" if low < 60 else "60+"
        print(f"  {label:>8} steps   {mean:8,.0f} tok/call   (n={n})")

    _rule("WHAT COMPOUNDS (decomposition of the growth term)")
    dec = report["decomposition"]
    if dec:
        print(f"  fitted growth {dec['growth_per_depth']:,.0f} tok per step of accumulated depth\n")
        for name, value in dec["components"]:
            print(f"    {name:52} {value:7,.0f} tok  {value / dec['growth_per_depth']:5.0%}")
        print(f"    {'-' * 52} {'-' * 7}")
        print(f"    {'measured text subtotal':52} {dec['measured']:7,.0f} tok  "
              f"{dec['measured'] / dec['growth_per_depth']:5.0%}")
        print(f"    {'residual (' + dec['residual_note'] + ')':52} {dec['residual']:7,.0f} tok  "
              f"{dec['residual'] / dec['growth_per_depth']:5.0%}")
        print(f"\n  cross-check: one 1280x720 frame is ~{vision_tokens():,} tokens by encoder "
              f"geometry.\n  DERIVED, not metered - see role_attribution below.")

    _rule("PER-ARM SIZING: what each ablation removes from one leg")
    print("  saving as a share of that leg's total input tokens\n")
    header = "  {:38}".format("arm") + "".join(f"{n:>12}" for n in REPORT_LEG_LENGTHS)
    print(header)
    for row in report["arms"]:
        cells = "".join(f"{row['by_leg_length'][n][1]:>11.1%} " for n in REPORT_LEG_LENGTHS)
        print(f"  {row['arm']:38}{cells}")
        print(f"    {row['rationale']}")

    _rule("A2 EVIDENCE: redundancy and retention")
    red = report["redundancy"]
    if red:
        print(f"  entries near-duplicating one of the previous 8 "
              f"(ratio > 0.72): {red['mean_share']:.0%} mean, "
              f"{red['median_share']:.0%} median")
        print(f"  over {red['runs']} runs, mean {red['mean_entries']:.0f} entries/run")
    if report["sweep"]:
        print(f"\n  retention replay - coverage of the recall the learner actually emitted")
        print(f"  {'threshold':>10} {'kept':>7} {'mean':>8} {'median':>8} {'5th pct':>9}")
        for row in report["sweep"]:
            if row:
                print(f"  {row['threshold']:>10.2f} {row['kept_fraction']:>6.0%} "
                      f"{row['mean']:>8.1%} {row['median']:>8.1%} {row['p5']:>9.1%}")
        print("\n  Pick the knee, and weight the 5th percentile: that is where a policy stops")
        print("  being free and starts changing what the learner can act on.")

    _rule("METERED PER-ROLE ATTRIBUTION")
    roles = report["roles"]
    print(f"  attempts carrying by_role: {roles['metered_attempts']} / {roles['attempts']}")
    if roles["by_role"]:
        total = sum(r["tokens_in"] for r in roles["by_role"].values()) or 1
        for name, row in sorted(roles["by_role"].items(),
                                key=lambda kv: -kv[1]["tokens_in"]):
            print(f"    {name:16} {row['tokens_in']:12,} tok  {row['tokens_in'] / total:5.1%}  "
                  f"({row['calls']:,} calls)")
    else:
        print("    none - this corpus predates per-role accounting, so the decomposition above")
        print("    is derived. Re-run after a metered battery to replace it with measurements.")

    _rule("OUTCOMES")
    out = report["outcomes"]
    if not out["have_verdicts"]:
        print("  sari_bench not importable - showing the completion PREDICATE only, which is not")
        print("  what the dashboard reports. Run from the repo so the verdict logic is shared.\n")
    print(f"  {'battery':34} {'predicate':>12} {'per-attempt':>13} {'per-prompt':>12} {'reviewed':>10}")
    print(f"  {'':34} {'(optimistic)':>12} {'(use this)':>13} {'(dashboard)':>12} {'':>10}")
    for battery, row in out["per_battery"].items():
        pred = f"{row['predicate_wins']}/{row['attempts']}"
        att = f"{row['attempt_wins']}/{row['included']}" if row["included"] else "-"
        pro = f"{row['prompt_wins']}/{row['n_prompts']}" if row["n_prompts"] else "-"
        pct = lambda a, b: f" {a / b:.0%}" if b else ""
        print(f"  {battery:34} {pred + pct(row['predicate_wins'], row['attempts']):>12} "
              f"{att + pct(row['attempt_wins'], row['included']):>13} "
              f"{pro + pct(row['prompt_wins'], row['n_prompts']):>12} "
              f"{row['reviewed']:>6}/{row['attempts']:<3}")
    print("\n  Per-prompt is the dashboard headline and SATURATES - poor ablation resolution.")
    print("  Per-attempt is the one to power an ablation on. Watch the reviewed column: where")
    print("  review coverage is thin, per-attempt is mostly the predicate wearing a better name,")
    print("  so two arms reviewed to different depths are not comparable.")
    print(f"\n  leg end reasons: {out['end_reasons']}")
    if out["median_tokens_success"] and out["median_tokens_failure"]:
        print(f"\n  median input tokens: success {out['median_tokens_success']:,.0f}  "
              f"vs failure {out['median_tokens_failure']:,.0f}")
        print("  CONFOUNDED: failing runs are long BECAUSE they fail. Do not use total tokens as")
        print("  an ablation outcome - judge on success rate and steps-to-success, cost separately.")
    print()


# ---------------------------------------------------------------------------------------------

def build_report(bench_root: str, chars_per_token: float, deep: bool, sample: int) -> dict:
    attempts = load_attempts(bench_root)
    legs = load_leg_dumps(bench_root, limit=None if deep else sample)
    fit = fit_growth(attempts)
    sizes = component_sizes(attempts, legs)
    decomposition = decompose_growth(fit, sizes, chars_per_token)
    sweep = retention_sweep(legs) if deep else [retention_replay(legs)]
    best = next((row for row in sweep if row and row.get("threshold") == 0.80), None) or \
        (sweep[0] if sweep else {})
    return {
        "bench_root": bench_root,
        "n_attempts": len(attempts),
        "n_metered": sum(1 for a in attempts if a.calls),
        "n_legs": len(legs),
        "n_step_dumps": sum(len(leg) for leg in legs),
        "sampled": not deep,
        "total_tokens_in": sum(a.tokens_in for a in attempts),
        "static": static_prompt_costs(),
        "component_sizes": sizes,
        "fit": fit,
        "per_call": tokens_per_call_by_leg_length(attempts),
        "decomposition": decomposition,
        "arms": arm_savings(fit, sizes, decomposition, best, chars_per_token),
        "redundancy": redundancy(attempts),
        "sweep": sweep,
        "roles": role_attribution(attempts),
        "outcomes": outcomes(attempts),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bench-dir", default=os.path.join(REPO_ROOT, "bench_runs"),
                        help="battery root to analyse (default: <repo>/bench_runs)")
    parser.add_argument("--deep", action="store_true",
                        help="read every leg and sweep dedupe thresholds (minutes, not seconds)")
    parser.add_argument("--sample", type=int, default=120,
                        help="legs to read when not --deep (default: 120)")
    parser.add_argument("--chars-per-token", type=float, default=CHARS_PER_TOKEN,
                        help=f"text conversion factor (default: {CHARS_PER_TOKEN})")
    parser.add_argument("--json", default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.bench_dir):
        print(f"no such bench dir: {args.bench_dir}", file=sys.stderr)
        return 2

    report = build_report(args.bench_dir, args.chars_per_token, args.deep, args.sample)
    if not report["n_attempts"]:
        print(f"no attempts with token counts under {args.bench_dir}", file=sys.stderr)
        return 1

    render(report, args.chars_per_token)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"[written] {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
