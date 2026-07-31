# Sari Agent — performance report (easy / medium / hard)

**Batteries:** `bench_runs/20260727_020820-easy`, `bench_runs/20260728_143857-medium`,
`bench_runs/20260730_205702-hard`.
**Scope:** 54 prompts, 233 attempts, 137.5 h of summed agent runtime, 265.5 M tokens.
**Model:** `Qwen/Qwen3.6-27B` for every role, `arm=graph`, 150 steps / 40 min per leg.
**Scoring:** "success" below means a **human reviewer passed the attempt on replay**
(`verified_verdict == "pass"`). `invalid` and `already_successful` attempts are excluded from
every rate, never counted as failures — the same rule `sari_bench report` and the watch dashboard
use, imported from `sari_bench.watch.scan` rather than reimplemented.
**Reproduce:** `python3 analysis/agent-report/compile_report.py` regenerates every number here into
`results.txt`, `stats.json` and the per-attempt `attempts.csv`, from `bench_runs/` alone.
Log excerpts come from `python3 analysis/agent-report/mine_logs.py`.

> **Read the batteries as three separate experiments, not one ladder.** They differ in more than
> difficulty: easy ran with **no completion guard**, medium and hard ran `completion_guard=vlm`;
> hard ran **3 tries** per prompt where easy and medium ran 5; hard is the baseline arm of the
> context ablation. Comparisons across difficulty are directional, not controlled.

---

## Headline, per difficulty

| Metric | easy | medium | hard |
|---|---:|---:|---:|
| Prompts × tries | 23 × 5 | 12 × 5 | 19 × 3 |
| Attempts | 116 | 60 | 57 |
| **Human grading success rate** | **55.1%** (54/98) | **68.0%** (17/25) | **28.6%** (12/42) |
| **Avg time to a graded success** | **20.2 min** | **42.0 min** | **24.9 min** |
| Median time to a graded success | 12.1 min | 41.6 min | 8.9 min |
| **Avg tokens per graded success** | **675k** | **783k** | **389k** |
| **Total tokens used** | **102.6 M** | **49.9 M** | **113.0 M** |
| Tokens per *solved prompt* (all spend ÷ prompts solved) | 4.66 M | 4.53 M | **16.1 M** |
| Avg runtime per attempt, failures included | 23.4 min | 39.6 min | 57.6 min |
| Prompts solved (any try passed) | 22/23 (95.7%) | 11/12 (91.7%) | **7/19 (36.8%)** |

**Total across all three: average runtime per attempt including failures is 36.0 min**
(median 23.3 min) over 233 attempts. Overall human grading success rate is **50.3%** (83/165).
Summed agent runtime 137.5 h; combined calendar span 53.2 h (the batteries run attempts in
parallel — medium at 11.5× and hard at 7.4× effective parallelism, easy essentially serial at 1.0×).

### Grading coverage — the caveat on the medium number

Not every attempt carries a verdict, so the denominators differ from the attempt counts:

| | easy | medium | hard |
|---|---:|---:|---:|
| Graded (pass + fail) | 98 | **25** | 42 |
| Excluded (`invalid` / `already_successful`) | 15 | **35** | 14 |
| Never reviewed | 3 | 0 | 1 |
| Grading coverage | 84.5% | **41.7%** | 73.7% |

Medium's 68% rests on 25 graded attempts. 35 of its 60 attempts were halted as
`already_successful` — the harness killing sibling tries once a prompt was already won — which is
why its coverage is low and its rate is the most fragile of the three.

---

## Cost of failure

Failure is where the tokens go. A graded failure costs roughly **twice** a graded success, and on
hard it costs **8× the token spend of a hard success**:

| | easy | medium | hard |
|---|---:|---:|---:|
| Avg tokens, graded success | 675k | 783k | 389k |
| Avg tokens, graded failure | 1,246k | 1,284k | **3,223k** |
| Avg runtime, graded failure | 32.0 min | 37.7 min | **74.5 min** |
| Total LLM calls | 8,527 | 6,269 | 7,905 |
| Avg LLM calls per success | 65 | 122 | 66 |
| Tokens in / out | 100.2M / 2.4M | 48.3M / 1.6M | 110.6M / 2.5M |

Two things fall out of this:

1. **Hard successes are cheap; hard failures are ruinous.** Hard's median success is the fastest of
   all three batteries (8.9 min) — when the agent can do a hard task, it does it briskly. When it
   can't, it burns 74 minutes and 3.2 M tokens grinding into the time cap. 13 of hard's 30 graded
   failures ended at `time_cap`, versus 10 of 44 on easy and 1 of 8 on medium.
2. **Input dominates output 30–45×** across all three. Spend is context, not generation — every step
   re-sends the image history and memory.

---

## Harness self-report vs human verdict

The agent's own success predicate disagrees with the reviewer often enough that it cannot be used
as the score:

| | easy | medium | hard |
|---|---:|---:|---:|
| Agreement (graded attempts) | 69.4% | 76.0% | 85.7% |
| Agent claimed win, human said **no** | 16 | 6 | 5 |
| Agent claimed loss, human said **yes** | 14 | 0 | 1 |

Easy — the battery with **no completion guard** — is the one where the agent both over-claims and
under-claims most, and the only battery with a meaningful count of false *failures*. Turning the
VLM completion guard on (medium, hard) removes essentially all of the false-loss cases and lifts
agreement 7–16 points. That is the single clearest config effect in the three runs.

---

## By task family (graded attempts, all difficulties pooled)

| Family | Graded | Pass | Rate | Avg pass time | Avg tokens/pass |
|---|---:|---:|---:|---:|---:|
| checkout | 16 | 13 | **81.2%** | 36.1 min | 712k |
| inspect | 10 | 8 | 80.0% | 33.7 min | 632k |
| pickup | 20 | 13 | 65.0% | 16.0 min | 670k |
| attribute | 70 | 35 | 50.0% | 17.8 min | 517k |
| counting | 10 | 5 | 50.0% | 26.1 min | 1,031k |
| reasoning | 29 | 8 | 27.6% | 39.0 min | 697k |
| navigate | 10 | 1 | **10.0%** | 93.7 min | 2,541k |

**Manipulation is solved; open-ended search is not.** Physically getting an item and scanning it
at checkout is the agent's strongest skill (81%). What breaks it is being told to *find* something
underspecified — `navigate` passes once in ten, at 94 minutes and 2.5 M tokens per pass, and
`reasoning` (budget/constraint tasks) passes barely a quarter of the time.

## Which try wins

Passes are spread almost evenly across try index — try01: 20, try02: 20, try03: 17, try04: 12,
try05: 14. **Retries are not converging on a solution; each try is close to an independent
coin-flip.** There's no learning across tries (memory is per-attempt), so pass@k grows the way
independent samples do — which is exactly why prompt-level solve rate (95.7% easy) sits so far
above per-attempt success rate (55.1% easy).

---

## Log excerpts

### 1. It plans the whole shop before moving — and interleaves pickup with checkout

`hard_09/try01`, *"Buy chips, noodles, and soup while staying within 150 php"*. The planner
decomposes into six legs and resolves each target to a checkpoint set, one item at a time rather
than carrying three items at once:

```
[ORCHESTRATOR] 6 leg(s) (resolver calls: 3):
  1. [pickup] Pick up the noodles.  -> cps [21, 43, 52, 62, 63]
  2. [checkout] Take the held noodles to the self-checkout: scan it and bag it.  -> cps [64]
  3. [pickup] Pick up the chips.  -> cps [20, 21, 24, 28, 44, 50, 51, 52, 62, 63]
  4. [checkout] Take the held chips to the self-checkout: scan it and bag it.  -> cps [64]
  5. [pickup] Pick up the soup.  -> cps [45, 47, 48]
  6. [checkout] Take the held soup to the self-checkout: scan it and bag it.  -> cps [64]

[GATE] off-target: at cp35, target at [21, 43, 52, 62, 63] — forcing navigation this step.
```

### 2. It catches itself holding the wrong box — from the gripper state, not the image

`hard_01/try03`. The best moment in the three batteries. The agent reports a successful grab, then
notices the simulator says its right hand contains Choco Crunchies when the goal was Hello Panda,
argues with itself about whether it's a mislabel, and resolves to put it back:

> *"…The `last_reach` says 'REACHABLE and GRABBED'. The `gripped_names` says the right hand is
> holding 'FIBISCO_CHOCOCRUNCHIES_ORIGINAL_200G'. This is a discrepancy. The agent likely grabbed
> the Choco Crunchies instead of the Hello Panda… Therefore, the task is not complete. I must
> release the Choco Crunchies and grab the Hello Panda."*
> — `next_action: "Release the Choco Crunchies held in the right hand to free it for grabbing the Hello Panda."`

And it writes the lesson into semantic memory so the next step inherits it:

> *"The agent previously grabbed Choco Crunchies by mistake when aiming for Hello Panda, so precise
> centering is required."*

### 3. It diagnoses its own death loop — and still dies in it

`hard_13/try02`, *"You only have 300 php. Choose which of these items to remove to stay under
budget: 2x Nestle Honey Stars, 1x Pancit Canton"*. The agent holds the budget arithmetic correctly
across dozens of steps and even names its failure mode:

> *"The agent is **stuck in a perception loop** at Checkpoint 31. To complete the task, it must
> navigate to the noodle aisle (Checkpoint 43) to find Pancit Canton. The price of Nestle Honey
> Stars is already known (105 PHP)."*
> — `next_action: "Navigate to Checkpoint 43 to find Pancit Canton."`

It repeats that same intent for the rest of the run and never leaves cp31. 5,234 s, 286 LLM calls,
7.3 M tokens later it files an honest report:

> *"I was unable to complete the budget calculation because I could not find the price tag for the
> Pancit Canton. I did confirm that the Nestle Honey Stars cost 105 PHP each, but without the
> second price, I cannot determine which item to remove to stay under your 300 PHP budget."*

Naming the loop and escaping the loop are clearly separate capabilities.

### 4. Its self-written "WHAT TO AVOID" is genuinely good advice

Episodic memory, rewritten every timestep. `hard_13/try02`, timestep 1:

> **WHAT WORKED:** *"…It appropriately utilized the 'perception' mode to scan the environment
> rather than immediately attempting navigation or interaction."*
> **WHAT TO AVOID:** *"The agent should avoid relying solely on… numerical checkpoint confirmation
> without verifying visual presence. It is crucial to trust visual cues over the assumption that
> being at a 'target checkpoint' guarantees immediate visibility of the item."*

Compare `hard_01/try01`, which learned a mode-discipline rule the hard way:

> **WHAT TO AVOID:** *"The agent should avoid attempting navigation actions (like 'move_forward')
> while in 'perception' mode, as these actions are invalid and will not execute."*

### 5. The centering controller talks back

Not the model — the harness — but it's the most-quoted line in the logs and it explains a large
share of wasted steps:

```
[CENTER] look 5: 'Pocky' center=(994,481) residual=(+34,-59)px  (12 candidate(s))
[CENTER] residual stopped improving ((33.6, -58.9)px) - stopping (target likely at frame edge).
[CENTER] result: STALLED - centring stopped improving at (33.6, -58.9)px; the target is likely
         near the frame edge - bring it more into view, then center again
```

Twelve simultaneous Pocky detections, none centerable. The agent's usual response is to re-center,
which stalls the same way — this is the mechanical core of the "perception loop" it names in §3.

### 6. The 94-minute pass that the harness scored as a failure

`easy_19/try05`, *"Find an item near the checkout"* — the slowest graded pass in all three
batteries. The target resolver timed out, so the plan shipped with a leg the orchestrator itself
flagged as doomed:

```
[PLAN] resolve failed for 'an item near the checkout': ReadTimeout: ... (read timeout=180.0)
  2. [pickup] Find and pick up an item near the checkout.  [INFEASIBLE: target resolved to no checkpoint]
[ORCHESTRATOR] WARNING: leg(s) [2] resolved to no checkpoint - the plan may be doomed,
               but running so the failure is measured, not assumed.
```

It ran 5,623 s, hit `time_cap`, and the harness recorded `success=False`. The reviewer watched the
replay and passed it. One attempt, three different answers depending on who you ask — which is the
argument for human grading in one artifact.

---

## What I'd take away

1. **Report prompt-level solve rate and per-attempt rate side by side.** They tell opposite stories
   on easy (95.7% vs 55.1%) and the gap is pure retry variance, not capability.
2. **The completion guard is doing real work.** Easy, the only battery without it, has 14 false
   failures; medium and hard have one between them.
3. **Hard's cost problem is a timeout problem.** 3.2 M tokens per hard failure with 13/30 ending at
   `time_cap` says the ceiling is a stuck-detector, not a smarter policy — the agent already
   *knows* it's looping (§3). An "I have not changed checkpoint in N steps → abandon leg" rule
   would reclaim most of hard's 113 M tokens.
4. **Medium's 68% needs more grading before it's quotable.** 25 of 60 attempts carry a verdict.
