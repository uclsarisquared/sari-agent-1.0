# Leg runner invariants

This document records the behavioral constraints behind the typed-leg execution loop. Keep the
implementation comments short; update this document when a measured failure changes one of these
rules.

## Module boundaries

- `leg_runner.py` owns the high-level loop, budgets, actor requests, state-machine ordering, and the
  public compatibility API.
- `leg_runtime.py` owns simulator-state construction, the model-facing state projection, durable grip
  identity, inspection evidence, and post-action reconciliation.
- `leg_artifacts.py` owns JSONL output, screenshots, full response dumps, and file-handle cleanup.
- `leg_actions.py` owns actor invocation accounting and dispatching a parsed action batch.
- `leg_completion.py` owns frame-bound guards, completion streaks, STOP decisions, refusal limits,
  and wrong-item correction.
- `leg_session.py` holds durable per-leg execution fields and the current step's intermediate values,
  keeping the runner loop focused on phase ordering.

`subtask_agents.py` and `validation/evals/pickup_navigation.py` still import historical private names
from `leg_runner.py`. The compatibility aliases must remain until those callers migrate.

## State and prompt projection

The completion predicate always receives the complete state. The actor receives a lean projection:

- `visited_checkpoints` is code-only and grows throughout the task, so it must not enter every model
  prompt.
- `last_checkout.steps` contains primitive-level diagnostics and is removed from the actor view.
- `last_inspection.frame_b64` is a full-resolution image already supplied as the current vision input;
  it must never be duplicated into text state.

The navigation advisor receives the bare current-leg goal. The actor and learner receive the
augmented task containing previous context and future-goal awareness. Future goals are informational
and must not be executed by the current leg.

## Location ownership

The graph, rather than the mode-router VLM, decides whether a pickup or goto leg has reached its
resolved target. A location-gated leg is considered on target when the nearest checkpoint is a
candidate or at most one graph hop from a candidate. The one-hop margin tolerates fine approach
motion near a shelf.

The gate fails open when candidates, localization, or a connected graph path are unavailable. It
must not force navigation based on incomplete grounding. Checkout and compare legs are excluded:
checkout has its own navigation macro, while compare intentionally visits multiple candidate sets.

The starting checkpoint counts as visited and is written into state before step one. This matters
for compare legs starting at a candidate and for the location gate's first decision.

Historical basis: run `0724_145735` showed a pickup router attaching the target to an item left at
checkout while the agent was 8–12 graph hops from every resolved candidate.

## Grip identity across frames and legs

Live hovered-object identity clears after a hand retracts from the shelf. The SKU reported at grip
time is therefore the durable identity used by pickup completion.

- Identity is tracked independently for the left and right hands.
- A carried name is seeded at a leg boundary only if that hand is still gripping.
- A name is cleared immediately when its hand releases.
- A pickup leg records which hands were already occupied at leg start. Untargeted pickup completion
  requires a newly occupied hand, not merely an item carried into the leg.
- Drop-style completion records whether a hand occupied at leg start was released, even when the
  other hand remains occupied.

This behavior was measured in July 2026 after hovered identities became `null` at the halt frame and
after dual-hand pickup lost the first item's identity when represented by a single slot.

## Frame-bound completion guards

A guard verdict is valid only for the screenshot and state from which it was produced.

- Targeted VLM pickup is evaluated before the action against the current frame and current held SKUs.
  If the action changes a grip, that verdict is not reused after the action; the next frame evaluates
  the new state.
- Inspection STOP verification uses the structured `reported_answer` emitted with that exact frame.
  Previous action text and termination placeholders are not inspection answers.
- Unknown and compare legs may fall back to the last actor text only where their completion contract
  explicitly permits it.
- With completion verification disabled, only an explicit STOP ends the leg. There is no implicit
  completion nudge or backstop.

A positive measurable predicate sets `goal_check`, telling the actor to STOP the current leg. After
`COMPLETION_BACKSTOP` consecutive positive observations, the runner ends successfully even if the
actor never requests STOP. A refused STOP clears both the nudge and the positive streak.

## Compare and inspection evidence

Compare guards cache one frame per target candidate set and are created only once all required target
frames exist. The frame metadata remains in structured logs for auditability.

Held-item inspection keeps a per-hand evidence ledger. Multi-item questions often require one label
per frame, so the guard evaluates the current frame together with earlier visibility-approved frames.
Evidence is removed when its hand releases. The actor-facing ledger contains metadata only; base64
frames remain guard-only.

An inspect leg with a held item is constrained to inspection-safe manipulation actions. An unheld
inspect leg receives a finite small-step approach budget. Inspection cleanup restores canonical hand
transforms after every terminal path and after exceptions, and cleanup failures never mask the
original leg outcome.

## STOP refusal and wrong-item correction

STOP is a request. The typed completion predicate is the sole authority that grants it.

Repeatedly stopping while holding a conclusively mismatched pickup can trigger one corrective release
per leg. Only a newly gripped, identified, mismatched hand may be released; an item carried from a
previous leg and an unidentified hand are protected. A successful corrective release resets the
refusal count and gives the actor another pickup attempt.

When the refusal cap is reached, the leg ends as `halt_forced`, not as success. This prevents an
unbounded STOP loop without weakening the completion predicate.

## Action dispatch

The mode router chooses before the actor chooses an action. A grab emitted one step early may be
promoted to manipulation only when `_grab_ready` proves readiness from the previous centering/reach
result. Promotion applies only to self-posing grab macros; raw hand motion and navigation remain under
their normal mode gates.

Parsing tolerates apostrophes in single-quoted product names. A parse failure consumes the step and is
logged; three actor or parse errors end the leg with `errors`.

## Budgets, metrics, and artifacts

A zero step or time cap means unbounded for that dimension. Positive caps retain `step_cap` and
`time_cap` terminal reasons.

Every step's screenshot, full response, and center-debug directory share the same timestamp suffix.
Saved screenshots are downscaled for storage; the actor receives native bytes. JSONL writes flush per
event so interrupted runs retain their trail. `LegArtifacts` owns the stream through a context manager
so exceptions cannot leak the handle.

`LegEventLogger` assembles event records, reserves the `event`, `leg`, and `step` keys, and attaches the
leg identifier consistently. Guard, failure, and full-step event families use dedicated assemblers so
their common schema is defined once.

The result dictionary remains the public contract. It includes the final reconciled state and semantic
entries created during the leg, plus timing, call count, refusal, correction, completion evidence, and
terminal-reason fields consumed by orchestration and benchmark reporting.
