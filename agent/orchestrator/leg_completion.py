"""Completion-guard lifecycle and STOP policy for typed leg execution."""

import time
from dataclasses import dataclass

from sim.env import _GRIP_LEFT_, _GRIP_RIGHT_
from orchestrator.pickup_vlm_guard import (
    cache_compare_candidate_frames,
    evaluate_hands,
    make_compare_guard,
    make_inspect_guard,
    make_unknown_guard,
)
from orchestrator.subtask_completion import (
    COMPLETION_BACKSTOP,
    HALT_REFUSAL_CAP,
    WRONG_ITEM_RELEASE_AFTER,
    blob_matches_target,
    completion_predicate,
    mismatched_hands,
    pickup_has_target,
    reported_completion_answer,
)
from orchestrator.leg_runtime import HANDS, fresh_agent_state


def deterministic_guard_details(leg, state):
    """Return per-hand diagnostics for a deterministic targeted pickup."""
    if leg.get("type") != "pickup" or not pickup_has_target(leg):
        return None
    names = state.get("gripped_names")
    names = names if isinstance(names, dict) else {}
    details = {}
    for side in HANDS:
        if not state.get(f"{side}GrippedState"):
            continue
        sku = names.get(side)
        if not sku:
            hovered = state.get(f"{side}HoveredObject")
            sku = hovered if hovered and str(hovered).lower() not in ("none", "null") else None
        match = bool(sku and blob_matches_target(sku, leg.get("target") or ""))
        details[side] = {
            "match": match,
            "reason": (
                "deterministic SKU/target match"
                if match
                else "deterministic SKU/target mismatch or unidentified held item"
            ),
            "conclusive": bool(sku),
            "latency_ms": 0.0,
            "sku": sku,
            "reused": False,
        }
    return details


@dataclass
class StepGuards:
    inspect: object = None
    unknown: object = None
    pickup_verdicts: dict | None = None
    early_met: bool = False
    early_reason: str = ""
    terminal: bool = False


@dataclass
class CompletionObservation:
    met: bool
    reason: str
    terminal: bool = False


class CompletionController:
    """Own completion evidence, streaks, refusal policy, and guard caches."""

    def __init__(
        self, agent, leg, backend, metrics, log, leg_idx, state_reader=fresh_agent_state
    ):
        self.agent = agent
        self.leg = leg
        self.backend = backend
        self.metrics = metrics
        self.events = log
        self.leg_idx = leg_idx
        self.state_reader = state_reader
        self.goal_met_streak = 0
        self.halt_refusals = 0
        self.corrective_release_done = False
        self.last_guard_skus = None
        self.compare_frames = {}
        self.compare_guard = None
        self.targeted_pickup = (
            backend == "vlm"
            and leg.get("type") == "pickup"
            and pickup_has_target(leg)
        )
        self.targeted_compare = (
            backend == "vlm"
            and leg.get("type") == "compare"
            and bool(leg.get("targets"))
        )
        self.targeted_unknown = (
            backend == "vlm"
            and leg.get("type")
            not in ("pickup", "checkout", "compare", "goto", "inspect")
        )

    def _client_args(self):
        vlm = self.agent.vlm_agent
        return vlm.client, vlm.config.model_id, vlm.config

    def _emit(self, event, *, step=None, **fields):
        if hasattr(self.events, "emit"):
            self.events.emit(event, step=step, **fields)
            return
        row = {"event": event, **fields}
        if step is not None:
            row["step"] = step
        self.events(row)

    def _guard_event(self, step, backend, verdict, *, guard=None, **context):
        if hasattr(self.events, "guard"):
            self.events.guard(
                step, backend, verdict, guard=guard, **context
            )
            return
        row = verdict if isinstance(verdict, dict) else {}
        fields = {
            "backend": backend,
            "match": row.get("match"),
            "reason": row.get("reason"),
            "conclusive": row.get("conclusive"),
            "latency_ms": row.get("latency_ms"),
            "reused": row.get("reused", False),
            **context,
        }
        if guard is not None:
            fields["guard"] = guard
        self._emit("completion_guard", step=step, **fields)

    def prepare(self, state, image_b64, step, inspection_evidence, last_actor_text):
        """Bind guards to the current frame and evaluate pre-action pickup completion."""
        guards = StepGuards()
        if self.leg.get("type") == "inspect" and self.backend != "none":
            evidence_frames = inspection_evidence.guard_frames()

            def log_inspect(query, auxiliary_context, verdict, reused):
                if not reused:
                    self.metrics["llm_calls"] += 1
                verdict = {
                    **(verdict if isinstance(verdict, dict) else {}),
                    "reused": reused,
                }
                self._guard_event(
                    step,
                    "vlm",
                    verdict,
                    guard="inspect",
                    query=query,
                    auxiliary_context=auxiliary_context,
                    evidence_frames=[frame["label"] for frame in evidence_frames],
                )

            guards.inspect = make_inspect_guard(
                *self._client_args(),
                image_b64,
                on_verdict=log_inspect,
                evidence_frames=evidence_frames,
            )

        if self.targeted_unknown:
            def log_unknown(task, auxiliary_context, verdict, reused):
                if not reused:
                    self.metrics["llm_calls"] += 1
                verdict = {
                    **(verdict if isinstance(verdict, dict) else {}),
                    "reused": reused,
                }
                self._guard_event(
                    step,
                    "vlm",
                    verdict,
                    guard="unknown",
                    query=task,
                    auxiliary_context=auxiliary_context,
                )

            guards.unknown = make_unknown_guard(
                *self._client_args(), image_b64, on_verdict=log_unknown
            )

        self._prepare_compare_guard(state, image_b64, step)
        guards.pickup_verdicts = (
            deterministic_guard_details(self.leg, state)
            if self.backend == "deterministic"
            else None
        )
        if self.backend == "none":
            self._clear_progress(state)
            return guards
        if self.targeted_pickup:
            self._evaluate_pickup_before_action(
                guards, state, image_b64, step, last_actor_text
            )
        return guards

    def _prepare_compare_guard(self, state, image_b64, step):
        if not self.targeted_compare or self.compare_guard is not None:
            return
        targets = list(self.leg.get("targets") or [])
        candidate_sets = self.leg.get("candidate_sets")
        if not isinstance(candidate_sets, list) or len(candidate_sets) != len(targets):
            return
        cache_compare_candidate_frames(
            self.compare_frames,
            targets,
            candidate_sets,
            state.get("nearest_checkpoint"),
            image_b64,
            step,
        )
        if len(self.compare_frames) != len(targets) or len(targets) < 2:
            return
        ordered_frames = [self.compare_frames[index] for index in range(len(targets))]

        def log_compare(criterion, auxiliary_context, verdict, reused):
            if not reused:
                self.metrics["llm_calls"] += 1
            verdict = {
                **(verdict if isinstance(verdict, dict) else {}),
                "reused": reused,
            }
            self._guard_event(
                step,
                "vlm",
                verdict,
                guard="compare",
                criterion=criterion,
                targets=targets,
                candidate_frames=[
                    {
                        "target": frame["target"],
                        "checkpoint": frame["checkpoint"],
                        "step": frame["step"],
                    }
                    for frame in ordered_frames
                ],
                auxiliary_context=auxiliary_context,
            )

        self.compare_guard = make_compare_guard(
            *self._client_args(), ordered_frames, on_verdict=log_compare
        )

    def _evaluate_pickup_before_action(
        self, guards, state, image_b64, step, last_actor_text
    ):
        held_skus = {
            side: state.get("gripped_names", {}).get(side)
            for side in HANDS
            if state.get(f"{side}GrippedState")
            and state.get("gripped_names", {}).get(side)
        }
        guard_skus = tuple(
            (side, held_skus.get(side)) for side in HANDS if held_skus.get(side)
        )
        if guard_skus != self.last_guard_skus:
            self.goal_met_streak = 0
            self.last_guard_skus = guard_skus
        if held_skus:
            guards.pickup_verdicts, calls = evaluate_hands(
                *self._client_args(),
                image_b64,
                self.leg.get("target") or "",
                held_skus,
            )
            self.metrics["llm_calls"] += calls
            for side, verdict in guards.pickup_verdicts.items():
                self._guard_event(
                    step,
                    "vlm",
                    verdict,
                    guard="pickup",
                    side=side,
                    sku=verdict.get("sku"),
                )
        guards.early_met, guards.early_reason = completion_predicate(
            self.leg,
            state,
            final_text=last_actor_text,
            guard_backend=self.backend,
            guard_verdicts=guards.pickup_verdicts,
        )
        self._record_progress(state, guards.early_met, guards.early_reason)
        if self.goal_met_streak >= COMPLETION_BACKSTOP:
            guards.terminal = True
            self._complete_without_stop(
                step,
                guards.early_reason,
                backend=self.backend,
                guard_verdicts=guards.pickup_verdicts,
            )

    def handle_stop(self, response, state, last_actor_text, guards, step, grip_tracker):
        """Return granted, corrected, refused, or forced for an explicit STOP request."""
        reported_answer = reported_completion_answer(response)
        if self.leg.get("type") == "inspect":
            final_text = reported_answer
        elif self.leg.get("type") == "compare" or self.targeted_unknown:
            final_text = reported_answer or last_actor_text or ""
        else:
            final_text = last_actor_text or response.get("text") or ""
        granted, reason = completion_predicate(
            self.leg,
            state,
            final_text=final_text,
            guard_backend=self.backend,
            guard_verdicts=guards.pickup_verdicts,
            inspect_guard=guards.inspect,
            compare_guard=self.compare_guard,
            unknown_guard=guards.unknown,
        )
        self._emit(
            "halt_request",
            step=step,
            granted=granted,
            reason=reason,
            completion_guard=self.backend,
            guard_verdicts=guards.pickup_verdicts,
            reported_answer=(
                final_text
                if self.leg.get("type") in ("inspect", "compare")
                or self.targeted_unknown
                else None
            ),
        )
        if granted:
            self.metrics["success"] = True
            self.metrics["end_reason"] = "halt_granted"
            self.metrics["completion_evidence"] = reason
            self.metrics["reported_answer"] = reported_answer or None
            print(f"[LEG {self.leg_idx} DONE] halt granted: {reason}")
            return "granted"

        self.halt_refusals += 1
        self.metrics["halts_refused"] = self.halt_refusals
        state["last_halt_refused"] = reason
        self._clear_progress(state)
        print(
            f"[GUARD] STOP refused ({self.halt_refusals}/{HALT_REFUSAL_CAP}): {reason}"
        )
        if self._maybe_release_wrong_item(
            state, guards.pickup_verdicts, step, reason, grip_tracker
        ):
            return "corrected"
        if self.halt_refusals >= HALT_REFUSAL_CAP:
            self.metrics["halt_forced"] = True
            self.metrics["end_reason"] = "halt_forced"
            print(
                f"[GUARD] refusal cap reached — force-ending leg (halt_forced): {reason}"
            )
            return "forced"
        return "refused"

    def _maybe_release_wrong_item(self, state, verdicts, step, reason, grip_tracker):
        if (
            self.leg.get("type") != "pickup"
            or self.corrective_release_done
            or self.halt_refusals < WRONG_ITEM_RELEASE_AFTER
        ):
            return False
        released = []
        for side in mismatched_hands(
            self.leg,
            state,
            grip_tracker.start_grips,
            guard_backend=self.backend,
            guard_verdicts=verdicts,
        ):
            try:
                (_GRIP_LEFT_ if side == "left" else _GRIP_RIGHT_)()
                released.append(f"{side}:{grip_tracker.names.get(side)}")
            except Exception as error:  # noqa: BLE001
                print(
                    f"[CORRECT] release toggle failed on {side}: "
                    f"{type(error).__name__}: {error}"
                )
        if not released:
            return False

        self.corrective_release_done = True
        self.halt_refusals = 0
        self.metrics["corrective_release"] = released
        fresh = self.state_reader()
        for key in (
            "leftGrippedState",
            "rightGrippedState",
            "leftHoveredObject",
            "rightHoveredObject",
        ):
            state[key] = fresh[key]
        grip_tracker.reconcile(state)
        sides = "/".join(item.split(":", 1)[0] for item in released)
        state["last_halt_refused"] = (
            f"{reason} | SELF-CORRECTION: the wrong item was auto-released from your "
            f"{sides} hand. Do NOT stop yet - find and grab the actual target "
            f"({self.leg.get('target')!r}), then STOP."
        )
        print(
            f"[CORRECT] auto-released wrong item(s) {released}; refusal budget reset."
        )
        self._emit("corrective_release", step=step, released=released)
        return True

    def observe_after_action(self, state, last_actor_text, guards, step):
        """Evaluate completion after state reconciliation and update the model nudge."""
        if self.backend == "none":
            self._clear_progress(state)
            return CompletionObservation(
                False, "completion guard disabled; waiting for explicit STOP"
            )
        if self.targeted_pickup:
            return CompletionObservation(guards.early_met, guards.early_reason)
        if self.leg.get("type") == "inspect":
            self._clear_progress(state)
            return CompletionObservation(
                False, "inspection awaits a structured STOP answer on a fresh frame"
            )

        started = time.monotonic()
        met, reason = completion_predicate(
            self.leg,
            state,
            final_text=last_actor_text,
            guard_backend=self.backend,
            inspect_guard=guards.inspect,
            compare_guard=self.compare_guard,
            unknown_guard=guards.unknown,
        )
        latency_ms = round((time.monotonic() - started) * 1000, 1)
        verdicts = deterministic_guard_details(self.leg, state)
        for side, verdict in (verdicts or {}).items():
            verdict["latency_ms"] = latency_ms
            self._guard_event(
                step,
                "deterministic",
                verdict,
                side=side,
                sku=verdict.get("sku"),
            )
        self._record_progress(state, met, reason)
        terminal = self.goal_met_streak >= COMPLETION_BACKSTOP
        if terminal:
            self._complete_without_stop(step, reason)
        return CompletionObservation(met, reason, terminal)

    def _record_progress(self, state, met, reason):
        if met:
            self.goal_met_streak += 1
            state["goal_check"] = (
                f"MEASURED: your CURRENT GOAL appears complete — {reason}. "
                "Emit STOP to finish THIS subtask; a fresh agent handles any future goals. "
                "Do NOT keep going."
            )
        else:
            self._clear_progress(state)

    def _clear_progress(self, state):
        self.goal_met_streak = 0
        state["goal_check"] = None

    def _complete_without_stop(
        self, step, reason, backend=None, guard_verdicts=None
    ):
        self.metrics["success"] = True
        self.metrics["end_reason"] = "completed_no_stop"
        if backend is None:
            self.metrics["completion_evidence"] = reason
        prefix = "VLM completion" if backend else "completion"
        print(
            f"[LEG {self.leg_idx} DONE] {prefix} backstop: goal measurably held for "
            f"{self.goal_met_streak} steps without a STOP — ending leg (success). {reason}"
        )
        fields = {"streak": self.goal_met_streak, "reason": reason}
        if backend:
            fields.update(backend=backend, guard_verdicts=guard_verdicts)
        self._emit("completed_no_stop", step=step, **fields)
