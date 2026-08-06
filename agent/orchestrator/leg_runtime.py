"""State construction and reconciliation for typed-leg execution."""

from dataclasses import dataclass, field
import time

from sim.env import TransformAgent, TransformHands
from agent_core.prompt_loader import load_prompt, render_prompt
from orchestrator.held_item_inspection import _inspection_macro_summary


HANDS = ("left", "right")
MODEL_STATE_DROP = {"visited_checkpoints"}
LEG_INSPECTION_PROMPT = load_prompt("orchestrator/leg_inspection")


def fresh_agent_state() -> dict:
    """Read simulator state and initialize derived per-leg channels."""
    agent_pos = TransformAgent((0, 0, 0), (0, 0, 0))
    hands_pos = TransformHands((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))
    state = {
        "translation": (0, 0, 0),
        "rotation": (0, 0, 0),
        "isColliding": False,
        "leftTranslation": (0, 0, 0),
        "leftRotation": (0, 0, 0),
        "rightTranslation": (0, 0, 0),
        "rightRotation": (0, 0, 0),
        "leftHoveredObject": "None",
        "leftGrippedState": False,
        "rightHoveredObject": "None",
        "rightGrippedState": False,
        "last_grab_failed": False,
        "last_action_blocked": False,
        "last_center": None,
        "last_checkout": None,
        "last_inspection": None,
        "last_halt_refused": None,
        "nearest_checkpoint": None,
        "goal_check": None,
        "gripped_name": None,
        "mode": "perception",
    }
    state.update(agent_pos)
    state.update(hands_pos)
    return state


def model_facing_state(state: dict) -> dict:
    """Return the state visible to the model, excluding large code-only fields."""
    view = {key: value for key, value in state.items() if key not in MODEL_STATE_DROP}
    checkout = view.get("last_checkout")
    if isinstance(checkout, dict) and "steps" in checkout:
        view["last_checkout"] = {
            key: value for key, value in checkout.items() if key != "steps"
        }
    inspection = view.get("last_inspection")
    if isinstance(inspection, dict):
        view["last_inspection"] = _inspection_macro_summary(inspection)
    return view


def normalize_leg(leg):
    return leg if isinstance(leg, dict) else {"type": "unknown", "text": str(leg)}


def build_augmented_task(leg, context="", future_legs=None):
    """Build the actor prompt for one leg without changing the typed leg itself."""
    future_legs = future_legs or []
    leg_text = leg.get("text") or ""
    parts = [render_prompt("orchestrator/leg_current_goal", LEG_TEXT=leg_text)]
    parts.append(
        f"CONTEXT FROM PREVIOUS SUBTASKS:\n{context}"
        if context
        else "CONTEXT FROM PREVIOUS SUBTASKS: None — this is the first subtask."
    )
    if leg.get("type") == "inspect":
        parts.append(LEG_INSPECTION_PROMPT)
    if future_legs:
        numbered = "\n".join(
            f"  {index + 1}. {item.get('text') if isinstance(item, dict) else item}"
            for index, item in enumerate(future_legs)
        )
        parts.append(
            render_prompt(
                "orchestrator/leg_future_goals", NUMBERED_GOALS=numbered
            )
        )
    else:
        parts.append("FUTURE GOALS: None — this is the final subtask.")
    return "\n\n".join(parts)


def initial_metrics(leg, completion_guard):
    return {
        "type": leg.get("type"),
        "text": leg.get("text") or "",
        "t_manip": None,
        "t_grip": None,
        "t_checkout": None,
        "success": False,
        "timesteps": 0,
        "llm_calls": 0,
        "errors": 0,
        "completion_guard": completion_guard,
        "halts_refused": 0,
        "halt_forced": False,
        "corrective_release": None,
        "end_reason": None,
        "completion_evidence": None,
        "reported_answer": None,
    }


@dataclass
class GripTracker:
    """Durable per-hand identity and leg-boundary grip transitions."""

    names: dict
    start_grips: set

    @classmethod
    def from_state(cls, state, carried_names=None):
        carried = carried_names if isinstance(carried_names, dict) else {}
        names = {
            side: carried.get(side) if state.get(f"{side}GrippedState") else None
            for side in HANDS
        }
        start_grips = {
            side for side in HANDS if state.get(f"{side}GrippedState")
        }
        tracker = cls(names=names, start_grips=start_grips)
        state["gripped_names"] = dict(names)
        state["gripped_name"] = None
        return tracker

    def record_grab(self, result):
        if result.get("gripped") and result.get("hovered"):
            self.names[result.get("hand") or "left"] = result["hovered"]

    def reconcile(self, state):
        for side in HANDS:
            if not state.get(f"{side}GrippedState"):
                self.names[side] = None
        self.apply_to(state)

    def apply_to(self, state):
        state["gripped_names"] = dict(self.names)
        state["gripped_name"] = " ".join(name for name in self.names.values() if name) or None
        state["new_grip_this_leg"] = any(
            state.get(f"{side}GrippedState") and side not in self.start_grips
            for side in HANDS
        )
        state["released_grip_this_leg"] = any(
            side in self.start_grips and not state.get(f"{side}GrippedState")
            for side in HANDS
        )


@dataclass
class InspectionEvidence:
    """Per-hand frames that passed the held-item label visibility gate."""

    by_hand: dict = field(default_factory=dict)

    def guard_frames(self):
        return [
            {
                "label": (
                    f"{self.by_hand[side]['sku'] or 'held item'} "
                    f"({side} hand, step {self.by_hand[side]['step']})"
                ),
                "image_b64": self.by_hand[side]["image_b64"],
            }
            for side in HANDS
            if side in self.by_hand
        ]

    def record(self, result, grip_tracker, step):
        hand = result.get("hand")
        if (
            result.get("blocked")
            or not result.get("label_visible")
            or not result.get("frame_b64")
            or hand not in HANDS
        ):
            return None
        evidence = {
            "hand": hand,
            "sku": grip_tracker.names.get(hand),
            "step": step,
            "label_legible": bool(result.get("label_legible")),
            "best_effort_read": bool(result.get("best_effort_read")),
            "image_b64": result["frame_b64"],
        }
        self.by_hand[hand] = evidence
        return evidence

    def reconcile(self, state):
        for hand in list(self.by_hand):
            if not state.get(f"{hand}GrippedState"):
                self.by_hand.pop(hand)
        state["inspection_evidence"] = [
            {key: value for key, value in self.by_hand[side].items() if key != "image_b64"}
            for side in HANDS
            if side in self.by_hand
        ]


def reconcile_after_actions(
    *,
    outcome,
    mode,
    last_inspection_result,
    store_map,
    visited,
    grip_tracker,
    inspection_evidence,
    metrics,
    started_at,
    read_state=fresh_agent_state,
):
    """Read live state and merge durable effects from the dispatched action batch."""
    state = read_state()
    state.update(
        mode=mode,
        last_action_blocked=outcome.blocked_reason,
        last_center=outcome.center_message,
        last_reach=outcome.last_reach,
        last_grab_failed=outcome.grab_failed,
    )
    if outcome.checkout_result is not None:
        state["last_checkout"] = outcome.checkout_result
        if metrics["t_checkout"] is None:
            metrics["t_checkout"] = round(time.time() - started_at, 1)
    if last_inspection_result is not None:
        state["last_inspection"] = last_inspection_result

    inspection_evidence.reconcile(state)
    near = store_map.nearest_checkpoint(
        (state["translation"][0], state["translation"][2])
    )
    state["nearest_checkpoint"] = near
    visited.add(near)
    state["visited_checkpoints"] = set(visited)

    if (
        state.get("leftGrippedState") or state.get("rightGrippedState")
    ) and metrics["t_grip"] is None:
        metrics["t_grip"] = round(time.time() - started_at, 1)
    grip_tracker.reconcile(state)
    return state, near
