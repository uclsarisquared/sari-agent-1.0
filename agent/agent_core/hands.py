"""Stateful simulator hand controller used by the embodied runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class HandController:
    """Track hand activation and canonical pose, emitting only real transitions."""

    active: Optional[bool] = None
    pose: Optional[str] = None

    def set_active(self, active: bool) -> None:
        if self.active == active:
            return
        from sim.env import SetHandsActive

        SetHandsActive(active)
        self.active = active
        self.pose = None

    def set_pose(self, pose: str) -> None:
        self.set_active(True)
        if self.pose == pose:
            return
        from manip.manipulation import set_hand_pose

        for side in ("left", "right"):
            arrived, reported, residual = set_hand_pose(pose, hand=side)
            if not arrived:
                logger.warning(
                    f"[hand-pose] {side} '{pose}' did not converge "
                    f"(resid={residual:.3f} m, "
                    f"reported={tuple(round(value, 3) for value in reported)}) - "
                    "frame/clamp issue?"
                )
        self.pose = pose

    def invalidate_pose(self) -> None:
        self.set_active(True)
        self.pose = None

    def restore_after_inspection(self) -> dict:
        """Restore canonical transforms and clear closed-but-empty grippers."""
        self.set_active(True)
        self.pose = None
        from sim.env import ResetHands, ToggleLeftGrip, ToggleRightGrip, TransformHands

        state = ResetHands()
        recovered_ghost_grips = []
        toggles = {"left": ToggleLeftGrip, "right": ToggleRightGrip}
        for side in ("left", "right"):
            holding_key = f"{side}HoldingItem"
            closed_key = f"{side}GripClosedState"
            if holding_key not in state or closed_key not in state:
                continue
            if state[closed_key] and not state[holding_key]:
                toggles[side]()
                recovered_ghost_grips.append(side)

        if recovered_ghost_grips:
            zero = (0, 0, 0)
            state = TransformHands(zero, zero, zero, zero)
            still_ghosted = [
                side
                for side in recovered_ghost_grips
                if state.get(f"{side}GripClosedState") and not state.get(f"{side}HoldingItem")
            ]
            if still_ghosted:
                raise RuntimeError(
                    "could not open closed-but-empty inspection hand(s): "
                    + ", ".join(still_ghosted)
                )

        self.pose = "rest"
        return {
            "restored": True,
            "recovered_ghost_grips": recovered_ghost_grips,
            "hands": {
                side: {
                    "translation": state.get(f"{side}Translation"),
                    "rotation": state.get(f"{side}Rotation"),
                    "gripped": state.get(f"{side}GrippedState"),
                    "holding_item": state.get(f"{side}HoldingItem"),
                    "grip_closed": state.get(f"{side}GripClosedState"),
                }
                for side in ("left", "right")
            },
        }

