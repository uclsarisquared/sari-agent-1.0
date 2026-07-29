"""Offline hand-state simulator for manually exercising inspection tools."""

import os
import sys


OVERHAUL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if OVERHAUL_DIR not in sys.path:
    sys.path.insert(0, OVERHAUL_DIR)

from manip import manipulation as manipulation  # noqa: E402


class SimulatedHands:
    """Minimal stateful replacement for the simulator's TransformHands call."""

    def __init__(self, left_rotation=(0, 0, 0), right_rotation=(0, 0, 0)):
        self.left_translation = [0.0, 0.0, 0.0]
        self.right_translation = [0.0, 0.0, 0.0]
        self.left_rotation = [float(v) % 360 for v in left_rotation]
        self.right_rotation = [float(v) % 360 for v in right_rotation]
        self.left_gripped = True
        self.right_gripped = True
        self.rotation_deltas = []

    def __call__(self, left_translation_delta, left_rotation_delta,
                 right_translation_delta, right_rotation_delta):
        for index in range(3):
            self.left_translation[index] += float(left_translation_delta[index])
            self.right_translation[index] += float(right_translation_delta[index])
            self.left_rotation[index] = (
                self.left_rotation[index] + float(left_rotation_delta[index])) % 360
            self.right_rotation[index] = (
                self.right_rotation[index] + float(right_rotation_delta[index])) % 360
        self.rotation_deltas.append({
            "left": tuple(float(v) for v in left_rotation_delta),
            "right": tuple(float(v) for v in right_rotation_delta),
        })
        return self.state()

    def state(self):
        return {
            "leftTranslation": tuple(round(v, 6) for v in self.left_translation),
            "rightTranslation": tuple(round(v, 6) for v in self.right_translation),
            "leftRotation": tuple(round(v, 6) for v in self.left_rotation),
            "rightRotation": tuple(round(v, 6) for v in self.right_rotation),
            "leftGrippedState": self.left_gripped,
            "rightGrippedState": self.right_gripped,
            "leftHoveredObject": "SIMULATED_LEFT_ITEM",
            "rightHoveredObject": "SIMULATED_RIGHT_ITEM",
        }

    def assert_no_z_commands(self):
        violations = [
            delta for delta in self.rotation_deltas
            if delta["left"][2] != 0 or delta["right"][2] != 0
        ]
        if violations:
            raise AssertionError(f"inspection tool commanded a Z delta: {violations}")


TOOL_CALLS = {
    "present_left_item_for_inspection": manipulation.present_left_item_for_inspection,
    "present_right_item_for_inspection": manipulation.present_right_item_for_inspection,
    "rotate_left_to_next_inspection_face": manipulation.rotate_left_to_next_inspection_face,
    "rotate_right_to_next_inspection_face": manipulation.rotate_right_to_next_inspection_face,
}


def install(left_rotation=(0, 0, 0), right_rotation=(0, 0, 0)):
    hands = SimulatedHands(left_rotation=left_rotation, right_rotation=right_rotation)
    manipulation.TransformHands = hands
    manipulation._reset_inspection_sweep("left")
    manipulation._reset_inspection_sweep("right")
    return hands


def call_tool(name):
    try:
        tool = TOOL_CALLS[name]
    except KeyError as exc:
        raise ValueError(f"unknown tool {name!r}; choose one of {sorted(TOOL_CALLS)}") from exc
    return tool(times=1)
