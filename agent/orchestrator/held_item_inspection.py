"""Deterministic held-item inspection actions and evidence capture."""

import base64
import os

from sim.env import _REQUEST_SCREENSHOT_, TransformHands, downscale_for_storage
from toolset.actions import MANIPULATION_ACTIONS_REF
from orchestrator.pickup_vlm_guard import (
    classify_inspection_label_presence,
    classify_inspection_visibility,
)

_INSPECT_MACRO_ACTIONS = {
    "inspect_held_item": "auto",
    "inspect_held_item_left": "left",
    "inspect_held_item_right": "right",
}
_RESTRICTED_INSPECTION_TURNS = (
    tuple(("x", index, (45.0, 0.0, 0.0)) for index in range(1, 9))
    + (
        ("y_top", 1, (0.0, 90.0, 0.0)),
        ("y_default", 1, (0.0, -90.0, 0.0)),
        ("y_bottom", 1, (0.0, -90.0, 0.0)),
    )
)
_INSPECTION_MAX_PASSES = 5
_INSPECTION_CLOSER_STEP_M = 0.05
_INSPECTION_OCR_CROP_WIDTH_FRAC = 0.70
_INSPECTION_OCR_CROP_HEIGHT_FRAC = 0.90
# A complete pass ends 90 degrees below its starting orientation: the eight X turns total 360,
# while the Y sequence totals -90. Apply this once between passes to repeat the same sweep.
_INSPECTION_PASS_RESET_DELTA = (0.0, 90.0, 0.0)


def _inspection_rotation_delta(hand, logical_delta):
    """Map a camera-relative inspection turn onto the selected hand's local Euler axes.

    Unity's right-hand transform has its local X/Y inspection axes transposed relative to the
    left hand. Keep the sweep plan camera-relative (X sides first, then Y top/bottom), and swap
    those axes only when issuing a right-hand TransformHands command.
    """
    delta = tuple(logical_delta)
    if str(hand).lower() == "right":
        return delta[1], delta[0], delta[2]
    return delta


def _inspection_action_batch(actions, times):
    """Make the restricted held-item inspection macro the only call in its timestep.

    The actor never sequences rotations itself. If it selects the macro, discard every other proposed
    action; the macro owns presentation, turns, screenshots, and visibility checks.
    """
    batch = list(zip(actions or [], times or []))
    for action, time_units in batch:
        base_action = str(action or "").strip().split("(", 1)[0]
        if base_action in _INSPECT_MACRO_ACTIONS:
            return [(action, time_units)]
    return batch


# Macro-result keys that must never reach a log row or the model prompt: `steps` is per-primitive
# logging, `frame_b64` is a full-resolution screenshot the completion guard consumes in code. Both
# sinks (the macro's own log rows and _model_facing_state) go through the helper below.
_INSPECTION_RESULT_DROP = ("steps", "frame_b64")


def _inspection_macro_summary(result: dict) -> dict:
    """The loggable/model-facing view of an inspection macro result."""
    return {k: v for k, v in result.items() if k not in _INSPECTION_RESULT_DROP}


def _run_held_item_inspection_macro(
        agent, query, state, log_event=None, frames_dir=None, hand="auto"):
    """Deterministically sweep a held item until a fresh VLM context finds the requested label.

    A non-blocked result that found the label carries `frame_b64`: the exact frame whose visibility
    gate passed. run_leg files it in the leg's inspection evidence ledger so a multi-item inspect
    can be verified across frames (see subtask_completion.inspection_evidence_gap); nothing else
    reads it, and _inspection_macro_summary keeps it out of logs and the prompt.
    """
    held_sides = [
        side for side in ("left", "right")
        if isinstance(state, dict) and state.get(f"{side}GrippedState")
    ]
    requested_hand = str(hand or "auto").strip().lower()
    if requested_hand not in ("auto", "left", "right"):
        return {
            "blocked": True,
            "executed": False,
            "reason": "inspect_held_item hand must be 'auto', 'left', or 'right'",
            "vlm_calls": 0,
        }
    if not held_sides:
        return {
            "blocked": True,
            "executed": False,
            "reason": "inspect_held_item requires a held item",
            "vlm_calls": 0,
        }
    if requested_hand != "auto" and requested_hand not in held_sides:
        return {
            "blocked": True,
            "executed": False,
            "hand": requested_hand,
            "reason": (
                f"inspect_held_item_{requested_hand} requires an item held "
                f"in the {requested_hand} hand"
            ),
            "vlm_calls": 0,
        }
    if agent is None:
        return {
            "blocked": True,
            "executed": False,
            "reason": "inspect_held_item requires the active agent VLM client",
            "vlm_calls": 0,
        }
    query = str(query or "").strip()
    if not query:
        return {
            "blocked": True,
            "executed": False,
            "reason": "inspect_held_item requires the inspection request from the current leg",
            "vlm_calls": 0,
        }

    hand = held_sides[0] if requested_hand == "auto" else requested_hand
    emit = log_event if callable(log_event) else lambda row: None

    def finish_failure(result):
        """Restore REST immediately when no evidence frame needs to remain presented."""
        try:
            restore = getattr(agent, "restore_hands_after_inspection", None)
            restore = restore or getattr(agent, "_restore_hands_after_inspection", None)
            if callable(restore):
                cleanup = restore()
            else:
                from sim.env import ResetHands
                reset_state = ResetHands()
                cleanup = {
                    "restored": True,
                    "hands": {
                        side: {
                            "translation": reset_state.get(f"{side}Translation"),
                            "rotation": reset_state.get(f"{side}Rotation"),
                            "gripped": reset_state.get(f"{side}GrippedState"),
                        }
                        for side in ("left", "right")
                    },
                }
        except Exception as exc:  # noqa: BLE001 - report cleanup failure without hiding sweep result
            cleanup = {
                "restored": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        result["failure_cleanup"] = cleanup
        emit({
            "event": "inspection_failure_cleanup",
            "hand": hand,
            **cleanup,
        })
        emit({"event": "inspection_macro_end", **_inspection_macro_summary(result)})
        return result

    if frames_dir:
        os.makedirs(frames_dir, exist_ok=True)
    emit({
        "event": "inspection_macro_start",
        "hand": hand,
        "query": query,
        "turn_plan": [
            {
                "phase": phase,
                "index": index,
                "logical_delta": delta,
                "commanded_delta": _inspection_rotation_delta(hand, delta),
            }
            for phase, index, delta in _RESTRICTED_INSPECTION_TURNS
        ],
    })

    # Inspection rotations are relative, so always begin from Unity's canonical hand transforms.
    # Reset BOTH hands before presenting the selected item; this prevents translation/rotation left
    # behind by a prior manipulation from becoming the inspection sweep's starting orientation.
    try:
        restore = getattr(agent, "restore_hands_after_inspection", None)
        restore = restore or getattr(agent, "_restore_hands_after_inspection", None)
        if callable(restore):
            pre_reset = restore()
        else:
            from sim.env import ResetHands
            reset_state = ResetHands()
            pre_reset = {
                "restored": True,
                "hands": {
                    side: {
                        "translation": reset_state.get(f"{side}Translation"),
                        "rotation": reset_state.get(f"{side}Rotation"),
                        "gripped": reset_state.get(f"{side}GrippedState"),
                    }
                    for side in ("left", "right")
                },
            }
    except Exception as exc:  # noqa: BLE001 - report a clean blocked macro result
        pre_reset = {
            "restored": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    emit({"event": "inspection_pre_reset", "hand": hand, **pre_reset})
    if not pre_reset.get("restored"):
        result = {
            "blocked": True,
            "executed": False,
            "hand": hand,
            "label_visible": False,
            "sweep_exhausted": False,
            "vlm_calls": 0,
            "reason": "could not reset the hands before inspection",
            "pre_inspection_reset": pre_reset,
            "steps": [],
        }
        emit({"event": "inspection_macro_end", **_inspection_macro_summary(result)})
        return result

    present_action = f"present_{hand}_item_for_inspection"
    presentation = MANIPULATION_ACTIONS_REF[present_action](1) or {}
    emit({
        "event": "inspection_presentation",
        "hand": hand,
        "action": present_action,
        "result": presentation,
    })
    if presentation.get("blocked") or presentation.get("arrived") is False:
        result = {
            "blocked": True,
            "executed": True,
            "hand": hand,
            "label_visible": False,
            "sweep_exhausted": False,
            "vlm_calls": 0,
            "reason": presentation.get("reason") or "inspection presentation did not converge",
            "steps": [],
        }
        return finish_failure(result)

    steps = []
    vlm_calls = 0

    def check_visible(check_index, pass_index, phase, turn_index, rotation_delta=None,
                      image_bytes=None, ocr_text=None):
        """Capture and verify that the held item is visibly presented for inspection."""
        nonlocal vlm_calls
        if image_bytes is None:
            image_bytes = _REQUEST_SCREENSHOT_()["image"]
        frame_path = None
        if frames_dir:
            frame_path = os.path.join(frames_dir, f"check{check_index:02d}_{phase}.png")
            with open(frame_path, "wb") as frame_file:
                frame_file.write(downscale_for_storage(image_bytes))
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        classify_kwargs = (
            {"ocr_lines": ocr_text} if ocr_text is not None else {})
        verdict = classify_inspection_visibility(
            agent.vlm_agent.client,
            agent.vlm_agent.config.model_id,
            agent.vlm_agent.config,
            image_b64,
            query,
            **classify_kwargs,
        )
        vlm_calls += 1
        row = {
            "check_index": check_index,
            "pass_index": pass_index,
            "phase": phase,
            "turn_index": turn_index,
            "rotation_delta": rotation_delta,
            "match": verdict.get("match"),
            "conclusive": verdict.get("conclusive"),
            "reason": verdict.get("reason"),
            "latency_ms": verdict.get("latency_ms"),
            "frame": frame_path,
        }
        steps.append(row)
        emit({"event": "inspection_visibility_check", "hand": hand, "query": query, **row})
        return verdict, image_b64, image_bytes

    def ocr_locked_frame(image_bytes, pass_index, phase):
        """OCR only the centered held-item region after the front-facing label side is locked."""
        try:
            from io import BytesIO
            from PIL import Image
            from vision.ocr_client import ocr_lines

            with Image.open(BytesIO(image_bytes)) as opened:
                opened.load()
                image = opened.convert("RGB")
                width, height = image.size
                crop_width = max(1, round(width * _INSPECTION_OCR_CROP_WIDTH_FRAC))
                crop_height = max(1, round(height * _INSPECTION_OCR_CROP_HEIGHT_FRAC))
                left = max(0, (width - crop_width) // 2)
                top = max(0, (height - crop_height) // 2)
                crop = image.crop((left, top, left + crop_width, top + crop_height))
                lines = ocr_lines(crop)
            result = {"lines": list(lines), "error": None}
        except Exception as exc:  # OCR is auxiliary; a VLM inspection must remain usable without it.
            result = {
                "lines": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        emit({
            "event": "inspection_ocr",
            "hand": hand,
            "pass_index": pass_index,
            "phase": phase,
            "crop_width_frac": _INSPECTION_OCR_CROP_WIDTH_FRAC,
            "crop_height_frac": _INSPECTION_OCR_CROP_HEIGHT_FRAC,
            **result,
        })
        return result

    def check_label_presence(image_b64, check_index, pass_index, phase, turn_index):
        """Verify that a visible inspection frame contains a readable product label."""
        nonlocal vlm_calls
        verdict = classify_inspection_label_presence(
            agent.vlm_agent.client,
            agent.vlm_agent.config.model_id,
            agent.vlm_agent.config,
            image_b64,
            query,
        )
        vlm_calls += 1
        if steps:
            steps[-1]["label_present"] = verdict.get("match")
            steps[-1]["label_presence_reason"] = verdict.get("reason")
        emit({
            "event": "inspection_label_presence_check",
            "hand": hand,
            "query": query,
            "check_index": check_index,
            "pass_index": pass_index,
            "phase": phase,
            "turn_index": turn_index,
            "match": verdict.get("match"),
            "conclusive": verdict.get("conclusive"),
            "reason": verdict.get("reason"),
            "latency_ms": verdict.get("latency_ms"),
        })
        return verdict

    def lock_side_and_approach(pass_index, check_index, phase, presence_verdict,
                               locked_image_bytes):
        """Keep the detected label-facing orientation and spend remaining stages moving closer."""
        nonlocal vlm_calls
        emit({
            "event": "inspection_label_locked",
            "hand": hand,
            "pass_index": pass_index,
            "phase": phase,
            "reason": presence_verdict.get("reason"),
        })
        from manip.manipulation import INSPECTION_POSE, set_hand_pose
        initial_ocr = ocr_locked_frame(locked_image_bytes, pass_index, phase)
        best_ocr_lines = list(initial_ocr["lines"])
        latest_ocr_lines = list(initial_ocr["lines"])
        # The most recent locked frame, kept current as the hand moves closer: whichever frame this
        # branch finally returns on is the evidence frame the completion guard replays.
        locked_b64 = base64.b64encode(locked_image_bytes).decode("utf-8")

        # Re-run the strict gate on this exact locked frame with PaddleOCR as untrusted auxiliary
        # text. This can confirm legibility without moving, but cannot relax the front-facing test.
        if latest_ocr_lines:
            ocr_verdict = classify_inspection_visibility(
                agent.vlm_agent.client,
                agent.vlm_agent.config.model_id,
                agent.vlm_agent.config,
                locked_b64,
                query,
                ocr_lines=latest_ocr_lines,
            )
            vlm_calls += 1
            emit({
                "event": "inspection_ocr_legibility_check",
                "hand": hand,
                "pass_index": pass_index,
                "phase": phase,
                "ocr_lines": latest_ocr_lines,
                **ocr_verdict,
            })
            if ocr_verdict.get("match") and ocr_verdict.get("conclusive"):
                result = {
                    "blocked": False,
                    "executed": True,
                    "hand": hand,
                    "label_visible": True,
                    "label_legible": True,
                    "label_locked": True,
                    "best_effort_read": False,
                    "sweep_exhausted": False,
                    "visible_phase": phase,
                    "visible_pass": pass_index,
                    "checks": check_index,
                    "vlm_calls": vlm_calls,
                    "ocr_lines": latest_ocr_lines,
                    "reason": ocr_verdict.get("reason"),
                    "frame_b64": locked_b64,
                    "steps": steps,
                }
                emit({"event": "inspection_macro_end", **_inspection_macro_summary(result)})
                return result

        for closer_stage in range(pass_index + 1, _INSPECTION_MAX_PASSES + 1):
            closer_pose = (
                INSPECTION_POSE[0],
                INSPECTION_POSE[1],
                INSPECTION_POSE[2]
                - (_INSPECTION_CLOSER_STEP_M * (closer_stage - 1)),
            )
            arrived, translation, residual = set_hand_pose(
                closer_pose, hand=hand, max_iters=5)
            emit({
                "event": "inspection_reposition",
                "hand": hand,
                "label_locked": True,
                "completed_pass": closer_stage - 1,
                "next_pass": closer_stage,
                "closer_by_m": round(
                    _INSPECTION_CLOSER_STEP_M * (closer_stage - 1), 3),
                "target_translation": closer_pose,
                "reported_translation": translation,
                "translation_residual": residual,
                "arrived": arrived,
            })
            if not arrived:
                result = {
                    "blocked": True,
                    "executed": True,
                    "hand": hand,
                    "label_visible": True,
                    "label_legible": False,
                    "label_locked": True,
                    "sweep_exhausted": False,
                    "passes_completed": closer_stage - 1,
                    "checks": check_index,
                    "vlm_calls": vlm_calls,
                    "reason": f"could not move the {hand} hand closer to the detected label",
                    "steps": steps,
                }
                return finish_failure(result)

            closer_image = _REQUEST_SCREENSHOT_()["image"]
            current_ocr = ocr_locked_frame(
                closer_image, closer_stage, "locked_closer")
            latest_ocr_lines = list(current_ocr["lines"])
            if sum(map(len, latest_ocr_lines)) > sum(map(len, best_ocr_lines)):
                best_ocr_lines = list(latest_ocr_lines)

            check_index += 1
            legible, locked_b64, _image_bytes = check_visible(
                check_index, closer_stage, "locked_closer", closer_stage,
                image_bytes=closer_image, ocr_text=latest_ocr_lines)
            if legible.get("match") and legible.get("conclusive"):
                result = {
                    "blocked": False,
                    "executed": True,
                    "hand": hand,
                    "label_visible": True,
                    "label_legible": True,
                    "label_locked": True,
                    "best_effort_read": False,
                    "sweep_exhausted": False,
                    "visible_phase": phase,
                    "visible_pass": closer_stage,
                    "checks": check_index,
                    "vlm_calls": vlm_calls,
                    "ocr_lines": latest_ocr_lines or best_ocr_lines,
                    "reason": legible.get("reason"),
                    "frame_b64": locked_b64,
                    "steps": steps,
                }
                emit({"event": "inspection_macro_end", **_inspection_macro_summary(result)})
                return result

        # The target label is still facing the camera at the closest allowed position. Preserve that
        # frame and explicitly hand it to the actor for a best-effort transcription.
        result = {
            "blocked": False,
            "executed": True,
            "hand": hand,
            "label_visible": True,
            "label_legible": False,
            "label_locked": True,
            "best_effort_read": True,
            "sweep_exhausted": False,
            "visible_phase": phase,
            "visible_pass": _INSPECTION_MAX_PASSES,
            "passes_completed": _INSPECTION_MAX_PASSES,
            "checks": check_index,
            "vlm_calls": vlm_calls,
            "ocr_lines": latest_ocr_lines or best_ocr_lines,
            "reason": (
                "requested label is facing the camera at the closest inspection position but "
                "remains illegible; attempt a best-effort read from the current frame"
            ),
            "frame_b64": locked_b64,
            "steps": steps,
        }
        emit({"event": "inspection_macro_end", **_inspection_macro_summary(result)})
        return result

    check_index = 0
    zero = (0, 0, 0)
    for pass_index in range(1, _INSPECTION_MAX_PASSES + 1):
        check_index += 1
        verdict, image_b64, image_bytes = check_visible(
            check_index, pass_index, "initial", 0)
        if verdict.get("match") and verdict.get("conclusive"):
            result = {
                "blocked": False,
                "executed": True,
                "hand": hand,
                "label_visible": True,
                "label_legible": True,
                "best_effort_read": False,
                "sweep_exhausted": False,
                "visible_phase": "initial",
                "visible_pass": pass_index,
                "checks": check_index,
                "vlm_calls": vlm_calls,
                "reason": verdict.get("reason"),
                "frame_b64": image_b64,
                "steps": steps,
            }
            emit({"event": "inspection_macro_end", **_inspection_macro_summary(result)})
            return result
        presence = check_label_presence(
            image_b64, check_index, pass_index, "initial", 0)
        if presence.get("match") and presence.get("conclusive"):
            return lock_side_and_approach(
                pass_index, check_index, "initial", presence, image_bytes)

        for phase, turn_index, rotation_delta in _RESTRICTED_INSPECTION_TURNS:
            commanded_rotation_delta = _inspection_rotation_delta(hand, rotation_delta)
            if hand == "left":
                turn_state = TransformHands(zero, commanded_rotation_delta, zero, zero)
            else:
                turn_state = TransformHands(zero, zero, zero, commanded_rotation_delta)
            emit({
                "event": "inspection_rotation",
                "hand": hand,
                "pass_index": pass_index,
                "phase": phase,
                "turn_index": turn_index,
                "logical_rotation_delta": rotation_delta,
                "commanded_rotation_delta": commanded_rotation_delta,
                "reported_rotation": turn_state.get(f"{hand}Rotation"),
            })
            check_index += 1
            verdict, image_b64, image_bytes = check_visible(
                check_index, pass_index, phase, turn_index, commanded_rotation_delta)
            if verdict.get("match") and verdict.get("conclusive"):
                result = {
                    "blocked": False,
                    "executed": True,
                    "hand": hand,
                    "label_visible": True,
                    "label_legible": True,
                    "best_effort_read": False,
                    "sweep_exhausted": False,
                    "visible_phase": phase,
                    "visible_pass": pass_index,
                    "checks": check_index,
                    "vlm_calls": vlm_calls,
                    "reason": verdict.get("reason"),
                    "frame_b64": image_b64,
                    "steps": steps,
                }
                emit({"event": "inspection_macro_end", **_inspection_macro_summary(result)})
                return result
            presence = check_label_presence(
                image_b64, check_index, pass_index, phase, turn_index)
            if presence.get("match") and presence.get("conclusive"):
                return lock_side_and_approach(
                    pass_index, check_index, phase, presence, image_bytes)

        if pass_index < _INSPECTION_MAX_PASSES:
            # Restore this pass's relative start orientation without ResetHands (which would visibly
            # flash through REST), then bring the held item 5 cm closer for the next full sweep.
            pass_reset_delta = _inspection_rotation_delta(
                hand, _INSPECTION_PASS_RESET_DELTA)
            if hand == "left":
                reset_state = TransformHands(
                    zero, pass_reset_delta, zero, zero)
            else:
                reset_state = TransformHands(
                    zero, zero, zero, pass_reset_delta)
            from manip.manipulation import INSPECTION_POSE, set_hand_pose
            closer_pose = (
                INSPECTION_POSE[0],
                INSPECTION_POSE[1],
                INSPECTION_POSE[2] - (_INSPECTION_CLOSER_STEP_M * pass_index),
            )
            arrived, translation, residual = set_hand_pose(
                closer_pose, hand=hand, max_iters=5)
            emit({
                "event": "inspection_reposition",
                "hand": hand,
                "completed_pass": pass_index,
                "next_pass": pass_index + 1,
                "closer_by_m": round(_INSPECTION_CLOSER_STEP_M * pass_index, 3),
                "target_translation": closer_pose,
                "reported_translation": translation,
                "translation_residual": residual,
                "reported_rotation": reset_state.get(f"{hand}Rotation"),
                "arrived": arrived,
            })
            if not arrived:
                result = {
                    "blocked": True,
                    "executed": True,
                    "hand": hand,
                    "label_visible": False,
                    "sweep_exhausted": False,
                    "passes_completed": pass_index,
                    "checks": check_index,
                    "vlm_calls": vlm_calls,
                    "reason": f"could not move the {hand} hand closer for inspection retry",
                    "steps": steps,
                }
                return finish_failure(result)

    result = {
        "blocked": False,
        "executed": True,
        "hand": hand,
        "label_visible": False,
        "sweep_exhausted": True,
        "visible_phase": None,
        "passes_completed": _INSPECTION_MAX_PASSES,
        "checks": check_index,
        "vlm_calls": vlm_calls,
        "reason": "five inspection sweeps exhausted without a legible target label",
        "steps": steps,
    }
    return finish_failure(result)
