"""Empirically classify the sim's TranslateAgent convention: EGOCENTRIC (body-relative, the sim
rotates the step into world space itself) vs WORLD-space (the step is added to world position
verbatim). This is the "measure, don't assume" check behind the 2026-07-22 translation fix.

Why this exists: the Python movement tools (env.move_*, explore.py, capture_walk.py) send a
translation to the sim's TranslateAgent. The sim FLIPPED that command's contract from world-space to
egocentric (SariSandboxV2 3940ce7, merged 2026-07-21) - the WebSocket dispatcher's `case
"TransformAgent"` is commented out and falls through to `case "TranslateAgent"`, which now wraps the
input in `EgocentricToWorldTranslation(...)`. Before the flip, Python pre-rotated the step by yaw;
after the flip that became a SECOND rotation, so "forward" came out rotated by the agent's yaw. The
fix makes Python send body-relative vectors. Run this against a live sim to confirm the sim is
egocentric (so the fix is right) and that env.move_forward now actually moves forward.

Requires the Unity SariSandboxV2 scene in Play mode with the WebSocket server on
ws://localhost:8080/commands and the V1 compatibility layer ON. It only makes small (<=0.5 m) moves
and restores the agent to its starting pose when done.

    python probe_translation.py
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "agent"))  # agent/ — env lives there

from sim.env import TransformAgent, move_forward, move_backward


def _pose(uri):
    s = TransformAgent((0, 0, 0), (0, 0, 0), uri)
    return s["translation"], s["rotation"]


def _rotate_to(world_yaw, uri):
    """Rotate in place to an absolute world yaw (rotation is applied as a delta)."""
    _, rot = _pose(uri)
    TransformAgent((0, 0, 0), (0, world_yaw - rot[1], 0), uri)


def _bearing(dx, dz):
    """World bearing of a displacement, same convention as mapping.angle_to_deg (0deg = +Z)."""
    return math.degrees(math.atan2(dx, dz))


def _ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def probe_heading(world_yaw, step, uri):
    """Face world_yaw, send a RAW body-relative forward step (0,0,step), measure the actual world
    displacement, and undo. Returns (moved, facing_yaw, disp_bearing, disp_len) or None if blocked.
    """
    _rotate_to(world_yaw, uri)
    pos_a, rot_a = _pose(uri)
    TransformAgent((0, 0, step), (0, 0, 0), uri)          # raw body-relative forward
    pos_b, _ = _pose(uri)
    dx, dz = pos_b[0] - pos_a[0], pos_b[2] - pos_a[2]
    disp_len = math.hypot(dx, dz)
    TransformAgent((0, 0, -step), (0, 0, 0), uri)          # undo
    if disp_len < 0.02:
        return None                                        # blocked (wall) - inconclusive here
    return (True, rot_a[1], _bearing(dx, dz), disp_len)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--uri", default="ws://localhost:8080/commands")
    ap.add_argument("--step", type=float, default=0.5, help="probe step distance (m)")
    args = ap.parse_args()

    start_pos, start_rot = _pose(args.uri)
    print(f"[probe] start pos=({start_pos[0]:.2f}, {start_pos[2]:.2f}) yaw={start_rot[1]:.1f}")
    print(f"[probe] sending RAW body-relative forward (0,0,{args.step}) at several headings; "
          f"comparing the actual world displacement bearing to the facing.\n")

    results = []
    for offset in (0.0, 90.0, 180.0, 270.0):
        world_yaw = start_rot[1] + offset
        r = probe_heading(world_yaw, args.step, args.uri)
        if r is None:
            print(f"  facing~{world_yaw % 360:6.1f}deg : blocked (<2cm move) - skipped")
            continue
        _, facing, bearing, disp_len = r
        ego_err = abs(_ang_diff(bearing, facing))     # ~0 if displacement follows facing (egocentric)
        world_err = abs(_ang_diff(bearing, 0.0))      # ~0 if displacement is world +Z regardless of facing
        results.append((facing, bearing, disp_len, ego_err, world_err))
        print(f"  facing={facing:6.1f}deg : moved {disp_len:.3f}m at bearing {bearing:6.1f}deg  "
              f"| off-from-facing={ego_err:5.1f}deg  off-from-worldZ={world_err:5.1f}deg")

    # Restore the starting pose exactly.
    _rotate_to(start_rot[1], args.uri)

    if not results:
        print("\n[probe] INCONCLUSIVE: every heading was blocked. Move the agent to open floor "
              "(e.g. mid-aisle) and re-run.")
        return

    mean_ego = sum(r[3] for r in results) / len(results)
    mean_world = sum(r[4] for r in results) / len(results)
    print(f"\n[probe] mean |disp - facing| = {mean_ego:.1f}deg ; mean |disp - world+Z| = {mean_world:.1f}deg")
    if mean_ego < mean_world and mean_ego < 15.0:
        print("[probe] => EGOCENTRIC: the sim rotates the step into world space itself. "
              "Python must send body-relative vectors (this is what the 2026-07-22 fix does). OK.")
    elif mean_world < mean_ego and mean_world < 15.0:
        print("[probe] => WORLD-SPACE: the sim adds the step to world position verbatim. The "
              "2026-07-22 egocentric fix would be WRONG for this build - revert to pre-rotating by yaw.")
    else:
        print("[probe] => UNCLEAR: neither model fits well. Inspect the per-heading rows above.")

    # Sanity-check the FIXED env.move_forward: it should now move along the facing.
    print("\n[probe] checking env.move_forward (the fixed translation tool)...")
    pos_a, rot_a = _pose(args.uri)
    move_forward(3)                                   # 3 x 0.1 m along facing
    pos_b, _ = _pose(args.uri)
    dx, dz = pos_b[0] - pos_a[0], pos_b[2] - pos_a[2]
    disp_len = math.hypot(dx, dz)
    move_backward(3)                                  # restore
    if disp_len < 0.02:
        print("  move_forward: blocked (<2cm) - point the agent at open floor and re-run to verify.")
    else:
        err = abs(_ang_diff(_bearing(dx, dz), rot_a[1]))
        verdict = "OK (moves along facing)" if err < 15.0 else f"OFF by {err:.1f}deg (still de-synced!)"
        print(f"  move_forward: {disp_len:.3f}m at bearing {_bearing(dx, dz):.1f}deg vs facing "
              f"{rot_a[1]:.1f}deg -> {verdict}")

    _rotate_to(start_rot[1], args.uri)
    end_pos, _ = _pose(args.uri)
    print(f"\n[probe] restored to pos=({end_pos[0]:.2f}, {end_pos[2]:.2f}) yaw={start_rot[1]:.1f}")


if __name__ == "__main__":
    main()
