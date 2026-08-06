"""step0_hand_pose_probe.py - Phase 6.1 STEP 0: measure-before-you-build the hand-at-REST idea.

Phase 6.1 wants hands ACTIVE at all times during a task (so a carried item survives navigation),
with the left hand parked at a named REST pose instead of being disabled. Before ANY of that gets
wired into agent.py, the plan (mapping/plans/phase6/phase6_long_horizon_tasks.md, "Things to
MEASURE before building on this") lists four things that have each burned this project when assumed.
This probe answers all four against the live sim - it changes NOTHING in the agent; it only measures.

Needs the sim in Play mode on ws://localhost:8080.

    python validation/calibration/step0_hand_pose_probe.py

The four measurements (each maps to commands below):
  M1  Does REST keep the hand OUT of the camera frame? (occlusion was the whole reason hands were
      disabled 2026-07-19.) -> `m1` runs the A/B: hand-at-REST shot vs hands-disabled shot, same pose.
      Do it BOTH at a shelf and in an aisle. If the resting hand intrudes, REST needs re-picking
      before anything else proceeds.
  M2  Does REST survive LiDAR? (the clearance gate must not see the resting hand as an obstacle, or
      navigation freezes.) -> `m2` reads swept_clearance_ahead with the hand at REST vs hands-off.
      Equal clearance => the hand is self-culled and REST is safe to navigate with.
  M3  Does a GRIPPED item survive a hand move AND a full checkpoint drive? -> `grab`, then drive
      around: every `w/s/a/d/...` move and every `drive <cp>` re-checks leftGrippedState and logs it.
      Zero drops across a multi-checkpoint route is the bar.
  M4  Does a CARRIED item occlude the camera or the LiDAR centre ray? (an empty hand may be culled
      while a box in it is not.) -> after `grab`, `lidar` reports the centre ray and every move saves
      an occlusion screenshot. Compare centre-ray distance empty vs holding, and eyeball the shots.
      If the item is in frame, RECORD it as a known cost (the VLM sees the carried item - arguably a
      feature), don't try to hide it.

REST / GRAB poses are the user's manual calibration (2026-07-22), left-hand agent-local xyz:
    REST = (-0.213, -0.09, 0.26)    GRAB = (-0.01, 0.006, 0.33)
This probe DOES NOT commit these anywhere - step 1 (set_hand_pose in agent.py) does. Here they are
just the pose the closed-loop `set_hand_pose` below drives to, so we can SEE whether a resting hand
is a workable idea at all. If the closed loop can't reach REST (reported translation won't converge),
that itself is a finding: the pose values are in a different frame than assumed - report it.

Commands at the `>` prompt (a trailing number is a repeat count, default 1):
  rest          drive the left hand to REST, screenshot, print the reported hand translation
  grab [label]  set GRAB pose, extend_arm_until_grabbed (raw), then return to REST carrying the item
  x             release the left grip (drop the held item)
  m1            M1 A/B: shot at REST, then hands-off shot, then hands back on at REST (do @ shelf+aisle)
  m2            M2: swept clearance at REST vs hands-off - equal => hand is LiDAR self-culled
  clear         one swept_clearance_ahead read straight ahead from the current pose
  lidar         one RequestLidarCenter read (centre-ray distance/hit/pitch/cam height)
  grip?         print leftGrippedState / leftHoveredObject / hand translation right now
  w s a d N     move forward/back/left/right N steps (0.1 m), then auto carry-check (grip+lidar+shot)
  j l i k N     pan left/right / tilt up/down N (2.5 deg), then auto carry-check
  drive <cp>    checkpoint drive to cp <cp> via NavSession (hands NOT stowed), then carry-check (M3)
  shot [label]  save a screenshot with an optional label
  off | on      disable | enable the hand prefabs (SetHandsActive) - the M1 A/B primitives
  note <text>   append a free-text line to notes.md (record what you SAW)
  ?             reprint this help
  q             quit (release grip, stand up, hands active at REST)

Everything measured is appended under validation/artifacts/calibration/hand_pose/<run>/;
screenshots land in the
same run dir; notes.md is seeded with the four questions for you to fill in as you go.
"""
import json
import math
import os
import sys
from datetime import datetime

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "agent")                      # agent/
_MAPPING = os.path.join(_ROOT, "mapping")
for _p in (_ROOT, _MAPPING):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import _bootstrap  # noqa: F401,E402 - mapping category dirs onto sys.path (flat-import contract)

# Left-hand agent-local poses (user calibration 2026-07-22). Local, not world: |GRAB| ~= 0.33 m.
REST_LOCAL = (-0.213, -0.09, 0.26)
GRAB_LOCAL = (-0.01, 0.006, 0.33)
HAND_MOVE_RANGE = 0.5      # Unity clamps a single TransformHands delta component at this (TranslateHand)
POSE_TOL = 0.012           # m; "arrived" when the reported translation is within this of the target

HELP = "Commands at" + __doc__.split("Commands at", 1)[1].split("Everything measured", 1)[0]


def _norm(v):
    return math.sqrt(sum(c * c for c in v))


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Probe:
    def __init__(self):
        from sim.env import (RequestScreenshot, TransformHands, TransformAgent, SetHandsActive,
                         RequestLidarCenter, ToggleLeftGrip, move_forward, move_backward,
                         move_left, move_right, pan_left, pan_right, tilt_up, tilt_down)
        from manip.manipulation import extend_arm_until_grabbed
        from lidar_client import RequestLidarScan
        from mapping import swept_clearance_ahead

        self.env = dict(
            RequestScreenshot=RequestScreenshot, TransformHands=TransformHands,
            TransformAgent=TransformAgent, SetHandsActive=SetHandsActive,
            RequestLidarCenter=RequestLidarCenter, ToggleLeftGrip=ToggleLeftGrip,
            extend_arm_until_grabbed=extend_arm_until_grabbed,
            RequestLidarScan=RequestLidarScan, swept_clearance_ahead=swept_clearance_ahead,
        )
        self.MOVES = {"w": move_forward, "s": move_backward, "a": move_left, "d": move_right,
                      "j": pan_left, "l": pan_right, "i": tilt_up, "k": tilt_down}

        self.run_dir = os.path.join(
            os.path.dirname(_ROOT), "validation", "artifacts", "calibration", "hand_pose",
            datetime.now().strftime("%m%d_%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)
        self.jsonl = os.path.join(self.run_dir, "measurements.jsonl")
        self.notes = os.path.join(self.run_dir, "notes.md")
        self._seed_notes()

        self.idx = 0
        self._hands_active = True
        self._nav = None          # lazy NavSession, only if `drive` is used

    # ---- artifacts ----------------------------------------------------------------------------
    def _seed_notes(self):
        with open(self.notes, "w", encoding="utf-8") as f:
            f.write(
                "# Phase 6.1 step 0 - hand-pose measurements\n\n"
                "REST = (-0.213, -0.09, 0.26)   GRAB = (-0.01, 0.006, 0.33)   (left-hand agent-local)\n\n"
                "## M1 - Does REST keep the hand out of the camera frame?\n"
                "- at a SHELF:  \n- in an AISLE:  \n- verdict (REST usable? re-pick pose?):  \n\n"
                "## M2 - Does REST survive LiDAR (clearance gate)?\n"
                "- clearance @ REST vs hands-off:  \n- verdict (self-culled?):  \n\n"
                "## M3 - Does a gripped item survive moves + a full checkpoint drive?\n"
                "- route driven:  \n- drops:  \n- verdict:  \n\n"
                "## M4 - Does a carried item occlude the camera / LiDAR centre ray?\n"
                "- centre-ray dist empty vs holding:  \n- item in frame?:  \n- verdict (known cost?):  \n")

    def log(self, kind, **fields):
        self.idx += 1
        row = {"idx": self.idx, "ts": datetime.now().isoformat(timespec="seconds"),
               "kind": kind, **fields}
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def shot(self, label):
        """Save a named screenshot; returns the basename or an error tag (never raises)."""
        name = f"{self.idx + 1:03d}_{label}.png"
        try:
            res = self.env["RequestScreenshot"](save_image=False)
            with open(os.path.join(self.run_dir, name), "wb") as f:
                f.write(res["image"])
            return name
        except Exception as e:
            return f"(no shot: {type(e).__name__})"

    # ---- hand state ---------------------------------------------------------------------------
    def hand_state(self):
        return self.env["TransformHands"]((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0))

    def set_hand_pose(self, target, max_iters=5):
        """Closed-loop drive the LEFT hand to `target` (agent-local xyz), reading the reported
        translation and nudging by the residual each iteration. Closed-loop on purpose:

          - it self-corrects Unity's per-call clamp (handMoveRange = 0.5 on each delta component):
            a large move that would be clamped just takes another iteration ("split into two calls",
            as the plan flags), so a single logical set_hand_pose still arrives.
          - it is the honest test of the pose's coordinate frame: if the reported translation does
            NOT march toward `target`, the residual won't shrink and this returns un-converged - which
            is the finding "the calibrated pose is in a different frame than assumed", not a crash.

        Returns (arrived: bool, reported_translation, residual_magnitude)."""
        cur = self.hand_state()["leftTranslation"]
        for _ in range(max_iters):
            err = [target[k] - cur[k] for k in range(3)]
            mag = _norm(err)
            if mag <= POSE_TOL:
                return True, cur, mag
            delta = tuple(_clamp(err[k], -(HAND_MOVE_RANGE - 1e-3), HAND_MOVE_RANGE - 1e-3)
                          for k in range(3))
            st = self.env["TransformHands"](delta, (0, 0, 0), (0, 0, 0), (0, 0, 0))
            cur = st["leftTranslation"]
        return _norm([target[k] - cur[k] for k in range(3)]) <= POSE_TOL, cur, \
            _norm([target[k] - cur[k] for k in range(3)])

    # ---- carry check (M3 + M4 per leg) --------------------------------------------------------
    def carry_check(self, context):
        """After any motion while (maybe) carrying: is the grip still closed, does the hand+item
        still read sane, and what does the centre ray see? One screenshot per leg for the M4 eyeball.
        This is the per-leg assertion the plan's Gate 6.1 will formalise; here it just measures."""
        hs = self.hand_state()
        gripped = bool(hs.get("leftGrippedState"))
        hovered = hs.get("leftHoveredObject")
        hand_t = hs.get("leftTranslation")
        try:
            c = self.env["RequestLidarCenter"]()
        except Exception as e:
            c = {"error": f"{type(e).__name__}: {e}"}
        shot = self.shot(f"carry_{context}")
        self.log("carry_check", context=context, gripped=gripped, hovered=hovered,
                 hand_translation=list(hand_t) if hand_t else None,
                 center_distance=c.get("distance"), center_hit=c.get("hit"),
                 center_pitch=c.get("pitch_deg"), camera_height=c.get("camera_height"), shot=shot)
        drop = "" if gripped else "   <-- GRIP LOST (drop!)"
        print(f"  [carry:{context}] grip={gripped} hovered={hovered!r} "
              f"center_dist={c.get('distance')} hit={c.get('hit')}  [{shot}]{drop}")
        return gripped

    # ---- measurements -------------------------------------------------------------------------
    def m_rest(self):
        arrived, cur, resid = self.set_hand_pose(REST_LOCAL)
        shot = self.shot("rest")
        self.log("set_rest", arrived=arrived, target=list(REST_LOCAL),
                 reported=list(cur), residual=round(resid, 4), shot=shot)
        tag = "" if arrived else f"   <-- did NOT converge (resid={resid:.3f} m; frame mismatch?)"
        print(f"  [rest] reported hand={tuple(round(v, 3) for v in cur)} "
              f"target={REST_LOCAL} resid={resid:.3f}m  [{shot}]{tag}")

    def m_grab(self, label):
        if not self._hands_active:
            self.env["SetHandsActive"](True); self._hands_active = True
        arrived, _, _ = self.set_hand_pose(GRAB_LOCAL)
        res = self.env["extend_arm_until_grabbed"]()
        gripped = bool(res.get("gripped"))
        # Return to REST carrying the item - the parented item should ride along (that's M3).
        _, cur, resid = self.set_hand_pose(REST_LOCAL)
        shot = self.shot(f"grab_{label or 'item'}")
        self.log("grab", label=label, grab_pose_arrived=arrived, gripped=gripped,
                 hovered=res.get("hovered"), reason=res.get("reason"),
                 rest_after=list(cur), rest_residual=round(resid, 4), shot=shot)
        flag = "GRABBED" if gripped else "missed"
        print(f"  [grab] {flag} hovered={res.get('hovered')!r} -> back at REST "
              f"(resid={resid:.3f}m)  [{shot}]"
              + (f"\n    reason: {res['reason']}" if res.get("reason") else ""))
        if gripped:
            print("    now drive around (w/s/a/d, drive <cp>) - grip should survive every leg (M3).")

    def m_release(self):
        self.env["ToggleLeftGrip"]()
        hs = self.hand_state()
        self.log("release", gripped_after=bool(hs.get("leftGrippedState")))
        print(f"  [x] released; leftGrippedState now {bool(hs.get('leftGrippedState'))}")

    def _clearance(self):
        scan = self.env["RequestLidarScan"]()
        clearance, dbg = self.env["swept_clearance_ahead"](scan, 0.0, debug=True)
        return clearance, dbg

    def m_clear(self):
        try:
            clearance, _ = self._clearance()
        except Exception as e:
            print(f"  [clear] LiDAR scan failed: {type(e).__name__}: {e}"); return
        self.log("clearance", hands_active=self._hands_active, clearance=clearance)
        print(f"  [clear] swept clearance ahead = {clearance:.3f} m "
              f"(hands {'ON' if self._hands_active else 'OFF'})")

    def m2_ab(self):
        """M2: swept clearance with the hand at REST (active) vs hands disabled, same spot.
        Equal clearance => the hand is LiDAR self-culled and REST is safe to navigate with."""
        self.env["SetHandsActive"](True); self._hands_active = True
        self.set_hand_pose(REST_LOCAL)
        try:
            c_rest, _ = self._clearance()
        except Exception as e:
            print(f"  [m2] LiDAR scan failed: {type(e).__name__}: {e}"); return
        self.env["SetHandsActive"](False); self._hands_active = False
        c_off, _ = self._clearance()
        self.env["SetHandsActive"](True); self._hands_active = True
        self.set_hand_pose(REST_LOCAL)
        diff = abs(c_rest - c_off)
        culled = diff <= 0.05
        self.log("m2_clearance_ab", clearance_rest=c_rest, clearance_off=c_off,
                 diff=round(diff, 4), self_culled=culled)
        print(f"  [M2] clearance  REST={c_rest:.3f} m   hands-off={c_off:.3f} m   "
              f"diff={diff:.3f} m  ->  {'SELF-CULLED (safe)' if culled else 'HAND SEEN AS OBSTACLE'}")

    def m1_ab(self):
        """M1: the occlusion A/B - a shot at REST vs a shot with hands disabled, same pose.
        Run it once at a shelf and once in an aisle; eyeball the two PNGs."""
        self.env["SetHandsActive"](True); self._hands_active = True
        self.set_hand_pose(REST_LOCAL)
        shot_rest = self.shot("m1_rest")
        self.env["SetHandsActive"](False); self._hands_active = False
        shot_off = self.shot("m1_handsoff")
        self.env["SetHandsActive"](True); self._hands_active = True
        self.set_hand_pose(REST_LOCAL)
        self.log("m1_occlusion_ab", shot_rest=shot_rest, shot_handsoff=shot_off)
        print(f"  [M1] A/B saved: REST={shot_rest}  hands-off={shot_off}  "
              "-> compare; does the resting hand intrude? (do this @ shelf AND @ aisle)")

    def m_lidar(self):
        try:
            c = self.env["RequestLidarCenter"]()
        except Exception as e:
            print(f"  [lidar] failed: {type(e).__name__}: {e}"); return
        self.log("lidar_center", distance=c.get("distance"), hit=c.get("hit"),
                 pitch_deg=c.get("pitch_deg"), camera_height=c.get("camera_height"))
        print(f"  [lidar] center distance={c.get('distance')} hit={c.get('hit')} "
              f"pitch={c.get('pitch_deg')} cam_h={c.get('camera_height')} "
              "(M4: compare empty vs holding)")

    def m_gripq(self):
        hs = self.hand_state()
        print(f"  [grip?] gripped={bool(hs.get('leftGrippedState'))} "
              f"hovered={hs.get('leftHoveredObject')!r} "
              f"hand={tuple(round(v, 3) for v in hs.get('leftTranslation', ()))}")

    def drive(self, cp_id):
        """M3, the full version: a real checkpoint drive with hands NOT stowed (stowing them is the
        very bug 6.1 removes), then a carry-check. Uses the frozen map via NavSession."""
        if self._nav is None:
            try:
                from nav.store_map import StoreMap, NavSession
                from sim.env import default_uri
                sm = StoreMap()
                self._nav = NavSession(sm, uri=default_uri(),
                                       stow_hands=False)   # do NOT disable hands - carrying!
            except Exception as e:
                print(f"  [drive] could not open the map/NavSession: {type(e).__name__}: {e}")
                return
        try:
            ok = self._nav.goto(int(cp_id))
        except Exception as e:
            print(f"  [drive] goto(cp{cp_id}) raised {type(e).__name__}: {e}")
            self.log("drive", cp=cp_id, error=f"{type(e).__name__}: {e}")
            return
        print(f"  [drive] goto(cp{cp_id}) -> {'arrived' if ok else 'REFUSED'}")
        self.log("drive", cp=cp_id, arrived=bool(ok))
        self.carry_check(f"cp{cp_id}")

    # ---- loop ---------------------------------------------------------------------------------
    def run(self):
        env = self.env
        try:
            env["RequestScreenshot"](save_image=False)
        except Exception as e:
            print(f"Could not reach the sim ({type(e).__name__}: {e}). Is it in Play mode on :8080?")
            return
        print(HELP)
        print(f"run dir: {self.run_dir}\nmeasurements -> measurements.jsonl   notes -> notes.md\n")
        self.m_rest()
        try:
            while True:
                try:
                    line = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line:
                    continue
                parts = line.split()
                cmd = parts[0].lower()
                rest = parts[1:]
                count = int(rest[0]) if (rest and rest[0].lstrip("-").isdigit()) else 1
                if cmd in ("q", "quit", "exit"):
                    break
                elif cmd in ("?", "help"):
                    print(HELP)
                elif cmd == "rest":
                    self.m_rest()
                elif cmd == "grab":
                    self.m_grab(" ".join(rest))
                elif cmd == "x":
                    self.m_release()
                elif cmd == "m1":
                    self.m1_ab()
                elif cmd == "m2":
                    self.m2_ab()
                elif cmd == "clear":
                    self.m_clear()
                elif cmd == "lidar":
                    self.m_lidar()
                elif cmd in ("grip?", "grip"):
                    self.m_gripq()
                elif cmd in self.MOVES:
                    self.MOVES[cmd](count)
                    self.carry_check(cmd)
                elif cmd == "drive":
                    if rest:
                        self.drive(rest[0])
                    else:
                        print("  usage: drive <cp_id>")
                elif cmd == "shot":
                    print(f"  saved {self.shot(' '.join(rest) or 'shot')}")
                elif cmd == "off":
                    env["SetHandsActive"](False); self._hands_active = False; print("  hands OFF")
                elif cmd == "on":
                    env["SetHandsActive"](True); self._hands_active = True
                    self.set_hand_pose(REST_LOCAL); print("  hands ON, at REST")
                elif cmd == "note":
                    with open(self.notes, "a", encoding="utf-8") as f:
                        f.write(f"- ({datetime.now().strftime('%H:%M:%S')}) {' '.join(rest)}\n")
                    print("  noted.")
                else:
                    print(f"  ? unknown command {cmd!r} - type ? for help")
        finally:
            # Leave the agent in a known, non-destructive state: grip released, standing, hands
            # active at REST (NOT disabled - matches the 6.1 end-of-task contract, not stow).
            try:
                env["SetHandsActive"](True)
                self.set_hand_pose(REST_LOCAL)
            except Exception:
                pass
            print(f"\ndone. run dir -> {self.run_dir}")


def main():
    Probe().run()


if __name__ == "__main__":
    main()
