"""
center_live_test.py - exercise the CLOSED-LOOP centring against the LIVE Unity sim.

Needs the sim in Play mode (ws://localhost:8080) with the agent already facing the shelf so
the target is somewhere in frame. Drives center_object_on_screen for one target and reports
the residual trajectory (printed per look by the function itself) plus whether it converged.
Every image lands under validation/artifacts/calibration/center/<timestamp>/:
center_before/after.png plus
a per-look look<i>_bbox.png showing the VLM's candidate boxes (yellow), the locked target (red)
and the aim crosshair (green). The hands are disabled for the run (active hand prefabs can fill
the camera and occlude the target) and re-enabled after.

    python center_live_test.py --target "a can of Century Tuna"
    python center_live_test.py --target "Piattos" --aim 0.5 0.67      # aim at the grab point
    python center_live_test.py --target "Ritz crackers" --max-iters 4 --tol 15

Read the [CENTER] lines: residual should shrink toward (0,0) and end 'within tolerance'. If it
oscillates or grows, that is the thing to catch here (wrong sign / gain), not in a full run.
"""
import argparse
import os
import shutil
import sys
from datetime import datetime

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(os.path.dirname(os.path.dirname(_THIS)), "agent")  # agent/ — runtime modules (env, perception) live there
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Every image this test produces lands under here, one timestamped folder per run so reruns
# don't clobber and each run's before/after/per-look frames stay together.
CENTERTESTS_DIR = os.path.join(
    os.path.dirname(_ROOT), "validation", "artifacts", "calibration", "center")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="what to centre on")
    ap.add_argument("--aim", type=float, nargs=2, default=[0.5, 0.5], metavar=("X", "Y"),
                    help="normalised aim point (default 0.5 0.5 = centre; try 0.5 0.67 for grabs)")
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--tol", type=float, default=20.0)
    ap.add_argument("--gain", type=float, default=0.8,
                    help="damping on each correction (<1 undershoots; curbs overshoot/ringing)")
    ap.add_argument("--front-bias", type=float, default=0.25,
                    help="look-1 seed's pull toward the FRONT item of a row (0 = pure "
                         "nearest-to-aim, the old behaviour; A/B against that)")
    args = ap.parse_args()

    from sim.env import RequestScreenshot, SetHandsActive
    from vision.perception import center_object_on_screen

    run_dir = os.path.join(CENTERTESTS_DIR, datetime.now().strftime("%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs("screenshots", exist_ok=True)

    def snap(name):
        src = os.path.join("screenshots", "ClientScreenshot.png")
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(run_dir, name))

    try:
        RequestScreenshot(save_image=True)
    except Exception as e:
        print(f"Could not reach the sim ({type(e).__name__}: {e}). Is it in Play mode on :8080?")
        sys.exit(1)

    # Disable the hands for the run: active hand prefabs can fill the camera ("giant ghost
    # hands") and occlude the target, skewing the bbox reads the centring loop depends on.
    # Toggled back on in `finally` so a crash mid-run never leaves the sim hands-off.
    SetHandsActive(False)
    try:
        RequestScreenshot(save_image=True)
        snap("center_before.png")
        result = center_object_on_screen(f"main_goal={args.target}", aim_norm=tuple(args.aim),
                                         max_iters=args.max_iters, tol_px=args.tol, gain=args.gain,
                                         front_bias=args.front_bias, debug_dir=run_dir)
        snap("center_after.png")
    finally:
        SetHandsActive(True)

    print("\n=== RESULT ===")
    print(f"centered={result.get('centered')}  iters={result.get('iters')}  "
          f"residual_px={result.get('residual_px')}")
    print(f"images -> {run_dir}")
    print("  center_before.png, center_after.png, look<i>_bbox.png "
          "(yellow=VLM candidates, red=locked target, green=aim)")


if __name__ == "__main__":
    main()
