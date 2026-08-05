"""
center_offline_check.py - A/B the centring MATH offline, no Unity sim required.

Given saved capture PNG(s) and a target description, this runs the SAME deterministic
detector center_object_on_screen now uses (perception._detect_bbox_px, temp 0), then
reports, per image:
  * detected label + bbox centre
  * residual from frame centre (pixels and normalised)
  * the rotation the OLD model (px / 19.2) vs the NEW model (atan / FOCAL_PX) would command
  * SEED A/B: over ALL detected instances (_detect_boxes_px), which one the OLD look-1 seed
    (pure nearest-to-aim) vs the NEW front-biased seed (--front-bias) locks onto, and whether
    they disagree. This is how you validate the front-of-row seed on a saved frame BEFORE
    trusting the default: eyeball look<i>_bbox.png / the printed boxes and confirm the NEW pick
    is the item actually in FRONT.

What it PROVES offline: the detector is deterministic at temp 0 (--repeat), and how far the
two projection models disagree on the commanded angle. What it does NOT prove: closed-loop
convergence - that needs the live sim (rotate, re-screenshot, re-measure). Use the pickup navigation eval
for the end-to-end grab number.

`--selftest` needs no images and no network - it checks the projection math against the
known FOV geometry (centre -> 0, right edge -> half hFOV, bottom edge -> half vFOV = 30 deg)
and prints how far the retired /19.2 model was off at the edges.

Usage:
    python center_offline_check.py --selftest
    python center_offline_check.py IMG.png --target "a can of Century Tuna" --repeat 3
    python center_offline_check.py "mapping/output/captures/*_primary.png" --target "Piattos"
"""
import argparse
import glob
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(os.path.dirname(os.path.dirname(_THIS_DIR)), "agent")  # agent/ — perception lives there
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

OLD_GAIN = 19.2  # the retired linear px/deg constant, kept only for the A/B print


def old_model_rotation(cx, cy, w=1920, h=1080):
    """(pitch, yaw) the pre-fix code would have commanded: a single linear gain px/19.2."""
    return (cy - h / 2) / OLD_GAIN, (cx - w / 2) / OLD_GAIN


def selftest():
    from vision.perception import bbox_to_rotation, FOCAL_PX, ORIGINAL_WIDTH, ORIGINAL_HEIGHT
    aim_x, aim_y = ORIGINAL_WIDTH / 2, ORIGINAL_HEIGHT / 2
    print(f"FOCAL_PX = {FOCAL_PX:.2f} px  (expected ~935.31 for 60 deg vFOV @ 1080p)")

    # dead centre -> no rotation
    p, y = bbox_to_rotation(aim_x, aim_y, aim_x, aim_y)
    assert abs(p) < 1e-9 and abs(y) < 1e-9, (p, y)
    print(f"  centre       new pitch={p:+6.2f} yaw={y:+6.2f}")

    # right edge -> exactly half the horizontal FOV
    p, y = bbox_to_rotation(ORIGINAL_WIDTH, aim_y, aim_x, aim_y)
    hfov_half = math.degrees(math.atan2(aim_x, FOCAL_PX))
    assert abs(y - hfov_half) < 1e-6, (y, hfov_half)
    _, oy = old_model_rotation(ORIGINAL_WIDTH, aim_y)
    print(f"  right edge   new yaw={y:+6.2f} (=half hFOV {hfov_half:.2f})   "
          f"old yaw={oy:+6.2f}   [old overshoots {oy - y:+.2f} deg]")

    # bottom edge -> exactly half the vertical FOV = 30 deg
    p, y = bbox_to_rotation(aim_x, ORIGINAL_HEIGHT, aim_x, aim_y)
    assert abs(p - 30.0) < 1e-6, p
    op, _ = old_model_rotation(aim_x, ORIGINAL_HEIGHT)
    print(f"  bottom edge  new pitch={p:+6.2f} (=half vFOV 30.00)  "
          f"old pitch={op:+6.2f}   [old undershoots {op - p:+.2f} deg]")

    # a mid-frame point, to show the everyday near-centre gap
    cx, cy = aim_x + 300, aim_y + 150
    np_, ny = bbox_to_rotation(cx, cy, aim_x, aim_y)
    op, oy = old_model_rotation(cx, cy)
    print(f"  +300,+150px  new (pitch={np_:+.2f}, yaw={ny:+.2f})  "
          f"old (pitch={op:+.2f}, yaw={oy:+.2f})   [old rotates ~{oy / ny - 1:+.0%} on yaw]")
    print("selftest OK")


def check_image(path, target, repeat):
    from vision.perception import _detect_bbox_px, bbox_to_rotation, ORIGINAL_WIDTH, ORIGINAL_HEIGHT
    from PIL import Image
    img = Image.open(path)
    print(f"\n== {path}   target={target!r} ==")

    dets = []
    for k in range(repeat):
        b = _detect_bbox_px(img, f"main_goal={target}")
        if b is None:
            print(f"  [{k}] not detected")
            continue
        dets.append(b)
        print(f"  [{k}] {b['label']!r} center=({b['cx']:.0f},{b['cy']:.0f})")
    if not dets:
        return

    if repeat > 1:
        cxs = {round(b['cx']) for b in dets}
        cys = {round(b['cy']) for b in dets}
        stable = len(cxs) == 1 and len(cys) == 1
        print(f"  determinism: {len(cxs)} distinct cx / {len(cys)} distinct cy over "
              f"{len(dets)} calls  ({'STABLE' if stable else 'JITTER'})")

    b = dets[0]
    aim_x, aim_y = ORIGINAL_WIDTH / 2, ORIGINAL_HEIGHT / 2
    dx, dy = b['cx'] - aim_x, b['cy'] - aim_y
    np_, ny = bbox_to_rotation(b['cx'], b['cy'], aim_x, aim_y)
    op, oy = old_model_rotation(b['cx'], b['cy'])
    print(f"  residual : dx={dx:+.0f}px ({dx / ORIGINAL_WIDTH:+.1%})  "
          f"dy={dy:+.0f}px ({dy / ORIGINAL_HEIGHT:+.1%})")
    print(f"  NEW rot  : pitch={np_:+.2f}  yaw={ny:+.2f}")
    print(f"  OLD rot  : pitch={op:+.2f}  yaw={oy:+.2f}   "
          f"(delta pitch {op - np_:+.2f}, yaw {oy - ny:+.2f})")


def check_seed(path, target, front_bias):
    """A/B the look-1 SEED offline: detect ALL instances, then show which one the OLD seed
    (nearest-to-aim, front_bias=0) vs the NEW front-biased seed locks - and flag when they
    disagree (the front-of-row fix actually changing the pick). Boxes are listed largest-area
    first, since area is the front-of-row proxy the new seed weights."""
    from vision.perception import (_detect_boxes_px, _seed_front_instance,
                            ORIGINAL_WIDTH, ORIGINAL_HEIGHT)
    from PIL import Image
    img = Image.open(path)
    aim_x, aim_y = ORIGINAL_WIDTH / 2, ORIGINAL_HEIGHT / 2
    boxes = _detect_boxes_px(img, f"main_goal={target}")
    if not boxes:
        print("  seed A/B: no instances detected")
        return

    def _area(b):
        return (b['xmax'] - b['xmin']) * (b['ymax'] - b['ymin'])

    old = _seed_front_instance(boxes, aim_x, aim_y, 0.0)          # pure nearest-to-aim
    new = _seed_front_instance(boxes, aim_x, aim_y, front_bias)   # front-biased
    print(f"  seed A/B ({len(boxes)} instance(s), front_bias={front_bias}):")
    for b in sorted(boxes, key=_area, reverse=True):
        tag = ("OLD " if b is old else "    ") + ("NEW" if b is new else "")
        d = ((b['cx'] - aim_x) ** 2 + (b['cy'] - aim_y) ** 2) ** 0.5
        print(f"    center=({b['cx']:>4.0f},{b['cy']:>4.0f}) area={_area(b):>9.0f}px^2 "
              f"dist_to_aim={d:>4.0f}px  {tag}")
    print("    -> " + ("DIFFERENT pick (front-bias changed the lock)" if old is not new
                       else "same pick (front-bias made no difference on this frame)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="capture PNG path(s) or glob(s)")
    ap.add_argument("--target", default=None, help="target description for the detector")
    ap.add_argument("--repeat", type=int, default=1,
                    help="detect N times per image to show temp-0 determinism")
    ap.add_argument("--front-bias", type=float, default=0.25,
                    help="front-of-row seed strength for the seed A/B (0 = old nearest-to-aim)")
    ap.add_argument("--selftest", action="store_true",
                    help="check the projection math only (no images, no network)")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        if not args.images:
            return
    if args.images and not args.target:
        ap.error("--target is required when images are given")

    paths = []
    for pat in args.images:
        paths.extend(glob.glob(pat) or [pat])
    for p in paths:
        check_image(p, args.target, args.repeat)
        check_seed(p, args.target, args.front_bias)


if __name__ == "__main__":
    main()
