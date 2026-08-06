"""
Standalone unit tests for the pure-logic helpers in explore.py.
No live Unity/WebSocket connection needed.

Run with:
    uv run pytest validation/tests/mapping/test_explore.py
"""
import math
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # validation/tests/mapping
_MAPPING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR))), "agent", "mapping")
if _MAPPING_DIR not in sys.path:
    sys.path.insert(0, _MAPPING_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all mapping category dirs)

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("explore_frontier", os.path.join(_MAPPING_DIR, "drivers", "explore.py"))
explore = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(explore)

from occupancy_grid import OccupancyGrid  # noqa: E402
from voxel_grid import VoxelGrid  # noqa: E402


class TestBlockedPosition(unittest.TestCase):
    def test_straight_ahead_when_no_debug_info(self):
        pos = (0.0, 0.0, 0.0)
        bx, bz = explore.blocked_position(pos, rot_deg=0.0, clearance=1.5, clearance_debug=None)

        self.assertAlmostEqual(bx, 0.0, places=6)
        self.assertAlmostEqual(bz, 1.5, places=6)

    def test_uses_off_axis_hit_not_straight_ahead(self):
        # Regression: az_rel=-48.2 was observed live while mark_occupied() assumed the
        # obstruction was straight ahead (rot_deg alone) - poisoning empty space instead of
        # the real obstacle. The marked position must follow the actual ray's bearing/range.
        pos = (0.0, 0.0, 0.0)
        rot_deg = 24.0
        debug = {"az_rel_deg": -48.2, "range": 0.40}

        bx, bz = explore.blocked_position(pos, rot_deg, clearance=0.27, clearance_debug=debug)

        expected_bearing = math.radians(rot_deg + debug["az_rel_deg"])
        expected_x = math.sin(expected_bearing) * debug["range"]
        expected_z = math.cos(expected_bearing) * debug["range"]
        self.assertAlmostEqual(bx, expected_x, places=6)
        self.assertAlmostEqual(bz, expected_z, places=6)

        # And explicitly NOT the straight-ahead (old, buggy) position.
        straight_ahead_x = math.sin(math.radians(rot_deg)) * debug["range"]
        self.assertNotAlmostEqual(bx, straight_ahead_x, places=2)

    def test_offset_from_nonzero_position(self):
        pos = (2.0, 0.0, -3.0)  # (x, y, z) - blocked_position reads pos[0]/pos[2]
        debug = {"az_rel_deg": 10.0, "range": 2.0}
        bx, bz = explore.blocked_position(pos, rot_deg=90.0, clearance=1.0, clearance_debug=debug)

        bearing = math.radians(90.0 + 10.0)
        self.assertAlmostEqual(bx, 2.0 + math.sin(bearing) * 2.0, places=6)
        self.assertAlmostEqual(bz, -3.0 + math.cos(bearing) * 2.0, places=6)


def _make_ring_scan(obstacle_azimuths_and_ranges, max_range=20.0, azimuth_step_deg=1.0):
    """Single channel (v_deg=0, i.e. height == sensor_height_offset, safely in-band),
    one azimuth sample per degree around the full circle, all at max_range (no hit)
    except the given (azimuth_deg, range) overrides - lets a test place an obstacle
    at an exact bearing without needing a dense real sensor layout."""
    azimuth_samples = int(round(360.0 / azimuth_step_deg))
    ranges = [max_range] * azimuth_samples
    for az_deg, r in obstacle_azimuths_and_ranges:
        idx = int(round(az_deg / azimuth_step_deg)) % azimuth_samples
        ranges[idx] = r
    return {
        "channels": 1,
        "azimuth_samples": azimuth_samples,
        "min_range": 0.05,
        "max_range": max_range,
        "azimuth_start_deg": 0.0,
        "azimuth_step_deg": azimuth_step_deg,
        "sequence": 0,
        "timestamp_seconds": 0.0,
        "vertical_angles_deg": [0.0],
        "ranges": ranges,
    }


class TestFindClearHeading(unittest.TestCase):
    def test_returns_zero_offset_when_already_clear(self):
        scan = _make_ring_scan([])  # nothing anywhere

        result = explore.find_clear_heading(
            scan, desired_heading_deg=0.0, min_needed_step=0.3, safety_margin=0.1,
            body_radius=0.3, min_obstacle_height=0.05, max_obstacle_height=2.0,
            sensor_height_offset=1.485, self_exclusion_range=0.1,
        )

        self.assertIsNotNone(result)
        heading, clearance, debug, offset = result
        self.assertEqual(offset, 0.0, "should not nudge at all when the straight heading is already clear")

    def test_finds_smallest_nudge_that_clears_an_off_axis_obstacle(self):
        # Obstacle dead ahead (azimuth 0) at 1.0m: with body_radius=0.3, clearing it needs
        # sin(offset) > 0.3/1.0 = 0.3 -> offset > ~17.46 degrees. With nudge_step_deg=5, the
        # first offset magnitude that actually clears it is 20 degrees, not smaller ones.
        # safety_margin=0.8 (unrealistically large, deliberately) so that even the raw 1.0m
        # straight-ahead clearance (0.2m of margin) fails min_needed_step=0.3 and the search
        # is actually forced to happen, rather than trivially succeeding at offset=0.
        scan = _make_ring_scan([(0.0, 1.0)])

        result = explore.find_clear_heading(
            scan, desired_heading_deg=0.0, min_needed_step=0.3, safety_margin=0.8,
            body_radius=0.3, min_obstacle_height=0.05, max_obstacle_height=2.0,
            sensor_height_offset=1.485, self_exclusion_range=0.1,
            max_nudge_deg=30.0, nudge_step_deg=5.0,
        )

        self.assertIsNotNone(result)
        heading, clearance, debug, offset = result
        self.assertEqual(abs(offset), 20.0, "15 degrees is not enough to clear this obstacle; 20 is")
        self.assertGreaterEqual(clearance - 0.1, 0.3, "the returned heading must actually satisfy the safety margin")

    def test_returns_none_when_nothing_within_range_clears(self):
        # Obstacle closer than body_radius itself: no heading adjustment can move it
        # outside the swept-cylinder radius (even a 90-degree turn barely helps, and that's
        # far outside any reasonable nudge budget) - this must be reported as unresolvable
        # by nudging, not silently "succeed" with a heading that still isn't actually safe.
        scan = _make_ring_scan([(0.0, 0.2)])

        result = explore.find_clear_heading(
            scan, desired_heading_deg=0.0, min_needed_step=0.3, safety_margin=0.1,
            body_radius=0.3, min_obstacle_height=0.05, max_obstacle_height=2.0,
            sensor_height_offset=1.485, self_exclusion_range=0.1,
            max_nudge_deg=30.0, nudge_step_deg=5.0,
        )

        self.assertIsNone(result)


class TestFindEscapeHeading(unittest.TestCase):
    def test_picks_the_most_open_heading_away_from_a_frontal_wall(self):
        # A wall dead ahead (azimuth 0) at 0.4m, the rest of the circle open. Unlike
        # find_clear_heading (smallest deviation that merely clears), the escape heading is
        # the MAX-clearance one - it must point well away from the blocked straight-ahead,
        # toward the open space behind/beside.
        scan = _make_ring_scan([(0.0, 0.4)])  # only obstacle is straight ahead

        result = explore.find_escape_heading(
            scan, safety_margin=0.65, min_step=0.1,
            body_radius=0.3, min_obstacle_height=0.05, max_obstacle_height=2.0,
            sensor_height_offset=1.485, self_exclusion_range=0.1, sweep_step_deg=10.0,
        )

        self.assertIsNotNone(result)
        heading, clearance, _debug = result
        self.assertGreater(abs(explore.normalize_deg(heading)), 30.0,
                           "escape must turn well away from the blocked straight-ahead heading")
        self.assertGreaterEqual(clearance - 0.65, 0.1, "escape heading must fit at least a min_step move")

    def test_returns_none_when_boxed_in_on_all_sides(self):
        # Obstacles all around inside the safety envelope: no heading fits a step, so escape
        # is impossible and the caller must fall back to holding (which then trips the
        # planner's no-movement circuit breaker and ends the run) rather than pretend to move.
        ring = [(az, 0.4) for az in range(0, 360, 10)]
        scan = _make_ring_scan(ring, azimuth_step_deg=10.0)

        result = explore.find_escape_heading(
            scan, safety_margin=0.65, min_step=0.1,
            body_radius=0.3, min_obstacle_height=0.05, max_obstacle_height=2.0,
            sensor_height_offset=1.485, self_exclusion_range=0.1, sweep_step_deg=10.0,
        )

        self.assertIsNone(result)


class _FakeAgent:
    """Fake Unity connection for step_agent. Models the sim's EGOCENTRIC TranslateAgent: the
    incoming translation is body-relative (x=right, y=up, z=forward) and the sim rotates it into
    world space itself using the current facing (AgentControllerBase.EgocentricToWorldTranslation),
    then applies the delta rotation as a yaw increment. No network. Records calls.

    Egocentric-not-world is the whole point of the 2026-07-22 fix: a fake that added the raw delta
    to world XZ modeled the OLD (pre-3940ce7) sim and hid the double-rotation the fix removes."""
    def __init__(self, pos, yaw):
        self.pos = list(pos)
        self.yaw = yaw
        self.calls = []

    def step_agent(self, dtrans, drot, uri):
        self.calls.append((tuple(dtrans), tuple(drot)))
        # Unity left-handed Y-up: forward=(sin yaw, cos yaw), right=(cos yaw, -sin yaw).
        # world = right*x + forward*z, with dtrans = (right, up, forward). Translation uses the
        # CURRENT facing (the sim converts before applying the rotation delta), matching how
        # explore.py always sends rotation and translation as separate zero-crossed calls.
        yaw = math.radians(self.yaw)
        s, c = math.sin(yaw), math.cos(yaw)
        self.pos[0] += dtrans[0] * c + dtrans[2] * s
        self.pos[2] += -dtrans[0] * s + dtrans[2] * c
        self.yaw = explore.normalize_deg(self.yaw + drot[1])
        return tuple(self.pos), (0.0, self.yaw, 0.0), False


class _ScriptedPlanner:
    def __init__(self, navs):
        self._navs = list(navs)
        self.notify_blocked_calls = []

    def update(self, pos_xz, cell):
        return self._navs.pop(0)

    def notify_blocked(self, pos_xz):
        self.notify_blocked_calls.append(pos_xz)


class _FakeCloud:
    def add(self, points):
        pass

    def save(self, output_dir, tag, include_ply=True):
        pass


def _forward_hits_scan():
    """A small scan whose rays fan out and hit at ~2m, so integrate produces occupied cells
    around the agent - enough to prove integrate+collapse actually populated the grid."""
    channels = [-10.0, -3.0, 3.0, 10.0]
    az_samples = 8
    return {
        "channels": len(channels), "azimuth_samples": az_samples,
        "ranges": [2.0] * (len(channels) * az_samples),
        "max_range": 20.0, "min_range": 0.05,
        "azimuth_start_deg": 0.0, "azimuth_step_deg": 360.0 / az_samples,
        "vertical_angles_deg": channels,
    }


def _move_nav(target=(2.5, 2.5)):
    return SimpleNamespace(kind="move", target_world_xz=target, goal_world_xz=target, replanned=False)


def _done_nav():
    return SimpleNamespace(kind="done", target_world_xz=None, goal_world_xz=None, replanned=False)


class TestExploreLoopVoxelWiring(unittest.TestCase):
    def _run(self, navs, clearances, nudges, escapes=None, escape_when_wedged=False):
        agent = _FakeAgent((0.0, 0.0, 0.0), 0.0)
        voxel = VoxelGrid(size_m=6.0, resolution=0.1,
                          min_obstacle_height=0.05, max_obstacle_height=2.0)
        grid = voxel.grid
        planner = _ScriptedPlanner(navs)
        args = explore.build_parser().parse_args([])
        args.size = 6.0
        # Default OFF so the pre-existing wiring tests exercise the pure blocked-mark/hold
        # path without find_escape_heading consuming extra swept_clearance_ahead side-effects;
        # the escape test opts in explicitly. find_escape_heading is patched either way so a
        # stray call can never reach the real full-circle sweep during a wiring test.
        args.escape_when_wedged = escape_when_wedged
        esc_kwargs = {"side_effect": escapes} if escapes is not None else {"return_value": None}

        with patch.object(explore, "RequestLidarScan", return_value=_forward_hits_scan()), \
             patch.object(explore, "scan_to_world_points_3d", return_value=[]), \
             patch.object(explore, "save_snapshot"), \
             patch.object(explore, "step_agent", side_effect=agent.step_agent), \
             patch.object(explore, "swept_clearance_ahead", side_effect=clearances), \
             patch.object(explore, "find_clear_heading", side_effect=nudges), \
             patch.object(explore, "find_escape_heading", **esc_kwargs):
            explore._explore_loop(args, voxel, grid, _FakeCloud(),
                                  (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), planner)
        return agent, voxel, grid, planner

    def test_scans_integrate_and_collapse_into_the_2d_grid(self):
        # Two normal (clear) steps, then done. The collapsed grid must show occupied cells
        # (the scan's hits) and freed cells (rays through open space) - proof integrate()
        # AND collapse() ran and wrote the shared OccupancyGrid the planner reads.
        _agent, voxel, grid, _planner = self._run(
            navs=[_move_nav(), _move_nav(), _done_nav()],
            clearances=[(5.0, None), (5.0, None)],
            nudges=[],
        )
        occ = (grid.log_odds > OccupancyGrid.OCCUPIED_THRESHOLD).sum()
        free = (grid.log_odds < OccupancyGrid.FREE_THRESHOLD).sum()
        self.assertGreater(occ, 0, "collapse must surface the scan's occupied hits")
        self.assertGreater(free, 0, "collapse must surface freed space along the rays")
        # the 2D grid is the voxel grid's own collapsed output, not a separate object
        self.assertIs(grid, voxel.grid)

    def test_blocked_step_marks_the_voxel_grid_and_survives_collapse(self):
        # A single blocked step (tiny clearance, no nudge escape). The obstacle must be
        # recorded in the VOXEL grid (so the next collapse still shows it), never lost by
        # writing only to the 2D grid that collapse() overwrites. notify_blocked must fire.
        debug = {"height_above_root": 1.2, "az_rel_deg": 0.0, "range": 0.25,
                 "channel": 0, "v_deg": 0.0, "lateral": 0.0}
        _agent, voxel, grid, planner = self._run(
            navs=[_move_nav(), _done_nav()],
            clearances=[(0.05, debug)],
            nudges=[None],
        )
        self.assertEqual(len(planner.notify_blocked_calls), 1)
        # the blocked obstacle lives in the voxel grid at the offending hit's height bin...
        blocked_bin = voxel._height_bin(1.2)
        self.assertGreater((voxel.voxels[:, :, blocked_bin] > 0).sum(), 0,
                           "blocked mark must be occupied evidence in the voxel grid")
        # ...and a fresh collapse still shows occupied cells there (it wasn't a 2D-only mark
        # that the next collapse would erase).
        voxel.collapse()
        self.assertGreater((grid.log_odds > OccupancyGrid.OCCUPIED_THRESHOLD).sum(), 0)

    def test_wedged_step_escapes_toward_the_open_side(self):
        # Blocked straight ahead AND no nudge clears -> with escape enabled the agent must
        # turn to the most-open heading and physically STEP out (leave the pocket), not just
        # hold. The blocked mark + notify_blocked still fire first (the planner is forced to
        # replan from the escaped cell), and the step must be a real, nonzero relocation.
        debug = {"height_above_root": 1.2, "az_rel_deg": 0.0, "range": 0.30,
                 "channel": 0, "v_deg": 0.0, "lateral": 0.0}
        esc_debug = {"height_above_root": 1.0, "az_rel_deg": 0.0, "range": 5.0,
                     "channel": 0, "v_deg": 0.0, "lateral": 0.0}
        agent, _voxel, _grid, planner = self._run(
            navs=[_move_nav(), _done_nav()],
            clearances=[(0.05, debug)],            # main-loop clearance: blocked
            nudges=[None],                          # no +-nudge clears
            escapes=[(180.0, 5.0, esc_debug)],      # most-open heading is behind, 5m clear
            escape_when_wedged=True,
        )
        self.assertEqual(len(planner.notify_blocked_calls), 1,
                         "the blocked mark/notify must still fire before escaping")
        translations = [dt for dt, _dr in agent.calls if dt != (0.0, 0.0, 0.0)]
        self.assertTrue(translations, "escape must issue a real translation, not hold position")
        self.assertGreater(math.hypot(agent.pos[0], agent.pos[2]), 0.1,
                           "escape must relocate the agent out of the wedge")

    def test_wedged_step_holds_when_escape_disabled(self):
        # Same wedge, but with escape off: the agent must NOT move (old hold-and-wait
        # behavior preserved), proving the escape is gated on the flag.
        debug = {"height_above_root": 1.2, "az_rel_deg": 0.0, "range": 0.30,
                 "channel": 0, "v_deg": 0.0, "lateral": 0.0}
        agent, _voxel, _grid, planner = self._run(
            navs=[_move_nav(), _done_nav()],
            clearances=[(0.05, debug)],
            nudges=[None],
            escape_when_wedged=False,
        )
        self.assertEqual(len(planner.notify_blocked_calls), 1)
        translations = [dt for dt, _dr in agent.calls if dt != (0.0, 0.0, 0.0)]
        self.assertFalse(translations, "with escape disabled the agent must hold position")

    def test_arrival_at_waypoint_does_not_trigger_escape(self):
        # step_len falls below min_step here because dist-to-waypoint is ~0 (the agent is AT
        # the waypoint), NOT because of an obstacle - clearance is a wide-open 5m. That is a
        # waypoint arrival, not a wedge; escape must NOT fire (would be a pointless detour and
        # would burn escape budget). Regression for a real live-run false positive: the agent
        # parked at a final observation-vantage waypoint and spuriously "escaped".
        agent, _voxel, _grid, planner = self._run(
            navs=[_move_nav(target=(0.0, 0.0)), _done_nav()],  # target == start -> dist ~ 0
            clearances=[(5.0, None)],                          # wide open: not clearance-limited
            nudges=[None],
            escapes=[(180.0, 5.0, {"height_above_root": 1.0, "az_rel_deg": 0.0, "range": 5.0,
                                   "channel": 0, "v_deg": 0.0, "lateral": 0.0})],
            escape_when_wedged=True,                           # enabled, but must stay unused here
        )
        self.assertEqual(len(planner.notify_blocked_calls), 1, "arrival still notifies/replans")
        translations = [dt for dt, _dr in agent.calls if dt != (0.0, 0.0, 0.0)]
        self.assertFalse(translations, "a waypoint arrival (open clearance) must not trigger an escape step")

    def test_forward_step_travels_along_facing_not_double_rotated(self):
        # Regression for the world-vs-egocentric de-sync (2026-07-22). The sim rotates the
        # body-relative step into world space itself, so after facing an off-axis target the
        # loop must step STRAIGHT along its new facing (here +X), advancing toward the target -
        # NOT perpendicular to it. Pre-rotating the step in Python (the old code) produced a
        # sideways move once the sim went egocentric, because the sim rotated it a SECOND time.
        # The translation the loop sends must be body-relative forward (0, 0, step_len).
        agent, _voxel, _grid, _planner = self._run(
            navs=[_move_nav(target=(2.0, 0.0)), _done_nav()],  # due +X from the start
            clearances=[(5.0, None)],                          # wide open: full step_size move
            nudges=[],
        )
        # angle_to_deg(dx=2, dz=0) == 90deg, so the loop rotates to yaw 90 (facing +X)...
        self.assertAlmostEqual(agent.yaw, 90.0, places=3, msg="must rotate to face the +X target")
        # ...then a body-relative forward step must carry the agent along +X toward the target,
        # with no sideways (Z) drift. Under the old double-rotation this step landed on -Z.
        self.assertGreater(agent.pos[0], 0.1, "forward step must advance toward the +X target")
        self.assertAlmostEqual(agent.pos[2], 0.0, places=6,
                               msg="a forward step after facing the target must not drift sideways")
        # And the translation the loop actually sent was body-relative forward, not a
        # pre-rotated world vector.
        fwd_steps = [dt for dt, _dr in agent.calls if dt != (0.0, 0.0, 0.0)]
        self.assertEqual(len(fwd_steps), 1)
        self.assertAlmostEqual(fwd_steps[0][0], 0.0, places=6, msg="no body-relative right component")
        self.assertGreater(fwd_steps[0][2], 0.1, "body-relative forward (z) component")


class TestClearOutputDir(unittest.TestCase):
    def test_removes_leftover_files_from_a_previous_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("grid_0.npy", "grid_0.png", "grid_final.npy", "points_final.ply"):
                open(os.path.join(tmp, name), "w").close()

            explore._clear_output_dir(tmp)

            self.assertEqual(os.listdir(tmp), [])

    def test_leaves_subdirectories_alone(self):
        # Defensive: only ever remove files directly inside output_dir, never recurse into
        # (let alone delete) a subdirectory that happens to live there.
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "keep_me")
            os.makedirs(sub)
            open(os.path.join(sub, "untouched.txt"), "w").close()
            open(os.path.join(tmp, "grid_0.npy"), "w").close()

            explore._clear_output_dir(tmp)

            self.assertEqual(os.listdir(tmp), ["keep_me"])
            self.assertEqual(os.listdir(sub), ["untouched.txt"])

    def test_missing_output_dir_is_a_silent_no_op(self):
        explore._clear_output_dir(os.path.join(tempfile.gettempdir(), "definitely-does-not-exist-12345"))

    def test_empty_output_dir_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            explore._clear_output_dir(tmp)  # must not raise or print a bogus "cleared 0 files"

            self.assertEqual(os.listdir(tmp), [])


if __name__ == "__main__":
    unittest.main()
