import math

SENSOR_HEIGHT_OFFSET_M = 1.485
"""Meters the LiDAR sensor sits above the agent's reported root position
(world_pos[1], i.e. what TransformAgent reports as translation). Unity's
AlignLevelToSource (LidarSensor.cs) re-parents the sensor to the live
camera/head position each scan, not the root transform that Python reads
back - empirically, the rig's Head Pivot sits ~1.485m above root (local Y
1.632 under a 0.91-scaled "IK Humanoid Agent" prefab root). Rig-specific;
re-derive/verify against a live scan next to a known-height object if the
character model or scale ever changes.
"""

SELF_EXCLUSION_RANGE_M = 0.1
"""Meters within which LiDAR hits are ignored regardless of scan["min_range"] -
distinct from the sensor's own hardware min-range spec, this is a minimal
sensor-housing-noise floor only.

DELIBERATELY kept small. An earlier version of this constant was 0.8m,
intended to blanket-exclude a suspected self-hit (a stowed hand/arm/shoulder
observed at ~0.6-0.7m, sensor-mount height, lateral offset landing right at
body_radius). That was confirmed live to be actively unsafe: with the
blind spot that wide, the agent walked into and scraped along a real shelf
without swept_clearance_ahead ever reporting reduced clearance - the shelf
fell inside the same 0-0.8m dead zone as the suspected self-hit. A blanket
range-based exclusion cannot distinguish "the agent's own body" from "a real
obstacle at a similar distance," so it was rolled back. Do not raise this
back toward 0.6-0.8m without a more targeted (non-range-only) self-hit
signature - e.g. matching the specific v_deg/lateral combination observed -
or a live-confirmed source (try --no-stow-hands and compare the debug output)
so real close obstacles stay detectable.
"""


def normalize_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def angle_to_deg(dx, dz):
    return math.degrees(math.atan2(dx, dz))


def _hit_height_above_root(v_deg, r, sensor_height_offset=SENSOR_HEIGHT_OFFSET_M):
    """Height of a LiDAR hit relative to the agent's own root position (NOT
    absolute world-Y-from-floor - self-relative height sidesteps needing to
    know where the true floor sits relative to the reported root)."""
    return r * math.sin(math.radians(v_deg)) + sensor_height_offset


def iter_banded_hits(scan, world_pos, yaw_deg, min_obstacle_height=0.05,
                     max_obstacle_height=2.0, sensor_height_offset=SENSOR_HEIGHT_OFFSET_M,
                     self_exclusion_range=SELF_EXCLUSION_RANGE_M):
    """Shared projection core for the height-banded 2D mapping path: yields
    (wx, wz, height_above_root) for every LiDAR hit that clears the range/
    self-exclusion filters AND falls within [min_obstacle_height,
    max_obstacle_height] of the agent's own body (see _hit_height_above_root).

    Both scan_to_world_points() - which discards the height - and voxel_grid's
    3D integrate - which bins by it - consume this, so the fragile rotation-sign
    convention and the height-band filter live in exactly ONE place and can't
    drift between the 2D grid and the voxel grid. Rotation sign follows Unity's
    left-handed Y-up Euler(0, yaw, 0): verify against a real scan next to a known
    wall before trusting it.

    Horizontal placement uses the ground-projected distance r*cos(v_deg), NOT the
    raw slant range r: a downward/upward-angled channel's hit is closer to the
    sensor horizontally than its slant range by exactly cos(v), and an earlier
    version that used r placed steep channels' hits too far out - smearing a shelf
    face across a range of depths (worst near the FOV edges, e.g. ~40% over-range
    at 45deg) and reading it as an over-thick, ragged block. This now matches the
    same r*cos(v) decomposition _hit_height_above_root (r*sin(v)) and
    scan_to_world_points_3d already use, so height and ground-distance come from
    one consistent spherical->cartesian split.
    """
    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    channels = scan["channels"]
    azimuth_samples = scan["azimuth_samples"]
    ranges = scan["ranges"]
    max_range = scan["max_range"]

    exclusion_range = max(scan.get("min_range", 0.0), self_exclusion_range)
    for ch in range(channels):
        v_deg = scan["vertical_angles_deg"][ch]
        horiz_scale = math.cos(math.radians(v_deg))
        row_start = ch * azimuth_samples
        for az_i in range(azimuth_samples):
            r = ranges[row_start + az_i]
            if r >= max_range - 1e-3:
                continue
            if r <= exclusion_range + 1e-3:
                # Below the sensor's stated minimum reliable range OR within
                # self_exclusion_range - the latter covers the agent's own body
                # (stowed hands/arms/shoulders sitting near the sensor), confirmed
                # via a live scan: a fixed hit at ~0.6-0.7m, sensor-mount height,
                # and lateral offset landing right at body_radius kept showing up
                # regardless of true position/heading - not a real obstacle.
                continue
            height = _hit_height_above_root(v_deg, r, sensor_height_offset)
            if not (min_obstacle_height <= height <= max_obstacle_height):
                continue
            horiz = r * horiz_scale  # ground-projected distance, not slant range
            az = math.radians(scan["azimuth_start_deg"] + az_i * scan["azimuth_step_deg"])
            lx, lz = math.sin(az) * horiz, math.cos(az) * horiz
            wx = lx * cos_y + lz * sin_y + world_pos[0]
            wz = -lx * sin_y + lz * cos_y + world_pos[2]
            yield wx, wz, height


def scan_to_world_points(scan, world_pos, yaw_deg, min_obstacle_height=0.05,
                          max_obstacle_height=2.0, sensor_height_offset=SENSOR_HEIGHT_OFFSET_M,
                          self_exclusion_range=SELF_EXCLUSION_RANGE_M):
    """Project LiDAR hits whose height falls within [min_obstacle_height,
    max_obstacle_height] of the agent's own body (see _hit_height_above_root)
    into world XZ, for the 2D occupancy grid. Height-band filtering (rather
    than a fixed vertical angle) means overhanging ceiling signage - out of
    the agent's real reach - doesn't get baked into the map as an obstacle,
    while a small min_obstacle_height keeps ordinary floor hits from
    downward-angled channels from doing the same at the low end.

    Thin wrapper over iter_banded_hits() (which owns the projection/filter
    logic, shared with voxel_grid); this one just drops the height component.
    """
    return [
        (wx, wz)
        for wx, wz, _height in iter_banded_hits(
            scan, world_pos, yaw_deg,
            min_obstacle_height=min_obstacle_height, max_obstacle_height=max_obstacle_height,
            sensor_height_offset=sensor_height_offset, self_exclusion_range=self_exclusion_range,
        )
    ]


def scan_to_world_points_3d(scan, world_pos, yaw_deg, vertical_band_deg=90.0,
                             sensor_height_offset=SENSOR_HEIGHT_OFFSET_M):
    """Project LiDAR hits into world XYZ, keeping each channel's own height.

    Unlike scan_to_world_points() (which filters to a collision-relevant
    height band for the 2D mapping plane), this keeps every channel within
    vertical_band_deg for the full 3D structure (point-cloud visualization),
    correcting for sensor_height_offset (see SENSOR_HEIGHT_OFFSET_M - the
    sensor tracks the live camera/head position, not world_pos[1] itself).
    Same rotation convention as scan_to_world_points() (Unity left-handed
    Y-up Euler(0, yaw, 0)).
    """
    yaw = math.radians(yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    points = []
    channels = scan["channels"]
    azimuth_samples = scan["azimuth_samples"]
    ranges = scan["ranges"]
    max_range = scan["max_range"]

    for ch in range(channels):
        v_deg = scan["vertical_angles_deg"][ch]
        if abs(v_deg) > vertical_band_deg:
            continue
        v = math.radians(v_deg)
        row_start = ch * azimuth_samples
        for az_i in range(azimuth_samples):
            r = ranges[row_start + az_i]
            if r >= max_range - 1e-3:
                continue
            az = math.radians(scan["azimuth_start_deg"] + az_i * scan["azimuth_step_deg"])
            horiz = r * math.cos(v)
            ly = _hit_height_above_root(v_deg, r, sensor_height_offset)
            lx, lz = math.sin(az) * horiz, math.cos(az) * horiz
            wx = lx * cos_y + lz * sin_y + world_pos[0]
            wz = -lx * sin_y + lz * cos_y + world_pos[2]
            wy = ly + world_pos[1]
            points.append((wx, wy, wz))
    return points


def clearance_ahead(scan, heading_deg, cone_deg=20.0, vertical_band_deg=40.0):
    """Minimum LiDAR range within a cone around heading_deg, where heading_deg
    is relative to the sensor's own forward axis (0 = straight ahead) — pass
    the about-to-be-applied yaw delta, since ranges/azimuth in the scan are
    already sensor-local. Uses a wider vertical band than the 2D map
    projection so shelf items above/below the mapping slice still count.

    Kept for callers that want a plain angular check; prefer
    swept_clearance_ahead() for step-size gating, since a fixed-degree cone
    corresponds to a shrinking linear margin as range decreases and can miss
    obstacles that would clip the agent's actual body width up close.
    """
    channels = scan["channels"]
    azimuth_samples = scan["azimuth_samples"]
    ranges = scan["ranges"]
    min_seen = scan["max_range"]

    for ch in range(channels):
        v_deg = scan["vertical_angles_deg"][ch]
        if abs(v_deg) > vertical_band_deg:
            continue
        row_start = ch * azimuth_samples
        for az_i in range(azimuth_samples):
            az = scan["azimuth_start_deg"] + az_i * scan["azimuth_step_deg"]
            if abs(normalize_deg(az - heading_deg)) > cone_deg:
                continue
            r = ranges[row_start + az_i]
            if r < min_seen:
                min_seen = r
    return min_seen


def swept_clearance_ahead(
    scan, heading_deg, body_radius=0.3, min_obstacle_height=0.05, max_obstacle_height=2.0,
    sensor_height_offset=SENSOR_HEIGHT_OFFSET_M, max_lateral_deg=70.0,
    self_exclusion_range=SELF_EXCLUSION_RANGE_M, debug=False
):
    """Minimum forward distance to any LiDAR hit that falls within body_radius
    of the straight-line path along heading_deg — i.e. clearance for a
    cylinder of that radius swept forward, not just a point.

    A fixed angular cone (see clearance_ahead) checks a constant *angle*, so
    the linear slice of the world it covers shrinks with range: an obstacle
    just outside the cone can still be well within the agent's actual body
    width once the agent is close to it. Decomposing each ray into a
    forward/lateral pair against the travel line avoids that, at the cost of
    needing an estimate of the agent's radius (shoulder/hand width) — tune
    body_radius to the sim's actual agent footprint.

    Height-band filtered ([min_obstacle_height, max_obstacle_height] relative
    to the agent's own root position — see _hit_height_above_root) rather than
    a fixed vertical angle, so overhanging ceiling signage the agent can never
    physically reach doesn't count as a blocking obstacle, while
    min_obstacle_height keeps ordinary floor hits from doing the same. Hits
    within max(scan["min_range"], self_exclusion_range) are also excluded -
    see SELF_EXCLUSION_RANGE_M for why a plain sensor min_range wasn't enough
    (confirmed live: the agent's own body produces a persistent close-range hit
    the sensor's own spec doesn't cover).

    If debug=True, returns (min_seen, debug_info) where debug_info is a dict
    describing the closest counted hit (or None if nothing counted) - use this
    to diagnose an unexpectedly small/constant clearance reading (e.g. a
    self-hit) rather than guessing from the returned distance alone.
    """
    channels = scan["channels"]
    azimuth_samples = scan["azimuth_samples"]
    ranges = scan["ranges"]
    max_range = scan["max_range"]
    exclusion_range = max(scan.get("min_range", 0.0), self_exclusion_range)
    min_seen = max_range
    min_seen_debug = None

    for ch in range(channels):
        v_deg = scan["vertical_angles_deg"][ch]
        row_start = ch * azimuth_samples
        for az_i in range(azimuth_samples):
            az = scan["azimuth_start_deg"] + az_i * scan["azimuth_step_deg"]
            rel_deg = normalize_deg(az - heading_deg)
            if abs(rel_deg) > max_lateral_deg:
                continue
            r = ranges[row_start + az_i]
            if r >= max_range - 1e-3:
                continue
            if r <= exclusion_range + 1e-3:
                continue
            height = _hit_height_above_root(v_deg, r, sensor_height_offset)
            if not (min_obstacle_height <= height <= max_obstacle_height):
                continue
            rel = math.radians(rel_deg)
            forward = r * math.cos(rel)
            lateral = r * math.sin(rel)
            if forward <= 0 or abs(lateral) > body_radius:
                continue
            if forward < min_seen:
                min_seen = forward
                if debug:
                    min_seen_debug = {
                        "channel": ch, "v_deg": v_deg, "range": r,
                        "height_above_root": height, "az_rel_deg": rel_deg, "lateral": lateral,
                    }
    if debug:
        return min_seen, min_seen_debug
    return min_seen
