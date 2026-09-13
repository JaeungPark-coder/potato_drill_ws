"""Pure-python helpers for stage 3: order detected potato eyes into an
efficient visiting sequence (open-path TSP, nearest-neighbor + 2-opt),
compute drill approach poses from each eye's position + surface normal
(as published by eye_detector on /potato_scan/eye_poses), and describe
the outcome of one insertion (DrillOutcome).

Deliberately depends on numpy alone -- both robot backends import
DrillOutcome from here, and robot_interface.py pulls in ur_rtde at
module scope, which isn't installed in a sim-only environment.
"""
from dataclasses import dataclass

import numpy as np


def approach_blocked_by_fixture(approach_position, potato_center, fixture_axis,
                                half_angle_deg):
    """True when an approach point falls inside the cone the mounting pin
    and its holder occupy.

    The potato is impaled on a pin, so a cone opening downward from the
    potato's centre is solid hardware. Eyes low on the potato have normals
    pointing down into it, and backing off along such a normal puts the
    approach point -- and the arm behind it -- exactly where the fixture
    is. The roll/tilt search cannot notice: it only asks whether a pose is
    kinematically reachable, and driving into a pin is perfectly reachable.

    `fixture_axis` points from the potato's centre toward the fixture
    (straight down, [0, 0, -1], for a potato sitting on top of a vertical
    pin).
    """
    offset = np.asarray(approach_position, dtype=float) - np.asarray(potato_center, dtype=float)
    distance = float(np.linalg.norm(offset))
    if distance < 1e-9:
        return True  # degenerate: inside the potato, treat as blocked

    axis = np.asarray(fixture_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    cos_to_axis = float(offset @ axis) / distance
    # plain bool, not np.bool_: this feeds straight into log lines and the
    # EyeAttempt records, and the early return above is a Python bool too
    return bool(cos_to_axis >= np.cos(np.radians(half_angle_deg)))


# Every way one eye's drilling attempt can end. The first four come from
# DrillOutcome (the insertion itself ran); the last two mean no insertion was
# attempted at all, and they are the ones that say something is wrong with the
# setup rather than with the force tuning.
ATTEMPT_STATUSES = (
    'reached',          # drilled to the commanded depth past contact
    'force_limit',      # hit max_force first
    'no_contact',       # fed the whole approach travel touching nothing
    'timeout',          # neither limit reached in time
    'unreachable',      # no roll/tilt candidate the arm would accept
    'fixture_blocked',  # every candidate approach came from inside the pin
)


@dataclass
class EyeAttempt:
    """One eye's outcome, recorded whether or not the drill ever ran."""
    index: int
    status: str
    depth_m: float = 0.0
    peak_force_n: float = 0.0
    tilt_deg: float = 0.0
    roll_deg: float = 0.0


def summarize_attempts(attempts):
    """Counts per status plus the success rate, in the shape the potato
    literature reports (complete / incomplete / missed)."""
    counts = {status: 0 for status in ATTEMPT_STATUSES}
    for attempt in attempts:
        counts[attempt.status] = counts.get(attempt.status, 0) + 1
    total = len(attempts)
    return {
        'total': total,
        'counts': counts,
        'success_rate': (counts['reached'] / total) if total else 0.0,
    }


def format_attempt_table(attempts):
    """Human-readable per-eye table plus the summary, for the end-of-run log.

    Worth printing even when everything succeeded: the tilt column shows how
    much task tolerance the run actually needed, which is the early warning
    that the fixture or potato_center is drifting out of a comfortable pose.
    """
    lines = ['', 'per-eye results:',
             f'  {"eye":>4} | {"status":<15} | {"depth":>8} | {"peak force":>10} | '
             f'{"tilt":>6} | {"roll":>6}',
             '  ' + '-' * 66]
    for attempt in attempts:
        lines.append(
            f'  {attempt.index:>4} | {attempt.status:<15} | '
            f'{attempt.depth_m * 1000:>6.1f}mm | {attempt.peak_force_n:>8.1f}N | '
            f'{attempt.tilt_deg:>5.1f}d | {attempt.roll_deg:>5.0f}d')

    summary = summarize_attempts(attempts)
    lines.append('  ' + '-' * 66)
    lines.append(f'  {summary["total"]} eyes, '
                 f'{summary["counts"]["reached"]} drilled to depth '
                 f'({summary["success_rate"] * 100:.1f}%)')
    breakdown = ', '.join(f'{status}={count}'
                          for status, count in summary['counts'].items()
                          if count and status != 'reached')
    if breakdown:
        lines.append(f'  failures: {breakdown}')
    return '\n'.join(lines)


@dataclass
class DrillOutcome:
    """Result of one force_drill insertion.

    Truthy iff the drill actually reached `max_depth` PAST the detected
    contact point, so existing `reached = robot.force_drill(...)` /
    `if not reached:` call sites keep working unchanged -- while gaining
    the detail needed to tell the failure modes apart, which call for
    opposite responses:

      'reached'      insertion went the full commanded depth into the potato
      'force_limit'  hit max_force first -- feed_force/max_depth too
                     aggressive for this potato, or the bit is binding
      'no_contact'   fed the entire allowed approach travel without the
                     force ever rising past contact_force: nothing was
                     there. The eye position/normal is wrong, or
                     potato_center/the fixture moved -- NOT a tuning
                     problem, and drilling deeper would not have helped
      'timeout'      neither limit reached before timeout_s

    depth_m is penetration measured from the contact point (0.0 when
    never contacted), NOT travel from the approach standoff point.
    """
    status: str
    depth_m: float = 0.0
    peak_force_n: float = 0.0
    contacted: bool = False

    def __bool__(self):
        return self.status == 'reached'


# Weight (metres per radian) on the orientation term of the travel cost
# below. Two eyes a few millimetres apart on a potato surface can have
# normals tens of degrees apart, and the drill has to re-aim between them:
# the TCP barely moves while the wrist swings a long way. Costing position
# alone calls those "adjacent" and visits them back to back, which is
# exactly the sequence that drags the bit across the potato or runs the
# wrist through a singularity. 0.1 m/rad makes a 60deg re-aim cost about
# as much as 105mm of travel.
ORIENTATION_WEIGHT = 0.1


def _step_costs(points, normals, orientation_weight=ORIENTATION_WEIGHT):
    """Cost of each consecutive hop along a tour: straight-line distance,
    plus the angle between the two surface normals when they are known."""
    cost = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if normals is None:
        return cost
    cos_angle = np.clip(np.einsum('ij,ij->i', normals[:-1], normals[1:]), -1.0, 1.0)
    return cost + orientation_weight * np.arccos(cos_angle)


def _tour_length(order, points, normals=None,
                 orientation_weight=ORIENTATION_WEIGHT):
    ordered_normals = None if normals is None else normals[order]
    return float(np.sum(_step_costs(points[order], ordered_normals, orientation_weight)))


def nearest_neighbor_order(points, start_index=0, normals=None,
                           orientation_weight=ORIENTATION_WEIGHT):
    n = len(points)
    visited = [False] * n
    order = [start_index]
    visited[start_index] = True
    for _ in range(n - 1):
        last = order[-1]
        dists = np.linalg.norm(points - points[last], axis=1)
        if normals is not None:
            cos_angle = np.clip(normals @ normals[last], -1.0, 1.0)
            dists = dists + orientation_weight * np.arccos(cos_angle)
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        order.append(nxt)
        visited[nxt] = True
    return order


def two_opt(order, points, max_passes=50, normals=None,
            orientation_weight=ORIENTATION_WEIGHT):
    order = list(order)
    best = _tour_length(order, points, normals, orientation_weight)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                candidate = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                candidate_length = _tour_length(candidate, points, normals, orientation_weight)
                if candidate_length < best - 1e-9:
                    order = candidate
                    best = candidate_length
                    improved = True
    return order


def plan_visit_order(positions, start_position=None, normals=None,
                     orientation_weight=ORIENTATION_WEIGHT):
    """positions: (N,3) array of eye positions in base frame.
    normals: optional (N,3) array of the matching outward surface normals.
    When given, the travel cost also charges for how far the drill has to
    re-aim between consecutive eyes (see ORIENTATION_WEIGHT), so the tour
    stops treating "close together but facing opposite ways" as cheap.

    Returns a list of indices giving an efficient visiting order,
    starting from whichever eye is closest to `start_position` (e.g. the
    robot's current TCP position) if given."""
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n <= 1:
        return list(range(n))

    if normals is not None:
        normals = np.asarray(normals, dtype=float)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    if start_position is not None:
        start_index = int(np.argmin(np.linalg.norm(positions - np.asarray(start_position), axis=1)))
    else:
        start_index = 0

    order = nearest_neighbor_order(positions, start_index, normals, orientation_weight)
    if n <= 12:  # 2-opt is O(n^2) per pass -- fine for typical eye counts per potato
        order = two_opt(order, positions, normals=normals,
                        orientation_weight=orientation_weight)
    return order


def approach_pose(position, normal, standoff):
    """Point `standoff` meters back along the outward normal from the
    eye -- the pre-drill approach point that the drill then plunges
    inward from. Equivalent to approach_pose_along_axis with the tool
    aimed straight down the normal (zero tilt)."""
    position = np.asarray(position, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    return position + normal * standoff


def approach_pose_along_axis(position, tool_z_axis, standoff):
    """Point `standoff` metres back along the drill own insertion axis
    (the tool frame +Z, which points into the surface).

    This is what keeps a TILTED approach honest. force_drill feeds along
    +Z from wherever it starts, so the start point has to sit on the line
    through the eye along that same axis. Backing off along the surface
    normal instead, while aiming the tool somewhere else, lands the bit
    standoff*sin(tilt) away from the eye -- about 8mm of pure miss at a
    30mm standoff and only 15deg of tilt.
    """
    position = np.asarray(position, dtype=float)
    tool_z_axis = np.asarray(tool_z_axis, dtype=float)
    tool_z_axis = tool_z_axis / np.linalg.norm(tool_z_axis)
    return position - tool_z_axis * standoff


def helical_cut_path(contact_position, tool_z_axis, depth, lateral_radius,
                     turns=2.0, points_per_turn=16):
    """Waypoints that widen a bored hole into a conical pocket on the way out.

    The plunge (robot_interface.force_drill) has already cut a hole of the
    bit's own diameter down to `depth`, measured from the contact point. This
    is the pass that follows: a helix that spirals outward as it rises, from
    the bottom of that hole (radius 0) to `lateral_radius` at the surface.

    That shape is the point. Removing a sprout eye is not the same as
    sampling tissue -- the literature's biopsy punch bores a straight core
    and tilts to detach it, while the pineapple machines cut a cone and lift
    the plug out. A cone is the right shape here, and it also means the
    widening pass IS the retraction: the tool ends at the surface having
    already left the material, with no separate withdrawal.

    The parameterisation is deliberately one that subsumes both options the
    project had not decided between, so the choice can be made by measurement
    (or learned) rather than up front:

        lateral_radius == 0     a straight pull-out: the motion is exactly
                                the bore the plunge already made
        lateral_radius > 0      a conical scoop, wider at the skin

    Returns (N, 3) positions in the base frame, ordered bottom to surface.
    `tool_z_axis` is the insertion axis, pointing INTO the surface, so the
    hole bottom sits at contact + axis * depth.
    """
    contact_position = np.asarray(contact_position, dtype=float)
    axis = np.asarray(tool_z_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    # any two unit vectors spanning the plane the helix turns in
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(reference, axis)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    n_points = max(2, int(round(turns * points_per_turn)) + 1)
    s = np.linspace(0.0, 1.0, n_points)          # 0 at the bottom, 1 at the surface
    angle = 2.0 * np.pi * turns * s
    radius = lateral_radius * s

    return (contact_position
            + np.outer(depth * (1.0 - s), axis)
            + np.outer(radius * np.cos(angle), u)
            + np.outer(radius * np.sin(angle), v))


# Puncture stress of raw potato flesh through the skin, derived from a
# published penetration test: a 2 mm cylindrical probe driven 15 mm at
# 1.5 mm/s peaked at 41.2-47.2 N across seasons on cv. Spunta. A 2 mm probe
# presents pi * (1 mm)^2 = 3.14 mm^2, so that is 13.1-15.0 MPa; the lower
# bound is used, being the conservative one for a part that must NOT
# penetrate. Denser cultivars are tougher (a needle probe on cv. Kufri
# Badshah peaked at 79 N), so a collar sized against the soft end is sized
# against the worst case for its own job.
POTATO_PUNCTURE_STRESS_PA = 13.1e6


def depth_collar_spec(bit_diameter_m, max_depth_m, max_force_n,
                      puncture_stress_pa=POTATO_PUNCTURE_STRESS_PA,
                      safety_factor=4.0):
    """Dimensions for a passive collar that bounds penetration mechanically.

    Software decides how deep to go; this is the part that makes it true
    anyway. The precedent is direct: the tissue-sampling robot this project
    is closest to had its vision compute too deep a target in 17 of 81
    trials, and the biopsy punch's hub -- simply wider than the blade --
    stopped every one of them at the intended 7 mm. A mechanical ceiling
    absorbed a software error, which is the whole design pattern, and the
    same idea is standard as depth-limiting collars and stops on surgical
    drills.

    It composes particularly well with force control. The collar meeting the
    skin is a sudden rise in contact force, so force_drill sees max_force and
    stops cleanly. In a position-controlled cell the same contact instead
    becomes a following error that accumulates over a run -- which is what
    the tissue-sampling robot reported, and what this cell structurally does
    not have.

    Returns the offset of the collar face from the bit tip (equal to the
    intended depth, so the face meets the skin exactly as the tip reaches
    it), the smallest outer diameter that will not itself puncture at
    max_force, and the pressure it would see there.
    """
    bit_area = 3.141592653589793 * (bit_diameter_m / 2.0) ** 2
    allowed_pressure = puncture_stress_pa / safety_factor
    required_area = max_force_n / allowed_pressure
    outer_diameter = 2.0 * ((required_area + bit_area) / 3.141592653589793) ** 0.5

    return {
        'offset_from_tip_m': float(max_depth_m),
        'min_outer_diameter_m': float(outer_diameter),
        'contact_area_m2': float(required_area),
        'pressure_at_max_force_pa': float(max_force_n / required_area),
        'puncture_stress_pa': float(puncture_stress_pa),
        'safety_factor': float(safety_factor),
    }
