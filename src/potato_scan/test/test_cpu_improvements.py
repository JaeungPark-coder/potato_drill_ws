"""Checks the three CPU-only improvements against the real shipped code:
orientation-aware tour cost, tilt-tolerant approach search, and the coverage
contamination that per-frame outlier removal fixes."""
import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from potato_scan.drill_task_planner import (
    ROLL_SEARCH_DEG, _tour_length, approach_candidates, approach_pose,
    plan_visit_order, tilt_search_sequence)
from potato_scan.surface_coverage import SurfaceCoverageGrid

CENTER = np.array([0.50, 0.0, 0.15])
POTATO_RADIUS = 0.035
STANDOFF = 0.03


# --- 1. does the orientation term change the visit order? ----------------

# Two eyes on opposite sides of a narrow waist: 10 mm apart in space, but
# their normals face nearly opposite ways. A third sits further away in
# distance while sharing the first one's orientation.
EYES = np.array([
    [0.500, 0.000, 0.185],      # 0: top, normal up
    [0.500, 0.008, 0.179],      # 1: closest to 0 in space, normal nearly opposite
    [0.500, -0.030, 0.180],     # 2: further from 0, normal up-ish
])
EYE_NORMALS = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 0.95, -0.31],
    [0.0, -0.40, 0.92],
])
EYE_NORMALS = EYE_NORMALS / np.linalg.norm(EYE_NORMALS, axis=1, keepdims=True)


def total_reaim_deg(order):
    normals = EYE_NORMALS[order]
    return float(np.sum(np.degrees(np.arccos(np.clip(
        np.einsum('ij,ij->i', normals[:-1], normals[1:]), -1.0, 1.0)))))


def test_position_alone_takes_the_nearest_eye_next():
    assert plan_visit_order(EYES, start_position=EYES[0]) == [0, 1, 2]


def test_the_orientation_term_defers_the_re_aim_heavy_hop():
    assert plan_visit_order(EYES, start_position=EYES[0],
                            normals=EYE_NORMALS) == [0, 2, 1]


def test_the_orientation_aware_tour_actually_re_aims_less():
    """The property the reordering exists for, rather than just a different
    sequence: total wrist rotation over the tour goes down."""
    position_only = plan_visit_order(EYES, start_position=EYES[0])
    orientation_aware = plan_visit_order(EYES, start_position=EYES[0],
                                         normals=EYE_NORMALS)
    assert total_reaim_deg(orientation_aware) < total_reaim_deg(position_only)


def test_no_normals_reproduces_the_old_cost_exactly():
    straight_line = float(np.sum(np.linalg.norm(np.diff(EYES, axis=0), axis=1)))
    assert _tour_length([0, 1, 2], EYES) == pytest.approx(straight_line, abs=1e-12)


# --- 2. tilt tolerance without missing the eye ---------------------------

@pytest.mark.parametrize('max_tilt,step,expected', [
    (15.0, 7.5, [0.0, 7.5, 15.0]),
    (0.0, 7.5, [0.0]),
    (10.0, 7.5, [0.0, 7.5, 10.0]),      # the remainder is kept, not dropped
])
def test_the_tilt_ladder_ends_on_the_configured_maximum(max_tilt, step, expected):
    assert tilt_search_sequence(max_tilt, step) == expected


@pytest.fixture(scope='module')
def candidates():
    eye = np.array([0.500, 0.010, 0.170])
    normal = np.array([0.3, -0.5, 0.81])
    normal = normal / np.linalg.norm(normal)
    found = list(approach_candidates(eye, normal, STANDOFF,
                                     tilt_search_deg=tilt_search_sequence(15.0, 7.5),
                                     roll_search_deg=ROLL_SEARCH_DEG))
    return eye, normal, found


def test_every_tilt_is_tried_at_every_roll(candidates):
    _, _, found = candidates
    assert len(found) == 3 * len(ROLL_SEARCH_DEG) == 24


def test_candidates_are_ordered_smallest_deviation_first(candidates):
    """So the controller settles for a tilted approach only after the
    straight one has been refused at every roll."""
    _, _, found = candidates
    tilts = [tilt for _, _, tilt, _ in found]
    assert tilts == sorted(tilts)


def test_every_candidate_still_lands_on_the_eye(candidates):
    """The bug this catches: backing off along the normal while tilting the
    tool misses by standoff * sin(tilt) -- 7.8 mm at 15 degrees."""
    eye, _, found = candidates
    worst = max(np.linalg.norm(
        (approach + Rot.from_rotvec(rotvec).as_matrix()[:, 2] * STANDOFF) - eye)
        for approach, rotvec, _, _ in found)
    assert worst < 1e-12, f'{worst * 1e6:.3f} um'


def test_the_tilt_is_exactly_the_requested_deviation_from_the_normal(candidates):
    _, normal, found = candidates
    for _, rotvec, tilt_deg, _ in found:
        tool_z = Rot.from_rotvec(rotvec).as_matrix()[:, 2]
        # arccos is ill-conditioned near dot == 1, so ~1e-6 deg is the floor
        got = np.degrees(np.arccos(np.clip(tool_z @ (-normal), -1.0, 1.0)))
        assert got == pytest.approx(tilt_deg, abs=1e-4)


def test_zero_tilt_reproduces_the_previous_behaviour_exactly(candidates):
    eye, normal, found = candidates
    old = approach_pose(eye, normal, STANDOFF)
    zero_tilt = [approach for approach, _, tilt, _ in found if tilt == 0.0]
    assert len(zero_tilt) == len(ROLL_SEARCH_DEG)
    for approach in zero_tilt:
        assert np.allclose(approach, old, atol=1e-15)


# --- 3. flying pixels falsely filling coverage cells ---------------------

@pytest.fixture(scope='module')
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope='module')
def contaminated(rng):
    """A half-scanned potato plus a rim of flying pixels.

    Flying pixels form ALONG the silhouette edge, which in a depth image is a
    long dense curve, and each is strung out along its own viewing ray -- so
    they are tightly grouped in direction but scattered in radius. That is
    what lets them fill coverage cells no camera ever saw.
    """
    surface = []
    while len(surface) < 6000:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if v[1] > 0.15:                     # only the +Y half has been scanned
            surface.append(CENTER + v * POTATO_RADIUS)
    surface = np.array(surface)

    flying = []
    for t in np.linspace(0.0, 2.0 * np.pi, 22, endpoint=False):
        for _ in range(5):
            v = np.array([np.cos(t), 0.08, np.sin(t)]) + rng.normal(scale=0.01, size=3)
            v /= np.linalg.norm(v)
            if v[1] > 0.15:                 # strictly outside the scanned half
                continue
            flying.append(CENTER + v * (POTATO_RADIUS + rng.uniform(-0.015, 0.015)))

    return surface, np.array(flying)


def filled_cells(points):
    grid = SurfaceCoverageGrid()
    grid.set_from_points(points, CENTER)
    return grid.filled_mask()


def statistical_outlier_mask(points, k=50, std_ratio=1.0):
    """Open3D's remove_statistical_outlier, applied directly.

    Open3D needs Python <= 3.12 and is not importable on every development
    machine, so the definition is reproduced rather than imported: drop points
    whose mean distance to their k nearest neighbours exceeds
    mean + std_ratio * std over the frame.
    """
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(k, len(points) - 1) + 1)
    mean_distance = distances[:, 1:].mean(axis=1)
    return mean_distance <= mean_distance.mean() + std_ratio * mean_distance.std()


def test_flying_pixels_do_falsely_fill_coverage_cells(contaminated):
    """Guards the guard: if the fixture stopped reproducing the contamination
    the sweep below would be filtering a problem that was not there."""
    surface, flying = contaminated
    clean = filled_cells(surface)
    dirty = filled_cells(np.vstack([surface, flying]))
    assert int(np.sum(dirty & ~clean)) > 0


@pytest.mark.parametrize('std_ratio', [1.0, 1.5, 2.0, 2.5])
def test_outlier_removal_reduces_false_coverage_at_every_setting(contaminated, std_ratio):
    surface, flying = contaminated
    combined = np.vstack([surface, flying])
    clean = filled_cells(surface)
    before = int(np.sum(filled_cells(combined) & ~clean))

    kept = filled_cells(combined[statistical_outlier_mask(combined, 50, std_ratio)])
    assert int(np.sum(kept & ~clean)) < before


def test_the_shipped_std_ratio_eliminates_false_coverage_without_losing_cells(contaminated):
    """params.yaml ships std_ratio 1.5.

    The asymmetry that picks it: a cell wrongly marked COVERED is never
    revisited, so the reconstruction keeps a hole it believes is filled. A
    cell wrongly left OPEN just costs one more view in gap-filling. So false
    coverage is the error worth spending real points to avoid.
    """
    surface, flying = contaminated
    combined = np.vstack([surface, flying])
    clean = filled_cells(surface)

    kept = filled_cells(combined[statistical_outlier_mask(combined, 50, 1.5)])
    assert int(np.sum(kept & ~clean)) == 0, 'no cell may be falsely covered'
    assert int(np.sum(clean & ~kept)) <= 3, 'and few real cells may be lost'
