"""Checks the fixture keep-out cone, the live potato-centre fit, and the
per-eye result table, against the real shipped code.

The pose convention lives in drill_task_planner, which is pure, so these
import the same functions the controller delegates to rather than a copy.
"""
import numpy as np
import pytest

from potato_scan.drill_task_planner import (
    ROLL_SEARCH_DEG, EyeAttempt, approach_blocked_by_fixture,
    approach_candidates, approach_pose, format_attempt_table,
    summarize_attempts, tilt_search_sequence)
from potato_scan.surface_coverage import SurfaceCoverageGrid, fit_sphere

CENTER = np.array([0.50, 0.00, 0.15])
POTATO_RADIUS = 0.035
STANDOFF = 0.03
PIN_AXIS = np.array([0.0, 0.0, -1.0])
HALF_ANGLE_DEG = 35.0
TILT_SEARCH = tilt_search_sequence(15.0, 7.5)


def candidates_for(normal):
    eye = CENTER + np.asarray(normal, dtype=float) * POTATO_RADIUS
    return list(approach_candidates(eye, normal, STANDOFF,
                                    tilt_search_deg=TILT_SEARCH,
                                    roll_search_deg=ROLL_SEARCH_DEG))


# --- 1. the fixture keep-out cone ----------------------------------------

@pytest.mark.parametrize('elevation_deg,blocked', [
    (90, False), (45, False), (0, False), (-30, False), (-50, False),
    (-60, True), (-80, True), (-90, True)])
def test_the_cone_boundary_sits_where_the_pin_puts_it(elevation_deg, blocked):
    """A 35 deg half-angle cone around a downward pin blocks everything below
    -55 deg of elevation, and nothing above it."""
    normal = np.array([np.cos(np.radians(elevation_deg)), 0.0,
                       np.sin(np.radians(elevation_deg))])
    approach = approach_pose(CENTER + normal * POTATO_RADIUS, normal, STANDOFF)
    assert approach_blocked_by_fixture(
        approach, CENTER, PIN_AXIS, HALF_ANGLE_DEG) is blocked


def test_an_eye_facing_straight_down_has_every_candidate_blocked():
    """15 deg of tilt tolerance cannot swing an approach out of a 35 deg cone,
    so the controller must report fixture_blocked rather than hunting."""
    found = candidates_for(np.array([0.0, 0.0, -1.0]))
    blocked = sum(approach_blocked_by_fixture(a, CENTER, PIN_AXIS, HALF_ANGLE_DEG)
                  for a, _, _, _ in found)
    assert blocked == len(found) > 0


def test_an_eye_facing_sideways_has_no_candidate_blocked():
    found = candidates_for(np.array([1.0, 0.0, 0.0]))
    blocked = sum(approach_blocked_by_fixture(a, CENTER, PIN_AXIS, HALF_ANGLE_DEG)
                  for a, _, _, _ in found)
    assert blocked == 0 and len(found) > 0


# --- 2. the live potato-centre fit ---------------------------------------

@pytest.fixture(scope='module')
def rng():
    return np.random.default_rng(7)


def partial_sphere(rng, center, radius, n, min_dot, noise=0.0005):
    """Only the part of a sphere the scan has reached so far."""
    points = []
    while len(points) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        if v[2] > min_dot:
            points.append(center + v * radius + rng.normal(scale=noise, size=3))
    return np.array(points)


@pytest.fixture(scope='module')
def scan(rng):
    """A bigger potato than configured, so it sits higher on the pin."""
    true_center = CENTER + np.array([0.002, -0.003, 0.012])
    return true_center, partial_sphere(rng, true_center, POTATO_RADIUS, 2500, min_dot=-0.2)


def test_the_sphere_fit_finds_the_centre_from_a_partial_scan(scan):
    true_center, cloud = scan
    fitted, _ = fit_sphere(cloud)
    assert np.linalg.norm(fitted - true_center) < 0.002


def test_estimate_center_refines_the_configured_guess(scan):
    true_center, cloud = scan
    got = SurfaceCoverageGrid().estimate_center(
        cloud, CENTER, max_shift=0.03, min_points=800)
    assert got is not None
    assert np.linalg.norm(got - true_center) < 0.002


def test_too_few_points_refuses_rather_than_fitting_noise(scan):
    _, cloud = scan
    assert SurfaceCoverageGrid().estimate_center(
        cloud[:200], CENTER, 0.03, min_points=800) is None


def test_a_fit_beyond_max_shift_is_refused(rng):
    """The fit refines a configured centre, it does not search for one. A
    sphere 90 mm away is the fixture or the background, not the potato."""
    far_cloud = partial_sphere(rng, CENTER + np.array([0.0, 0.0, 0.09]),
                               POTATO_RADIUS, 2500, min_dot=-0.2)
    assert SurfaceCoverageGrid().estimate_center(
        far_cloud, CENTER, max_shift=0.03, min_points=800) is None


def test_re_running_from_the_fitted_centre_does_not_churn(scan):
    """Once converged it must stop moving, or the scan re-plans every frame."""
    _, cloud = scan
    grid = SurfaceCoverageGrid()
    got = grid.estimate_center(cloud, CENTER, max_shift=0.03, min_points=800)
    assert grid.estimate_center(cloud, got, max_shift=0.03, min_points=800) is None


# --- 3. the per-eye result table -----------------------------------------

@pytest.fixture
def attempts():
    return [EyeAttempt(0, 'reached', 0.0080, 21.4, 0.0, 0),
            EyeAttempt(3, 'reached', 0.0080, 26.9, 7.5, -45),
            EyeAttempt(1, 'force_limit', 0.0031, 40.0, 0.0, 90),
            EyeAttempt(4, 'fixture_blocked'),
            EyeAttempt(2, 'no_contact', 0.0, 1.2, 15.0, 135)]


def test_the_summary_counts_every_outcome_separately(attempts):
    summary = summarize_attempts(attempts)
    assert summary['total'] == 5
    assert summary['counts']['reached'] == 2
    assert summary['counts']['fixture_blocked'] == 1
    assert summary['success_rate'] == pytest.approx(0.4)


def test_an_empty_run_does_not_divide_by_zero():
    assert summarize_attempts([])['success_rate'] == 0.0


def test_every_attempt_appears_in_the_printed_table(attempts):
    table = format_attempt_table(attempts)
    for attempt in attempts:
        assert attempt.status in table
