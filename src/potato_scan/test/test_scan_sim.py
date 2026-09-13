"""Checks the CPU surface model, the view-budget sweep, and the RL env.

None of this needs a robot, a camera or Isaac Sim -- it is the geometric
half of the scan, and it is what tells you the package is installed and
working before anything is plugged in.
"""
import numpy as np
import pytest

from potato_scan.potato_surface import PotatoSurface
from potato_scan.rl.cpu_scan_env import CpuScanEnv
from potato_scan.scan_budget import smallest_sufficient, sweep

CENTER = np.array([0.50, 0.0, 0.15])
FRONT = np.array([1.0, 0.0, 0.0])
SCAN_RADIUS = 0.15


# --- 1. the surface model ------------------------------------------------

@pytest.fixture(scope='module')
def surface():
    return PotatoSurface(CENTER, np.random.default_rng(0), n_points=6000)


def test_the_surface_is_never_inside_out(surface):
    assert np.all(surface.radii > 0)


@pytest.mark.parametrize('label,kwargs', [
    ('camera below min range', dict(min_range=0.15)),
    ('narrow field of view', dict(fov_deg=15.0)),
    ('strict grazing cutoff', dict(max_grazing_deg=30.0)),
    ('beyond max range', dict(max_range=0.05)),
])
def test_each_rejection_mode_bites_independently(surface, label, kwargs):
    """Four separate ways a real camera misses a patch. If one of them stops
    rejecting anything, the coverage numbers below quietly become optimistic."""
    camera = CENTER + FRONT * SCAN_RADIUS
    assert surface.visible(camera, **kwargs).sum() < surface.visible(camera).sum()


@pytest.mark.parametrize('kwargs', [dict(min_range=0.15), dict(max_range=0.05)])
def test_a_camera_out_of_range_sees_nothing_at_all(surface, kwargs):
    camera = CENTER + FRONT * SCAN_RADIUS
    assert surface.visible(camera, **kwargs).sum() == 0


def test_nothing_facing_away_is_ever_returned(surface):
    camera = CENTER + FRONT * SCAN_RADIUS
    seen = surface.visible(camera)
    facing = (surface.directions @ FRONT) > 0
    assert not np.any(seen & ~facing)


# --- 2. the view-budget finding ------------------------------------------

@pytest.fixture(scope='module')
def budget_rows():
    """~20 s: three potatoes at 12000 points against all eight step sizes."""
    return sweep(n_potatoes=3, n_surface_points=12000)


@pytest.mark.slow
def test_far_fewer_than_forty_views_reach_full_coverage(budget_rows):
    best = smallest_sufficient(budget_rows, 1.0)
    assert best is not None, 'something on the grid must reach full coverage'
    assert best[0] <= 12, 'the point of the exercise is that 40 is not needed'


@pytest.mark.slow
def test_the_shipped_raster_saturates(budget_rows):
    """Which is exactly why Phase B has nothing left to do at that setting --
    see the docstring in rl/cpu_scan_env.py before training anything."""
    shipped = [row for row in budget_rows if row[0] == 40]
    assert shipped and shipped[0][4] >= 0.999


@pytest.mark.slow
def test_a_coarse_raster_still_leaves_real_gaps(budget_rows):
    """The regime where a gap-filling policy would have work to do. If this
    stops holding, the RL environment can no longer pose a problem at all."""
    coarse = [row for row in budget_rows if row[0] <= 6]
    assert coarse and coarse[0][3] < 0.95


# --- 3. the RL environment honours the deployed policy contract ----------

@pytest.fixture(scope='module')
def shipped_env():
    env = CpuScanEnv(seed=0, n_surface_points=6000)
    _, info = env.reset()
    return env, info


def test_the_observation_matches_the_deployed_spec(shipped_env):
    env, _ = shipped_env
    observation, _, _, _, _ = env.step(env.action_space.sample())
    assert env.observation_space.contains(observation)


def test_a_step_reports_everything_the_policy_is_scored_on(shipped_env):
    env, _ = shipped_env
    _, _, _, _, info = env.step(env.action_space.sample())
    for key in ('coverage', 'resolved', 'views_taken',
                'gap_filling_views', 'eye_points_seen'):
        assert key in info


def test_a_coarser_raster_leaves_more_for_the_policy(shipped_env):
    _, shipped_info = shipped_env
    hard = CpuScanEnv(seed=0, n_surface_points=6000,
                      azimuth_step_deg=180.0, elevation_step_deg=50.0)
    _, hard_info = hard.reset()
    assert hard_info['coverage_after_raster'] < shipped_info['coverage_after_raster']


def test_the_coarse_setting_gives_the_policy_more_than_one_move():
    hard = CpuScanEnv(seed=0, n_surface_points=6000,
                      azimuth_step_deg=180.0, elevation_step_deg=50.0)
    hard.reset()
    steps = 0
    for _ in range(hard.max_views):
        _, _, terminated, truncated, _ = hard.step(hard.action_space.sample())
        steps += 1
        if terminated or truncated:
            break
    assert steps > 1
