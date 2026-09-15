"""The Isaac Sim potato, generated without Isaac Sim: the geometry has to be
the one make_potato_mesh always built (or a seed logged by isaac_scene.py
reproduces a different potato than the one it scanned), its ground truth
has to sit on its own surface, and its bumps alone must not read as eyes.
"""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from potato_scan.detection_accuracy import match_detections
from potato_scan.procedural_potato import (
    EYE_DEPTH_M, EYE_SIGMA_RAD, generate, sample_surface, visible_from)
from potato_scan.surface_curvature import find_eye_candidates
from potato_scan import sim_detection_check

CENTER = np.array([0.50, 0.0, 0.15])


def reference_loop(center, base_radius=0.035, bumpiness=0.35, n_lat=60, n_lon=90, seed=None):
    """make_potato_mesh's original per-vertex loop, verbatim, as the oracle."""
    rng = np.random.default_rng(seed)
    lats = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lons = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)

    n_bumps = int(rng.integers(4, 7))
    bump_dirs = rng.normal(size=(n_bumps, 3))
    bump_dirs /= np.linalg.norm(bump_dirs, axis=1, keepdims=True)
    bump_amp = rng.uniform(0.1, 1.0, size=n_bumps) * bumpiness * base_radius
    bump_width = rng.uniform(0.4, 1.0, size=n_bumps)

    n_eyes = int(rng.integers(3, 8))
    eye_dirs = rng.normal(size=(n_eyes, 3))
    eye_dirs /= np.linalg.norm(eye_dirs, axis=1, keepdims=True)
    eye_depth = rng.uniform(0.7, 1.3) * EYE_DEPTH_M
    eye_sigma = rng.uniform(0.85, 1.15) * EYE_SIGMA_RAD

    points = []
    for lat in lats:
        for lon in lons:
            d = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
            r = base_radius
            for bd, amp, w in zip(bump_dirs, bump_amp, bump_width):
                r += amp * max(0.0, float(np.dot(d, bd))) ** (1.0 / w)
            for ed in eye_dirs:
                angle = np.arccos(np.clip(float(np.dot(d, ed)), -1.0, 1.0))
                r -= eye_depth * np.exp(-(angle ** 2) / (2.0 * eye_sigma ** 2))
            points.append(center + d * r)

    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            a, b = i * n_lon + j, i * n_lon + j2
            c, dd = (i + 1) * n_lon + j2, (i + 1) * n_lon + j
            faces.append((a, b, c, dd))
    return np.array(points), np.array(faces), eye_dirs, bump_dirs


# --- 1. it is the same potato Kit builds ----------------------------------

@pytest.mark.parametrize('seed', [0, 3, 41, 2 ** 31 - 2])
def test_the_vectorised_geometry_is_the_original_loop_vertex_for_vertex(seed):
    points, faces, eye_dirs, bump_dirs = reference_loop(CENTER, n_lat=14, n_lon=20, seed=seed)
    geometry = generate(CENTER, n_lat=14, n_lon=20, seed=seed)
    assert np.allclose(geometry.points, points, atol=1e-12)
    assert np.array_equal(geometry.faces, faces)
    assert np.allclose(geometry.eye_dirs, eye_dirs)
    assert np.allclose(geometry.bump_dirs, bump_dirs)


def test_a_seed_is_recorded_and_reproduces_the_potato():
    first = generate(CENTER, n_lat=14, n_lon=20)
    again = generate(CENTER, n_lat=14, n_lon=20, seed=first.seed)
    assert np.array_equal(first.points, again.points)
    assert np.array_equal(first.eye_points, again.eye_points)


def test_removing_the_eyes_keeps_the_same_bumps():
    """The same seed's draws in the same order, only the pit depth zeroed --
    so a detection on the eyeless surface is attributable to the bumps."""
    with_eyes = generate(CENTER, n_lat=14, n_lon=20, seed=7)
    without = generate(CENTER, n_lat=14, n_lon=20, seed=7, with_eyes=False)
    assert np.array_equal(with_eyes.bump_dirs, without.bump_dirs)
    assert np.array_equal(with_eyes.bump_amp, without.bump_amp)
    assert np.array_equal(with_eyes.eye_dirs, without.eye_dirs)
    assert without.eye_depth == 0.0
    assert np.all(without.radius_toward(without.eye_dirs)
                  > with_eyes.radius_toward(with_eyes.eye_dirs))


# --- 2. the ground truth is on the surface --------------------------------

def test_every_ground_truth_eye_lies_on_the_surface_the_camera_sees():
    for seed in range(8):
        geometry = generate(CENTER, seed=seed)
        surface = sample_surface(geometry, voxel_size=0.001, rng=seed)
        distance, _ = cKDTree(surface).query(geometry.eye_points)
        # within a vertex spacing: the pit bottom sits between flat faces
        assert distance.max() < 0.002, (seed, distance)


def test_the_old_truth_formula_was_inside_the_potato_by_more_than_the_match_tolerance():
    """The bug this guards against: `base_radius - eye_depth` along eye_dir
    ignores the bumps, which add up to 12mm of radius. Over seeds 0-19 at
    least one eye lands past match_detections' 8mm tolerance, where a
    perfect detection scores as one miss plus one spurious."""
    worst = 0.0
    for seed in range(20):
        geometry = generate(CENTER, seed=seed)
        old = geometry.center + geometry.eye_dirs * (geometry.base_radius - geometry.eye_depth)
        worst = max(worst, float(np.linalg.norm(old - geometry.eye_points, axis=1).max()))
    assert worst > 0.008


def test_a_perfect_detection_scores_as_a_match_against_the_corrected_truth():
    geometry = generate(CENTER, seed=5)
    result = match_detections(geometry.eye_points, geometry.eye_points)
    assert len(result['matches']) == len(geometry.eye_points)
    assert result['spurious'] == []


# --- 3. the sampler ---------------------------------------------------------

def test_sampled_points_sit_at_the_voxel_spacing():
    geometry = generate(CENTER, seed=1)
    surface = sample_surface(geometry, voxel_size=0.001, rng=1)
    spacing, _ = cKDTree(surface).query(surface, k=2)
    median = float(np.median(spacing[:, 1]))
    assert 0.0005 < median < 0.0015, median


def test_noise_moves_points_and_nothing_else():
    geometry = generate(CENTER, seed=1)
    quiet = sample_surface(geometry, rng=1)
    noisy = sample_surface(geometry, noise=0.00015, rng=1)
    assert quiet.shape == noisy.shape
    jitter = np.linalg.norm(noisy - quiet, axis=1)
    assert 0.0001 < np.median(jitter) < 0.0004


def test_bump_edge_offset_is_zero_on_the_rim():
    geometry = generate(CENTER, seed=2)
    bump = geometry.bump_dirs[0]
    # any direction perpendicular to the bump is on its rim
    rim = np.cross(bump, [0.0, 0.0, 1.0])
    rim /= np.linalg.norm(rim)
    offset, index = geometry.bump_edge_offset_deg(geometry.center + rim * 0.04)
    assert offset[0] == pytest.approx(0.0, abs=1e-6)
    assert index[0] == 0


def test_a_view_sees_the_near_side_and_not_the_far_side():
    geometry = generate(CENTER, seed=3)
    surface = sample_surface(geometry, rng=3)
    camera = CENTER + np.array([0.15, 0.0, 0.0])
    seen = visible_from(geometry, surface, camera)
    toward_camera = (surface - CENTER)[:, 0] > 0.02
    away = (surface - CENTER)[:, 0] < -0.02
    # not all of it: grazing and the bumps' own shadows take their share
    assert seen[toward_camera].mean() > 0.5
    assert not seen[away].any()


def test_the_full_raster_covers_the_grid_and_partial_cuts_grow_monotonically():
    geometry = generate(CENTER, seed=4)
    surface = sample_surface(geometry, rng=4)
    coverages = [sim_detection_check.partial_scan(geometry, surface, n)[1] for n in (1, 3, 10, 40)]
    assert all(b >= a for a, b in zip(coverages, coverages[1:])), coverages
    assert coverages[0] < 0.3
    assert coverages[-1] > 0.95


# --- 4. what the detector does with it ------------------------------------

@pytest.mark.slow
def test_the_bumps_alone_are_not_eyes():
    """The README's 2026-09-15 hypothesis for 8 spurious detections was a
    concave valley between two bumps. Noiseless, eyes removed, the
    detector finds nothing on any of these potatoes."""
    for seed in range(6):
        geometry = generate(CENTER, seed=seed, with_eyes=False)
        surface = sample_surface(geometry, rng=seed)
        assert find_eye_candidates(surface, CENTER, **sim_detection_check.DEFAULTS) == [], seed


@pytest.mark.slow
def test_what_is_found_is_found_where_it_is():
    """Noiseless, the carved pits the detector does accept are localised
    inside step 5's 2mm bar, with 0 spurious -- the corrected truth and the
    detector agree on where an eye is."""
    for seed in range(6):
        geometry = generate(CENTER, seed=seed)
        surface = sample_surface(geometry, rng=seed)
        candidates = find_eye_candidates(surface, CENTER, **sim_detection_check.DEFAULTS)
        result = match_detections([c['position'] for c in candidates], geometry.eye_points)
        assert result['spurious'] == [], seed
        for _, _, distance, _ in result['matches']:
            assert distance < 0.002


@pytest.mark.slow
def test_the_mean_curvature_gate_survives_the_noise_that_breaks_kappa():
    """Why the gate changed on 2026-09-16. At 0.25mm of scanner noise the
    kappa floor crosses 0.015 and the old gate accepts the whole surface;
    H is the fitted physical curvature and the same clouds stay clean.
    Three potatoes rather than thirty, so the fast suite is not slowed by
    a result the tool reproduces at full size in seconds."""
    kappa_gate = dict(sim_detection_check.DEFAULTS, mean_curvature_min=None, curvature_min=0.015)
    spurious_kappa = spurious_h = found_h = 0
    for seed in range(3):
        geometry = generate(CENTER, seed=seed)
        surface = sample_surface(geometry, noise=0.00025, rng=seed)
        old = find_eye_candidates(surface, CENTER, **kappa_gate)
        new = find_eye_candidates(surface, CENTER, **sim_detection_check.DEFAULTS)
        spurious_kappa += len(match_detections([c['position'] for c in old],
                                               geometry.eye_points)['spurious'])
        result = match_detections([c['position'] for c in new], geometry.eye_points)
        spurious_h += len(result['spurious'])
        found_h += len(result['matches'])
    assert spurious_kappa > 20
    assert spurious_h == 0
    assert found_h > 0


@pytest.mark.slow
def test_the_check_tool_runs_both_ways(capsys):
    sim_detection_check.main(['--seeds', '2'])
    sim_detection_check.main(['--seed', '0'])
    sim_detection_check.main(['--seeds', '2', '--views', '3'])
    out = capsys.readouterr().out
    assert '2 potatoes' in out
    assert 'potato seed 0' in out
    assert 'first 3 views (coverage' in out
    assert 'detection accuracy vs ground truth' in out
