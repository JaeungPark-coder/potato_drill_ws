"""Validates the curvature maths against surfaces whose true curvature is
known analytically, then against a synthetic potato with dimples where it
has to actually separate eyes from the body.

This is the open3d-free replacement for the original detector, so the DBSCAN
contract it has to match is checked here too.
"""
import numpy as np
import pytest

from potato_scan.potato_surface import fibonacci_directions
from potato_scan.surface_curvature import (
    dbscan, describe_surface, estimate_normals, find_eye_candidates,
    luminance, neighbor_indices, neighborhood_eigen, principal_curvatures,
    shape_index, surface_variation)

CENTER = np.array([0.50, 0.0, 0.15])
SPHERE_RADIUS = 0.035
KNN = 30

EYE_DEPTH = 0.0035
EYE_SIGMA = 0.09
SCANNER_NOISE = 0.00015


@pytest.fixture(scope='module')
def rng():
    return np.random.default_rng(5)


# --- 1. the published Shape Index scale ----------------------------------

@pytest.mark.parametrize('label,k1,k2,expected', [
    ('cup (pit)', +1.0, +1.0, 0.00),
    ('rut (concave cylinder)', 0.0, +1.0, 0.25),
    ('saddle', +1.0, -1.0, 0.50),
    ('ridge (convex cylinder)', 0.0, -1.0, 0.75),
    ('dome (convex)', -1.0, -1.0, 1.00),
])
def test_the_canonical_shapes_land_on_the_published_values(label, k1, k2, expected):
    got = float(shape_index(np.array([max(k1, k2)]), np.array([min(k1, k2)]))[0])
    assert got == pytest.approx(expected, abs=1e-9), label


# --- 2. principal curvatures on an exact sphere --------------------------

@pytest.fixture(scope='module')
def sphere():
    points = fibonacci_directions(8000) * SPHERE_RADIUS
    indices = neighbor_indices(points, KNN)
    _, eigenvectors = neighborhood_eigen(points, indices)
    normals = estimate_normals(points, indices, eigenvectors, np.zeros(3))
    return points, indices, normals


def test_normals_are_oriented_outward(sphere):
    points, _, normals = sphere
    radial = points / np.linalg.norm(points, axis=1, keepdims=True)
    assert np.einsum('ij,ij->i', normals, radial).min() > 0.999


def test_estimated_curvature_matches_one_over_the_radius(sphere):
    """Sign convention: an outward normal on a convex surface gives -1/R."""
    points, indices, normals = sphere
    k1, _ = principal_curvatures(points, indices, normals)
    relative_error = abs((np.median(k1) + 1 / SPHERE_RADIUS) / (1 / SPHERE_RADIUS))
    assert relative_error < 0.05


def test_a_sphere_reads_as_a_dome(sphere):
    points, indices, normals = sphere
    k1, k2 = principal_curvatures(points, indices, normals)
    assert np.median(shape_index(k1, k2)) > 0.97


# --- 3. what surface variation is, and is not ----------------------------

def test_a_flat_plane_has_essentially_zero_surface_variation(rng):
    plane = np.column_stack([rng.uniform(-0.05, 0.05, 4000),
                             rng.uniform(-0.05, 0.05, 4000),
                             np.zeros(4000)])
    eigenvalues, _ = neighborhood_eigen(plane, neighbor_indices(plane, KNN))
    assert np.median(surface_variation(eigenvalues)) < 1e-6


def test_surface_variation_is_not_scale_invariant():
    """Stated as a test because the docstring once claimed the opposite.

    kappa is dimensionless and bounded, which is what makes it interpretable,
    but a denser scan gives a smaller neighbourhood and reads flatter -- so
    curvature_min has to be set for the scan density in use. Only the Shape
    Index is genuinely scale-invariant.
    """
    medians = []
    for n_points in (3000, 8000, 20000):
        points = fibonacci_directions(n_points) * SPHERE_RADIUS
        eigenvalues, _ = neighborhood_eigen(points, neighbor_indices(points, KNN))
        medians.append(float(np.median(surface_variation(eigenvalues))))

    assert medians == sorted(medians, reverse=True), medians
    assert medians[0] > 4 * medians[-1], 'the density effect is large, not marginal'


# --- 4. a synthetic potato with known eyes -------------------------------

def modulated_radius(directions):
    """A lumpy ellipsoid, before the eyes are pressed into it."""
    return SPHERE_RADIUS * (1.0 + 0.18 * directions[:, 2] ** 2
                            - 0.06 * directions[:, 0] ** 2)


def press_dimples(radius, directions, dimple_directions):
    """Gaussian pits in the radial direction; returns the ground-truth mask."""
    inside = np.zeros(len(directions), dtype=bool)
    for direction in dimple_directions:
        angle = np.arccos(np.clip(directions @ direction, -1.0, 1.0))
        radius = radius - EYE_DEPTH * np.exp(-(angle ** 2) / (2 * EYE_SIGMA ** 2))
        inside |= angle < EYE_SIGMA
    return radius, inside


def dimple_positions(dimple_directions):
    return CENTER + dimple_directions * (
        modulated_radius(dimple_directions) - EYE_DEPTH)[:, None]


@pytest.fixture(scope='module')
def potato(rng):
    """24000 points, four eyes, scanner noise. (points, is_eye, truth)"""
    directions = fibonacci_directions(24000)
    eye_directions = fibonacci_directions(4)
    radius, is_eye = press_dimples(modulated_radius(directions),
                                   directions, eye_directions)
    points = CENTER + directions * radius[:, None]
    points = points + rng.normal(scale=SCANNER_NOISE, size=points.shape)
    return points, is_eye, dimple_positions(eye_directions)


@pytest.fixture(scope='module')
def described(potato):
    points, _, _ = potato
    return describe_surface(points, CENTER, knn=KNN)


@pytest.mark.slow
def test_the_body_reads_as_ridge_to_dome_and_an_eye_as_a_cup(potato, described):
    _, is_eye, _ = potato
    s = described.shape_index
    body, eye = np.median(s[~is_eye]), np.median(s[is_eye])
    assert body > 0.6
    assert eye < 0.4
    assert body - eye > 0.4, 'separation on the 0-1 scale'


@pytest.mark.slow
def test_the_shape_index_buys_precision_curvature_alone_cannot(potato, described):
    """The reason the filter has two axes rather than one (with kappa as
    the curvature axis, as it was when this was established)."""
    _, is_eye, _ = potato
    kappa, s = described.kappa, described.shape_index
    curvature_min, shape_index_max = 0.002, 0.4

    def precision(selected):
        hits = int((selected & is_eye).sum())
        return hits / max(int(selected.sum()), 1)

    both = (kappa > curvature_min) & (s < shape_index_max)
    kappa_only = kappa > curvature_min
    assert precision(both) > precision(kappa_only)


# --- 5. the DBSCAN contract inherited from Open3D ------------------------

@pytest.fixture
def blobs(rng):
    return np.vstack([
        rng.normal(loc=(0, 0, 0), scale=0.0008, size=(40, 3)),
        rng.normal(loc=(0.05, 0, 0), scale=0.0008, size=(40, 3)),
        rng.normal(loc=(0, 0.05, 0), scale=0.0008, size=(40, 3)),
        np.array([[0.2, 0.2, 0.2]])])          # a lone outlier


def test_three_blobs_and_an_outlier_come_back_as_three_clusters(blobs):
    labels = dbscan(blobs, eps=0.003, min_points=8)
    clusters = sorted(set(labels) - {-1})
    assert len(clusters) == 3
    assert (labels == -1).sum() == 1
    assert sorted(int((labels == c).sum()) for c in clusters) == [40, 40, 40]


def test_min_points_actually_gates(blobs):
    assert len(set(dbscan(blobs, eps=0.003, min_points=100)) - {-1}) == 0


# --- 6. end to end -------------------------------------------------------

def match(found, truth, tolerance=0.008):
    """(matched count, worst error AMONG MATCHED).

    Worst error is taken only over eyes that were actually found: mixing in
    the distance to a missed eye reports a localisation failure where the
    real failure was a detection one.
    """
    if not found:
        return 0, float('nan')
    errors = [min(np.linalg.norm(c['position'] - t) for c in found) for t in truth]
    hits = [e for e in errors if e < tolerance]
    return len(hits), (max(hits) if hits else float('nan'))


@pytest.mark.slow
def test_the_default_gate_finds_every_eye_with_no_false_positives(potato):
    points, _, truth = potato
    found = find_eye_candidates(points, CENTER, knn=KNN)
    matched, worst = match(found, truth)
    assert matched == 4
    assert len(found) == 4
    assert worst < 0.002, 'sub-2 mm localisation is the point of the shape index'


@pytest.mark.slow
def test_the_eyes_are_an_order_of_magnitude_more_curved_than_the_body(potato, described):
    """What the mean-curvature gate rests on: H is the surface's physical
    curvature, positive inward with outward normals, and a 3.5 mm pit of
    sigma 3 mm bends at roughly depth / sigma^2 ~ 350/m, where a 35 mm
    body bends at -1/R ~ -29/m. The 150/m default sits between them with
    room on both sides -- and that room is in 1/m, not in a quantity that
    moves with scan density."""
    _, is_eye, _ = potato
    h = described.mean_curvature
    assert np.median(h[~is_eye]) < 0.0, 'the body is convex: negative H'
    assert np.percentile(h[~is_eye], 99) < 150.0
    assert np.percentile(h[is_eye], 75) > 150.0


@pytest.mark.slow
def test_curvature_rejects_and_the_shape_index_is_the_saddle_guard(potato):
    """Curvature is what rejects: on the shape index alone most of what
    comes back is not an eye. With kappa as the curvature axis the shape
    index was also what localised (kappa-only drifted to 2.4 mm); with a
    SIGNED mean curvature the ridges and domes it used to remove are
    already negative, so on a potato of Gaussian pits H alone lands where
    both axes do. S stays for what H's sign cannot see -- a saddle can
    have H well above 150/m -- and because it costs nothing here."""
    points, _, truth = potato
    both = find_eye_candidates(points, CENTER, knn=KNN)
    matched, worst = match(both, truth)
    assert matched == 4 and worst < 0.002

    h_only = find_eye_candidates(points, CENTER, knn=KNN, shape_index_max=1.01)
    matched_h, worst_h = match(h_only, truth)
    assert matched_h == 4
    assert len(h_only) == len(both)
    assert worst_h < 0.002

    shape_only = find_eye_candidates(points, CENTER, knn=KNN, mean_curvature_min=None)
    matched_shape, _ = match(shape_only, truth)
    assert (len(shape_only) - matched_shape) > (len(both) - matched), \
        'dropping curvature costs rejection'


@pytest.mark.slow
def test_the_kappa_gate_still_works_when_asked_for(potato):
    """The gate this detector shipped with until 2026-09-16, kept behind
    curvature_min: on the synthetic it was validated against it still
    finds every eye, so anyone switching back gets what they had."""
    points, _, truth = potato
    found = find_eye_candidates(points, CENTER, knn=KNN,
                                mean_curvature_min=None, curvature_min=0.015)
    matched, worst = match(found, truth)
    assert matched == 4 and len(found) == 4 and worst < 0.002


# --- 6b. max_center_distance, the gate potato_center alone doesn't provide
# CONFIRMED 2026-09-14 against a real Isaac Sim cloud that this matters: with
# no distance gate, 4 of 11 "eyes" detected against a real merged cloud
# clustered near the ROBOT'S OWN BASE, tens of cm from the potato --
# curvature and shape index alone can't tell a real eye from anything else
# in the scene with a matching local shape, and potato_center only orients
# normals (describe_surface), it never restricted which points were even
# considered.

@pytest.mark.slow
def test_max_center_distance_rejects_a_lookalike_far_from_the_potato(potato, rng):
    points, _, truth = potato
    stray_center = np.array([0.06, 0.05, 0.16])  # nowhere near CENTER
    # Same point count as `potato` (not fewer): fibonacci_directions spreads
    # uniformly over the WHOLE sphere, so fewer points at the same radius
    # means a sparser cloud, a larger KNN neighbourhood, and curvature that
    # reads flatter (same non-scale-invariance find_eye_candidates' own
    # docstring already warns about) -- enough to make the lookalike fail
    # curvature_min on density alone, which would make this test pass for
    # the wrong reason.
    stray_directions = fibonacci_directions(24000)
    # The dimple has to sit on the side of the stray sphere FACING AWAY from
    # CENTER. describe_surface orients every normal, including the stray
    # cluster's, using CENTER (the only potato_center find_eye_candidates is
    # given) -- so a dimple on the NEAR side gets its true outward normal
    # flipped to agree with that wrong reference, turning a cup into a dome
    # (shape_index near 1.0, not 0.0) and silently disappearing regardless
    # of max_center_distance. That is a real, separate degradation this same
    # bug causes -- worth knowing about -- but it is not what this test is
    # for, so the dimple is placed where the flip happens not to matter.
    dimple_direction = stray_center - CENTER
    dimple_direction = dimple_direction / np.linalg.norm(dimple_direction)
    stray_radius, _ = press_dimples(
        modulated_radius(stray_directions), stray_directions, dimple_direction[None, :])
    stray_points = stray_center + stray_directions * stray_radius[:, None]
    stray_points = stray_points + rng.normal(scale=SCANNER_NOISE, size=stray_points.shape)
    combined = np.vstack([points, stray_points])

    ungated = find_eye_candidates(combined, CENTER, knn=KNN)
    matched_ungated, _ = match(ungated, truth)
    assert matched_ungated == 4, 'the real eyes should still all be found'
    assert len(ungated) > 4, (
        'the stray lookalike should be detected too -- otherwise this test is not '
        'actually exercising the failure max_center_distance fixes')

    gated = find_eye_candidates(combined, CENTER, knn=KNN,
                                max_center_distance=0.07)
    matched_gated, worst_gated = match(gated, truth)
    assert matched_gated == 4, 'the real eyes must not be filtered out'
    assert len(gated) == 4, 'the far-away lookalike must be filtered out'
    assert worst_gated < 0.002


# --- 7. colour, the axis geometry cannot supply --------------------------

@pytest.fixture(scope='module')
def potato_with_clods(rng):
    """Soil clods with the SAME pit geometry as an eye.

    Geometry cannot tell them apart -- that is the known ceiling colour
    exists to lift, so the fixture has to actually produce the false
    positives before the gate can be shown to remove them.
    """
    directions = fibonacci_directions(24000)
    eye_directions = fibonacci_directions(4)
    clod_directions = np.array([[0.0, 1.0, 0.0], [0.0, -0.7, 0.7]])
    clod_directions = clod_directions / np.linalg.norm(
        clod_directions, axis=1, keepdims=True)

    radius, is_eye = press_dimples(modulated_radius(directions),
                                   directions, eye_directions)
    radius, is_clod = press_dimples(radius, directions, clod_directions)
    points = CENTER + directions * radius[:, None]
    points = points + rng.normal(scale=SCANNER_NOISE, size=points.shape)

    skin = np.array([0.72, 0.58, 0.40])
    colors = np.clip(np.tile(skin, (len(directions), 1))
                     + rng.normal(scale=0.02, size=(len(directions), 3)), 0, 1)
    colors[is_eye] *= 0.45                              # an eye is a dark pit
    colors[is_clod] *= np.array([1.02, 0.95, 0.88])     # a clod is not darker
    colors = np.clip(colors, 0, 1)

    return dict(points=points, colors=colors, is_eye=is_eye, is_clod=is_clod,
                eye_truth=dimple_positions(eye_directions),
                clod_truth=dimple_positions(clod_directions))


def test_the_fixture_really_does_make_clods_look_like_skin(potato_with_clods):
    """Guards the guard. If a clod were darker than skin, the colour gate
    would be separating something the scan never confuses in the first place,
    and the two tests below would prove nothing.
    """
    colors = potato_with_clods['colors']
    luma = luminance(colors)
    skin = float(np.median(luma[~(potato_with_clods['is_eye']
                                  | potato_with_clods['is_clod'])]))
    eye = float(np.median(luma[potato_with_clods['is_eye']]))
    clod = float(np.median(luma[potato_with_clods['is_clod']]))

    assert eye < 0.6 * skin, 'an eye must be visibly darker than skin'
    assert abs(clod - skin) < 0.1 * skin, 'a clod must not be darker than skin'


def classifier(eye_truth, clod_truth, tolerance=0.010):
    """Nearest ground-truth feature, so nothing is counted twice."""
    all_truth = np.vstack([eye_truth, clod_truth])

    def classify(candidate):
        distances = np.linalg.norm(all_truth - candidate['position'], axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] > tolerance:
            return 'spurious'
        return 'eye' if nearest < len(eye_truth) else 'clod'

    return classify


@pytest.mark.slow
def test_an_eye_is_dark_and_a_clod_is_not(potato_with_clods):
    points, colors = potato_with_clods['points'], potato_with_clods['colors']
    found = find_eye_candidates(points, CENTER, knn=KNN, colors=colors)
    classify = classifier(potato_with_clods['eye_truth'],
                          potato_with_clods['clod_truth'])

    contrasts = {'eye': [], 'clod': []}
    for candidate in found:
        kind = classify(candidate)
        if kind in contrasts:
            contrasts[kind].append(candidate['color_contrast'])

    assert contrasts['clod'], 'the fixture must produce clod false positives'
    assert min(contrasts['eye']) > 10 * max(contrasts['clod'])


@pytest.mark.slow
def test_the_colour_gate_removes_every_clod_and_costs_no_eye(potato_with_clods):
    points, colors = potato_with_clods['points'], potato_with_clods['colors']
    classify = classifier(potato_with_clods['eye_truth'],
                          potato_with_clods['clod_truth'])

    def tally(found):
        counts = {'eye': 0, 'clod': 0, 'spurious': 0}
        for candidate in found:
            counts[classify(candidate)] += 1
        return counts

    ungated = tally(find_eye_candidates(points, CENTER, knn=KNN, colors=colors))
    gated = tally(find_eye_candidates(points, CENTER, knn=KNN, colors=colors,
                                      min_color_contrast=0.25))

    assert ungated['clod'] > 0
    assert gated['clod'] == 0
    assert gated['eye'] == ungated['eye']


@pytest.mark.slow
def test_colour_is_measured_but_inert_until_a_threshold_is_set(potato_with_clods):
    """Passing colours must not silently change the geometric result -- the
    gate is opt-in, and params.yaml ships it off."""
    points, colors = potato_with_clods['points'], potato_with_clods['colors']
    geometry_only = find_eye_candidates(points, CENTER, knn=KNN)
    measured = find_eye_candidates(points, CENTER, knn=KNN, colors=colors)
    assert len(measured) == len(geometry_only)


# --- 8. normal consistency: measured, and deliberately not a gate --------

@pytest.mark.slow
def test_every_candidate_reports_how_much_its_members_agree(potato):
    """The number exists so one real run can settle what a threshold should
    be. It is useless if it is not there on every candidate."""
    points, _, _ = potato
    for candidate in find_eye_candidates(points, CENTER, knn=KNN):
        assert 0.0 < candidate['normal_consistency'] <= 1.0


@pytest.mark.slow
def test_agreeing_members_score_near_one(potato):
    """On a well-sampled synthetic potato the members do agree, which is
    also why this number cannot separate good normals from bad ones here --
    see the find_eye_candidates docstring."""
    points, _, _ = potato
    found = find_eye_candidates(points, CENTER, knn=KNN)
    assert min(c['normal_consistency'] for c in found) > 0.9


@pytest.mark.slow
def test_the_gate_is_off_by_default_and_changes_nothing(potato):
    """min_normal_consistency=None must reproduce prior behaviour exactly,
    the same contract min_color_contrast and max_center_distance keep."""
    points, _, _ = potato
    ungated = find_eye_candidates(points, CENTER, knn=KNN)
    explicit = find_eye_candidates(points, CENTER, knn=KNN,
                                   min_normal_consistency=None)
    assert len(ungated) == len(explicit)


@pytest.mark.slow
def test_the_gate_rejects_when_it_is_actually_set(potato):
    """It has to work when a real run finally earns a threshold."""
    points, _, _ = potato
    found = find_eye_candidates(points, CENTER, knn=KNN)
    just_above = max(c['normal_consistency'] for c in found) + 1e-6
    assert found
    assert find_eye_candidates(points, CENTER, knn=KNN,
                               min_normal_consistency=just_above) == []


def test_members_that_cancel_produce_no_candidate_rather_than_a_nan():
    """A latent divide-by-zero: the candidate normal is the mean of its
    members' unit normals, and normalising that mean without checking its
    length turns a cluster whose normals oppose each other into NaN -- a
    pose the arm would then be commanded to.
    """
    opposed = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    mean = opposed.mean(axis=0)
    assert np.linalg.norm(mean) < 1e-9, 'the fixture must actually cancel'
    with np.errstate(invalid='ignore', divide='ignore'):
        assert np.isnan(mean / np.linalg.norm(mean)).all(), 'this is what was shipped'
