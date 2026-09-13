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
    _, _, s = described
    body, eye = np.median(s[~is_eye]), np.median(s[is_eye])
    assert body > 0.6
    assert eye < 0.4
    assert body - eye > 0.4, 'separation on the 0-1 scale'


@pytest.mark.slow
def test_the_shape_index_buys_precision_curvature_alone_cannot(potato, described):
    """The reason the filter has two axes rather than one."""
    _, is_eye, _ = potato
    _, kappa, s = described
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
def test_both_axes_find_every_eye_with_no_false_positives(potato):
    points, _, truth = potato
    found = find_eye_candidates(points, CENTER, knn=KNN,
                                curvature_min=0.015, shape_index_max=0.35)
    matched, worst = match(found, truth)
    assert matched == 4
    assert len(found) == 4
    assert worst < 0.002, 'sub-2 mm localisation is the point of the shape index'


@pytest.mark.slow
def test_each_axis_is_load_bearing_in_a_different_way(potato):
    """kappa rejects, the shape index localises. Dropping either costs
    something, and they are not the same something."""
    points, _, truth = potato
    both = find_eye_candidates(points, CENTER, knn=KNN,
                               curvature_min=0.015, shape_index_max=0.35)
    matched, worst = match(both, truth)

    kappa_only = find_eye_candidates(points, CENTER, knn=KNN,
                                     curvature_min=0.015, shape_index_max=1.01)
    _, worst_kappa_only = match(kappa_only, truth)
    assert worst_kappa_only > worst, 'dropping the shape index costs localisation'

    shape_only = find_eye_candidates(points, CENTER, knn=KNN,
                                     curvature_min=0.0, shape_index_max=0.35)
    matched_shape, _ = match(shape_only, truth)
    assert (len(shape_only) - matched_shape) > (len(both) - matched), \
        'dropping curvature costs rejection'


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
