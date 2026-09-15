"""The Isaac Sim potato's geometry, with no Isaac Sim in it.

isaac_sim_common.make_potato_mesh builds the procedural potato the whole
simulated pipeline scans, detects on and drills -- and until now its
vertices, its eye pits and its random bumps existed only inside a running
Kit process, behind `pxr` and `carb` imports that nothing else can satisfy.
That made one question unanswerable anywhere but in a live Isaac session:
when eye_detector reports an eye where make_potato_mesh carved none, what
IS there?

The 2026-09-15 run posed exactly that. At 67% coverage, 8 spurious
detections appeared, all high-confidence by every number the detector
carries, clustered on one side of the potato. The README's hypothesis was
`bump_dirs`: two of the mesh's own random bumps meeting in a concave valley
that the raised mesh resolution had made visible for the first time. It
could not be checked, because `bump_dirs` was a local variable in a
function that only runs under Kit, on a potato re-seeded at random every
launch.

This module is the same geometry -- the identical random draws in the
identical order, so a seed here reproduces the potato Kit built -- as plain
numpy: the vertex grid, the quad faces, the eye pits, and the bumps, kept.
`make_potato_mesh` now builds its USD mesh from what this returns, so the
two cannot drift. `sample_surface` then stands in for the depth camera
(uniform points on the flat faces, voxel-thinned to the accumulator's grid)
so the real `find_eye_candidates` can be run against the real simulated
potato on a laptop, over as many seeds as it takes to make a claim.

What this is not: a sensor model. No noise, no grazing dropout, no
occlusion, every face seen. A detection that appears here is in the
geometry itself; one that appears only in Isaac is in the camera.
"""
from dataclasses import dataclass

import numpy as np

# The eye pits reuse the exact depth/width this project's own synthetic
# test suite already validates detection against (test_surface_curvature.py's
# EYE_DEPTH/EYE_SIGMA, at the same 35mm base_radius) rather than inventing
# new numbers -- see the 2026-09-15 field note in the README: the original
# `dot > 0.85` cap produced a ~39mm-diameter pit (a 31.8-degree cone at this
# radius), an order of magnitude wider than the 2-15mm min/max_eye_diameter
# the rest of the pipeline (eye_detector, force_drill_tuner, the depth-collar
# sizing for a 3.25mm bit) is built around -- curvature over a `knn`-point
# neighbourhood reads nearly flat on a bowl that wide, which is why every one
# of 7 ground-truth eyes measured kappa 2-50x below curvature_min even at 82%
# scan coverage.
EYE_DEPTH_M = 0.0035
EYE_SIGMA_RAD = 0.09


@dataclass
class PotatoGeometry:
    """Everything make_potato_mesh draws, kept.

    `points`/`faces` are the vertex grid and quads the USD mesh is built
    from. `eye_points`/`eye_normals` are the ground truth isaac_scene
    publishes. `bump_dirs`/`bump_amp`/`bump_width` are the bumps that used
    to be discarded, and `seed` is the one that reproduces all of it.
    """
    seed: int
    center: np.ndarray
    base_radius: float
    points: np.ndarray        # (n_lat * n_lon, 3) world-space vertices
    faces: np.ndarray         # (F, 4) vertex indices, quads
    eye_dirs: np.ndarray      # (E, 3) unit
    eye_points: np.ndarray    # (E, 3) world-space pit bottoms
    eye_normals: np.ndarray   # (E, 3) outward pit axis, == eye_dirs
    eye_depth: float
    eye_sigma: float
    bump_dirs: np.ndarray     # (B, 3) unit
    bump_amp: np.ndarray      # (B,) metres
    bump_width: np.ndarray    # (B,) the `w` in cos(theta) ** (1 / w)

    def radius_toward(self, directions, with_eyes=True):
        """The analytic surface radius along unit directions -- the same
        formula the vertex loop evaluates, vectorised, so a point on the
        surface can be queried without going through the mesh."""
        d = np.asarray(directions, dtype=float).reshape(-1, 3)
        r = np.full(len(d), self.base_radius)
        for bd, amp, w in zip(self.bump_dirs, self.bump_amp, self.bump_width):
            r += amp * np.maximum(0.0, d @ bd) ** (1.0 / w)
        if with_eyes:
            for ed in self.eye_dirs:
                angle = np.arccos(np.clip(d @ ed, -1.0, 1.0))
                r -= self.eye_depth * np.exp(-(angle ** 2) / (2.0 * self.eye_sigma ** 2))
        return r

    def surface_point_toward(self, directions):
        """World-space point where the surface crosses each unit direction."""
        d = np.asarray(directions, dtype=float).reshape(-1, 3)
        return self.center + d * self.radius_toward(d)[:, None]

    def bump_edge_offset_deg(self, positions):
        """For each world position, how far (degrees) its direction from
        the centre sits from the NEAREST bump's edge -- the ring 90 degrees
        from `bump_dir` where `max(0, cos)` switches on.

        Returns (offset_deg, bump_index). A candidate landing within a few
        degrees of 0 here is on a bump's rim, which is the geometry the
        spurious-detection hypothesis is about.
        """
        d = np.asarray(positions, dtype=float).reshape(-1, 3) - self.center
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        theta = np.degrees(np.arccos(np.clip(d @ self.bump_dirs.T, -1.0, 1.0)))  # (N, B)
        offset = np.abs(theta - 90.0)
        nearest = offset.argmin(axis=1)
        return offset[np.arange(len(d)), nearest], nearest


def generate(center, base_radius=0.035, bumpiness=0.35, n_lat=60, n_lon=90,
             seed=None, with_eyes=True, min_bump_width=0.4):
    """The procedural potato, exactly as make_potato_mesh has always drawn it.

    The random draws happen in the same order as the original in-Kit loop
    (bump count, dirs, amps, widths; eye count, dirs; per-potato depth and
    sigma jitter), so a seed gives the same potato here as in Isaac Sim.
    `seed=None` draws one and records it in the result rather than leaving
    the potato unreproducible -- the 2026-09-15 spurious detections came
    from a potato nobody can rebuild.

    `with_eyes=False` keeps every draw and zeroes only the pit depth: the
    same seed's bumps, with nothing else on them. Any eye the detector
    finds on that surface is by construction the bumps' doing.

    `min_bump_width` is the lower end of the `w` draw (default 0.4, the
    original). A bump's profile is cos(theta) ** (1/w), switched on at 90
    degrees by `max(0, ...)`; the exponent's slope at that edge is what
    decides whether the rim is a smooth blend (large exponent, small w) or a
    crease (exponent near 1, w near 1). Exposed so the sweep in
    sim_detection_check can move it; the default reproduces existing seeds.

    `n_lat`/`n_lon` default to 60x90 (5400 vertices, ~2mm spacing), not the
    original 24x36 (864 vertices, ~5mm spacing): with USD subdivision off,
    every face renders perfectly flat, so curvature only ever appears at a
    vertex where two faces meet -- a pit narrower than the vertex spacing
    has no vertex inside it to carve, and does not exist in the rendered
    geometry at all regardless of the depth formula. At ~2mm spacing a
    realistically-sized eye (EYE_SIGMA_RAD above) still spans several.
    """
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    seed = int(seed)
    center = np.asarray(center, dtype=float)
    rng = np.random.default_rng(seed)
    lats = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lons = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)

    n_bumps = int(rng.integers(4, 7))
    bump_dirs = rng.normal(size=(n_bumps, 3))
    bump_dirs /= np.linalg.norm(bump_dirs, axis=1, keepdims=True)
    bump_amp = rng.uniform(0.1, 1.0, size=n_bumps) * bumpiness * base_radius
    bump_width = rng.uniform(min_bump_width, 1.0, size=n_bumps)

    n_eyes = int(rng.integers(3, 8))
    eye_dirs = rng.normal(size=(n_eyes, 3))
    eye_dirs /= np.linalg.norm(eye_dirs, axis=1, keepdims=True)
    # Per-potato jitter around the validated values, not per-eye: keeps every
    # eye on one potato mutually consistent while still varying instance to
    # instance across re-seeds, same pattern the old eye_depth draw used.
    eye_depth = rng.uniform(0.7, 1.3) * EYE_DEPTH_M
    eye_sigma = rng.uniform(0.85, 1.15) * EYE_SIGMA_RAD
    if not with_eyes:
        eye_depth = 0.0

    geometry = PotatoGeometry(
        seed=seed, center=center, base_radius=float(base_radius),
        points=None, faces=None,
        eye_dirs=eye_dirs, eye_points=None, eye_normals=None,
        eye_depth=float(eye_depth), eye_sigma=float(eye_sigma),
        bump_dirs=bump_dirs, bump_amp=bump_amp, bump_width=bump_width)

    # The ground truth is where the pit actually is on the surface the
    # camera sees, bumps included -- not `base_radius - eye_depth` along
    # eye_dir, which is what this returned until 2026-09-15 with the note
    # "ignoring the smaller bump contribution ... close enough for RL
    # training targets". It was not small: bumps add up to
    # bumpiness * base_radius (12mm) of radius, and over 200 seeds the
    # published truth sat a median 6.2mm INSIDE the surface, 37% of eyes
    # further than the 8mm matching tolerance from their own pit. Every one
    # of those scored as one missed eye plus one spurious detection, however
    # well the detector did.
    #
    # The normal stays the pit's AXIS, eye_dir, and that is measured, not
    # assumed: the pit is carved radially, so its walls are symmetric about
    # eye_dir and the detector's normal (the mean of its members' normals)
    # tracks that axis -- 6.0deg mean / 11.7deg worst on noiseless clouds,
    # against 9.2 / 29.2 for the surface normal of the bump flank the pit
    # sits on. The axis is also the direction an insertion has to follow
    # to reach the pit's bottom, which is what the number is for.
    geometry.eye_points = geometry.surface_point_toward(eye_dirs)
    geometry.eye_normals = eye_dirs.copy()

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    directions = np.stack([np.cos(lat_grid) * np.cos(lon_grid),
                           np.cos(lat_grid) * np.sin(lon_grid),
                           np.sin(lat_grid)], axis=-1).reshape(-1, 3)
    geometry.points = center + directions * geometry.radius_toward(directions)[:, None]

    i = np.arange(n_lat - 1)[:, None]
    j = np.arange(n_lon)[None, :]
    j2 = (j + 1) % n_lon
    geometry.faces = np.stack([i * n_lon + j, i * n_lon + j2,
                               (i + 1) * n_lon + j2, (i + 1) * n_lon + j],
                              axis=-1).reshape(-1, 4)
    return geometry


def sample_surface(geometry, voxel_size=0.001, oversample=4.0, noise=0.0, rng=None):
    """Points on the mesh's flat faces, thinned to `voxel_size`, standing in
    for the accumulated scan.

    Uniform by area over the triangulated quads, at `oversample` times the
    voxel density, then averaged per voxel -- which is what
    pointcloud_accumulator's voxel_down_sample does to the camera's points.
    Sampling the FACES rather than returning the vertices matters: with
    subdivision off the camera sees flat quads with all the curvature at
    their shared edges, and that is the surface the detector actually
    measures kappa on.

    `noise` is isotropic Gaussian jitter (metres, 1 sigma) added after the
    voxel step, the one part of the camera this does model, because it
    turned out to decide everything: kappa is a covariance ratio, so a
    noise floor raises it across the whole surface, and the detector's
    curvature_min sits close enough to a nominal eye's noiseless kappa
    that the noise level alone moves recall from ~15% to ~55% and, past
    ~0.22mm, tips every point on the potato into a candidate. 0.15mm is
    what test_surface_curvature.py uses for its scanner.
    """
    rng = np.random.default_rng(rng)
    p = geometry.points
    quads = geometry.faces
    triangles = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    a, b, c = p[triangles[:, 0]], p[triangles[:, 1]], p[triangles[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total_area = float(area.sum())
    if total_area <= 0.0:
        return np.empty((0, 3))

    n = int(oversample * total_area / voxel_size ** 2)
    which = rng.choice(len(triangles), size=n, p=area / total_area)
    u, v = rng.uniform(size=n), rng.uniform(size=n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    samples = a[which] + u[:, None] * (b[which] - a[which]) + v[:, None] * (c[which] - a[which])

    cell = np.floor(samples / voxel_size).astype(np.int64)
    _, inverse = np.unique(cell, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    counts = np.bincount(inverse)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, samples)
    thinned = sums / counts[:, None]
    if noise > 0.0:
        thinned = thinned + rng.normal(scale=noise, size=thinned.shape)
    return thinned


def visible_from(geometry, points, camera_position, fov_deg=60.0, min_range=0.07,
                 max_range=0.50, max_grazing_deg=70.0, occlusion_samples=12):
    """Mask over `points`: which of them one view from `camera_position`
    captures. The camera looks at the potato's centre.

    The same four tests potato_surface.PotatoSurface.visible applies, for
    the same reasons its docstring gives (range, field of view, grazing,
    self-occlusion), on points sampled from THIS geometry instead of on
    its own direction set -- and with the occlusion test against the
    analytic radius rather than a nearest-sample lookup, since this
    surface has one. Still no noise, dropout or specularity: what this
    adds to sample_surface is only which patches a given orbit reaches,
    which is what a partial scan's recall depends on.
    """
    points = np.asarray(points, dtype=float)
    camera_position = np.asarray(camera_position, dtype=float)
    to_camera = camera_position - points
    distance = np.linalg.norm(to_camera, axis=1)
    view_direction = to_camera / np.maximum(distance, 1e-12)[:, None]

    in_range = (distance > min_range) & (distance < max_range)

    optical_axis = geometry.center - camera_position
    optical_axis = optical_axis / np.linalg.norm(optical_axis)
    off_axis = np.arccos(np.clip((-view_direction) @ optical_axis, -1.0, 1.0))
    in_fov = off_axis < np.radians(fov_deg) / 2.0

    # grazing against the radial direction, which for a star-shaped
    # surface stands in for the normal -- same stand-in PotatoSurface uses
    radial = points - geometry.center
    radial = radial / np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-12)
    grazing = np.arccos(np.clip(np.einsum('ij,ij->i', view_direction, radial), -1.0, 1.0))
    facing = grazing < np.radians(max_grazing_deg)

    candidate = in_range & in_fov & facing
    if not np.any(candidate):
        return candidate

    starts = points[candidate] + view_direction[candidate] * 1e-3
    steps = np.linspace(0.0, 1.0, occlusion_samples)[None, :, None]
    samples = starts[:, None, :] + steps * (camera_position - starts)[:, None, :]

    offsets = samples - geometry.center
    sample_distance = np.linalg.norm(offsets, axis=2)
    sample_directions = offsets / np.maximum(sample_distance, 1e-12)[..., None]
    surface_here = geometry.radius_toward(sample_directions.reshape(-1, 3)).reshape(sample_distance.shape)

    blocked = np.any(sample_distance < surface_here - 1e-4, axis=1)
    result = candidate.copy()
    result[np.flatnonzero(candidate)[blocked]] = False
    return result
