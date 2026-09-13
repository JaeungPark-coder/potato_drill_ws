"""A potato you can point a camera at, in numpy.

Enough of a sensor model to train a view policy against, with no simulator,
no GPU and no mesh library. The potato is star-shaped about its own centre --
its surface is a radius as a smooth function of direction -- which is true
enough of a real one and makes ray-surface intersection a sampling test along
a segment rather than a mesh traversal.

WHAT IT MODELS, AND WHY EACH PART EARNS ITS PLACE

A view either captures a patch of surface or it does not, and in this cell
there turn out to be four independent ways for it not to:

  range      A depth camera below its minimum does not degrade, it returns
             nothing. This is the failure camera_ranges.py screens for, and
             modelling it is what lets a policy learn that creeping closer
             stops paying at some point rather than improving forever.
  field of
  view       Off-axis surface is simply not imaged.
  grazing    Stereo depth falls apart where the surface turns away from the
             camera, well before it is geometrically hidden. A pure
             facing test (is the normal pointing at me) is too generous;
             the cutoff here is an angle, not a sign.
  occlusion  The potato's own lumps hide parts of it from some directions,
             which is the entire reason a scan needs more than one view and
             the thing a gap-filling policy exists to handle.

Left out on purpose: sensor noise, specularity, and the arm. Noise and
specularity change which points come back marginally; the arm decides which
poses are reachable at all, and a policy that learns to want unreachable
views is a problem for the reachability search, not for this model.
"""
import numpy as np
from scipy.spatial import cKDTree


def fibonacci_directions(n):
    """n roughly equally spaced unit vectors -- the sampling the surface,
    the eyes and the visibility tests all share."""
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([np.cos(azimuth) * np.sin(polar),
                     np.sin(azimuth) * np.sin(polar),
                     np.cos(polar)], axis=-1)


class PotatoSurface:
    """radius(direction) plus a few eyes, sampled on a fixed direction set.

    The same direction set is reused for every query, so "which points did
    this view capture" is a mask over a fixed array rather than a fresh
    intersection each time -- which is what keeps an episode cheap enough to
    train against on a laptop.
    """

    def __init__(self, center, rng, n_points=6000, base_radius=0.035,
                 lumpiness=0.18, n_eyes=None, eye_depth=0.0035, eye_sigma=0.09,
                 n_dents=3, dent_depth=0.010, dent_sigma=0.45):
        """`n_dents` is what makes the potato worth scanning more than once.

        A smooth ellipsoid is convex, and a convex surface has no
        self-occlusion at all: every patch is visible from some direction on
        the orbit, so a plain raster sweep covers it and there is no
        gap-filling problem left to solve. Measured, with dents off: the
        40-view raster reached 96.7% coverage and the episode ended
        immediately.

        Real potatoes are not convex. They have waists, flats and deep eye
        pockets that hide parts of themselves, and those are exactly the
        cells a sweep leaves empty. The dents are broad concave regions that
        put that back, so the policy is trained on the problem it is meant to
        handle rather than on a shape that never has one.
        """
        self.center = np.asarray(center, dtype=float)
        self.directions = fibonacci_directions(n_points)

        # a lumpy ellipsoid: low-order spherical harmonics are the cheap way
        # to get a smooth, closed, potato-ish surface with no mesh at all
        modulation = (1.0
                      + lumpiness * self.directions[:, 2] ** 2
                      - 0.5 * lumpiness * self.directions[:, 0] ** 2
                      + 0.3 * lumpiness * rng.normal() * self.directions[:, 1])
        self.radii = base_radius * modulation

        # broad concavities: this is what creates self-occlusion
        self.dent_directions = []
        for _ in range(int(n_dents)):
            dent = rng.normal(size=3)
            dent /= np.linalg.norm(dent)
            self.dent_directions.append(dent)
            angle = np.arccos(np.clip(self.directions @ dent, -1.0, 1.0))
            depth = dent_depth * rng.uniform(0.6, 1.4)
            self.radii -= depth * np.exp(-(angle ** 2) / (2.0 * dent_sigma ** 2))

        n_eyes = int(rng.integers(4, 9)) if n_eyes is None else n_eyes
        self.eye_directions = fibonacci_directions(max(n_eyes, 1))
        # rotate the eye pattern so successive potatoes do not share it
        spin = rng.normal(size=(3, 3))
        q, _ = np.linalg.qr(spin)
        self.eye_directions = self.eye_directions @ q
        self.is_eye = np.zeros(len(self.directions), dtype=bool)
        for eye in self.eye_directions:
            angle = np.arccos(np.clip(self.directions @ eye, -1.0, 1.0))
            self.radii -= eye_depth * np.exp(-(angle ** 2) / (2.0 * eye_sigma ** 2))
            self.is_eye |= angle < eye_sigma

        self.points = self.center + self.directions * self.radii[:, None]
        self.max_radius = float(self.radii.max())
        # Nearest direction by KD-tree rather than by an all-pairs dot
        # product. On unit vectors, nearest in Euclidean distance IS nearest
        # in angle, so this is exact -- and the occlusion test queries tens
        # of thousands of directions per view, where all-pairs made a single
        # view cost a third of a second and put training out of reach.
        self._direction_tree = cKDTree(self.directions)

    def radius_toward(self, directions):
        """Surface radius along arbitrary directions, by nearest sample.

        Nearest-neighbour rather than interpolation because it is only used
        by the occlusion test, where the question is "is this sample inside
        the potato" and the surface is smooth at the scale of the sampling.
        """
        _, index = self._direction_tree.query(np.asarray(directions), k=1)
        return self.radii[index]

    def visible(self, camera_position, fov_deg=60.0, min_range=0.07,
                max_range=0.50, max_grazing_deg=70.0, occlusion_samples=12):
        """Mask over self.points: which surface points this view captures."""
        camera_position = np.asarray(camera_position, dtype=float)
        to_camera = camera_position - self.points
        distance = np.linalg.norm(to_camera, axis=1)
        view_direction = to_camera / np.maximum(distance, 1e-12)[:, None]

        in_range = (distance > min_range) & (distance < max_range)

        # the camera looks at the potato's centre
        optical_axis = self.center - camera_position
        optical_axis = optical_axis / np.linalg.norm(optical_axis)
        off_axis = np.arccos(np.clip((-view_direction) @ optical_axis, -1.0, 1.0))
        in_fov = off_axis < np.radians(fov_deg) / 2.0

        # grazing measured against the radial direction, which for a
        # star-shaped surface stands in for the normal
        grazing = np.arccos(np.clip(
            np.einsum('ij,ij->i', view_direction, self.directions), -1.0, 1.0))
        facing = grazing < np.radians(max_grazing_deg)

        candidate = in_range & in_fov & facing
        if not np.any(candidate):
            return candidate

        # occlusion: march from just off the surface toward the camera and
        # ask whether any sample is inside the potato
        starts = self.points[candidate] + view_direction[candidate] * 1e-3
        steps = np.linspace(0.0, 1.0, occlusion_samples)[None, :, None]
        samples = starts[:, None, :] + steps * (camera_position - starts)[:, None, :]

        offsets = samples - self.center
        sample_distance = np.linalg.norm(offsets, axis=2)
        sample_directions = offsets / np.maximum(sample_distance, 1e-12)[..., None]
        flat = sample_directions.reshape(-1, 3)
        surface_here = self.radius_toward(flat).reshape(sample_distance.shape)

        blocked = np.any(sample_distance < surface_here - 1e-4, axis=1)
        result = candidate.copy()
        result[np.flatnonzero(candidate)[blocked]] = False
        return result
