"""Live, shape-agnostic scan coverage.

Rather than tracking whether a preset set of candidate viewpoints has
been visited (which says nothing about whether that viewpoint actually
saw anything useful -- the previous approach), this bins the REAL
accumulated point cloud (/potato_scan/merged_cloud) into a spherical
(elevation, azimuth) grid around the potato center and tracks which
surface directions have actual data landing in them. NBV picks straight
from the remaining gaps in the real reconstruction, so behaviour adapts
automatically to whatever shape the current potato happens to have --
no per-shape tuning, no training data, works the same on the first
never-before-seen potato as the hundredth.

A gap that keeps coming back empty after its nominal viewpoint is
retried with independent recovery motions (RECOVERY_OFFSETS: closer /
farther radius, tilted approach angle) -- only after those are exhausted
is the cell flagged `unscannable`, so the scan still terminates instead
of looping forever on one persistently occluded spot.
"""
import numpy as np

# Per-retry (radius_scale, elevation_delta_deg, azimuth_delta_deg) offsets
# tried, in order, when a targeted gap cell stays empty after the nominal
# view. Cheapest/most-likely fix first (distance), then angle changes,
# since occlusion is more often fixed by distance/angle than by
# brute-force repetition of the same shot.
RECOVERY_OFFSETS = [
    (0.75, 0.0, 0.0),    # move closer -- helps thin/low-contrast surface detail
    (1.35, 0.0, 0.0),    # back off -- helps if too close (FOV clipping, focus limit)
    (1.0, 20.0, 0.0),    # tilt up -- helps occlusion from below (fixture, gripper)
    (1.0, -20.0, 0.0),   # tilt down
    (1.0, 0.0, 30.0),    # swing azimuth -- helps occlusion by the potato's own shape
    (1.0, 0.0, -30.0),
]


def direction_to_spherical(direction):
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    elevation_deg = np.degrees(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
    azimuth_deg = np.degrees(np.arctan2(direction[1], direction[0])) % 360.0
    return elevation_deg, azimuth_deg


def spherical_to_direction(elevation_deg, azimuth_deg):
    elevation = np.radians(elevation_deg)
    azimuth = np.radians(azimuth_deg)
    z = np.sin(elevation)
    radius_xy = np.cos(elevation)
    x = radius_xy * np.cos(azimuth)
    y = radius_xy * np.sin(azimuth)
    return np.array([x, y, z])


def perturb_direction(direction, elevation_delta_deg, azimuth_delta_deg,
                       min_elevation_deg, max_elevation_deg):
    """Nudge `direction` by the given elevation/azimuth offsets, clipped to
    the reachable elevation band -- the "independent motion" used to
    re-attempt a spot the nominal orbit failed to capture."""
    elevation_deg, azimuth_deg = direction_to_spherical(direction)
    elevation_deg = np.clip(elevation_deg + elevation_delta_deg, min_elevation_deg, max_elevation_deg)
    azimuth_deg = azimuth_deg + azimuth_delta_deg
    return spherical_to_direction(elevation_deg, azimuth_deg)


class SurfaceCoverageGrid:
    def __init__(self, min_expected_radius=0.015, max_expected_radius=0.07,
                 radius_band=0.02, elevation_bin_deg=8.0, azimuth_bin_deg=8.0,
                 min_elevation_deg=-15.0, max_elevation_deg=85.0,
                 min_hits_to_fill=3):
        """No per-potato size measurement needed: `min_expected_radius` /
        `max_expected_radius` are generic bounds wide enough to admit any
        real potato this fixture could hold (and reject the table/fixture/
        gripper, which sit well outside that band) -- set once at setup,
        not per potato. The ACTUAL radius of whichever potato is currently
        mounted is estimated live from the accumulated cloud on every
        update (median radius of plausible points), and `radius_band`
        controls how far a point may sit from that live estimate and
        still count as "on the surface" -- generous enough for a potato's
        natural lumpiness/elongation.
        """
        self.min_expected_radius = min_expected_radius
        self.max_expected_radius = max_expected_radius
        self.radius_band = radius_band
        self.elevation_bin_deg = elevation_bin_deg
        self.azimuth_bin_deg = azimuth_bin_deg
        self.min_elevation_deg = min_elevation_deg
        self.max_elevation_deg = max_elevation_deg
        self.min_hits_to_fill = min_hits_to_fill

        self.n_elev = max(1, int(round((max_elevation_deg - min_elevation_deg) / elevation_bin_deg)))
        self.n_az = max(1, int(round(360.0 / azimuth_bin_deg)))
        self.hits = np.zeros((self.n_elev, self.n_az), dtype=int)
        self.unscannable = np.zeros((self.n_elev, self.n_az), dtype=bool)
        self.estimated_radius = None  # set once enough points have come in

    def cell_direction(self, e, a):
        elevation_deg = self.min_elevation_deg + (e + 0.5) * self.elevation_bin_deg
        azimuth_deg = (a + 0.5) * self.azimuth_bin_deg
        return spherical_to_direction(elevation_deg, azimuth_deg)

    def set_from_points(self, points, potato_center):
        """Recompute hit counts (and the live radius estimate) from the
        current full merged cloud. Called each time a new
        /potato_scan/merged_cloud arrives -- idempotent, so it always
        reflects the actual up-to-date reconstruction rather than
        accumulating stale/duplicate counts."""
        self.hits[:] = 0
        if points is None or len(points) == 0:
            return

        rel = np.asarray(points, dtype=float) - np.asarray(potato_center, dtype=float)
        radii = np.linalg.norm(rel, axis=1)
        plausible = (radii >= self.min_expected_radius) & (radii <= self.max_expected_radius)
        if not np.any(plausible):
            return

        # median is robust to a modest fraction of stray/background points
        # slipping into the plausible band -- this IS the per-potato size,
        # discovered from data instead of measured by hand.
        self.estimated_radius = float(np.median(radii[plausible]))

        on_surface = plausible & (np.abs(radii - self.estimated_radius) <= self.radius_band)
        if not np.any(on_surface):
            return
        rel = rel[on_surface]
        radii = radii[on_surface]
        directions = rel / radii[:, None]

        elevations = np.degrees(np.arcsin(np.clip(directions[:, 2], -1.0, 1.0)))
        azimuths = np.degrees(np.arctan2(directions[:, 1], directions[:, 0])) % 360.0
        in_range = (elevations >= self.min_elevation_deg) & (elevations <= self.max_elevation_deg)
        if not np.any(in_range):
            return
        elevations = elevations[in_range]
        azimuths = azimuths[in_range]

        e_idx = np.clip(((elevations - self.min_elevation_deg) / self.elevation_bin_deg).astype(int),
                         0, self.n_elev - 1)
        a_idx = (azimuths / self.azimuth_bin_deg).astype(int) % self.n_az
        np.add.at(self.hits, (e_idx, a_idx), 1)

    def filled_mask(self):
        return self.hits >= self.min_hits_to_fill

    def is_filled(self, e, a):
        return bool(self.hits[e, a] >= self.min_hits_to_fill)

    def mark_unscannable(self, e, a):
        self.unscannable[e, a] = True

    def coverage_ratio(self):
        """Fraction of cells with real scanned data -- the honest scan
        completeness number (excludes cells given up on)."""
        return float(np.mean(self.filled_mask()))

    def resolved_ratio(self):
        """Fraction of cells either filled or given up on -- what actually
        gates loop termination, so one stuck spot can't loop forever."""
        return float(np.mean(self.filled_mask() | self.unscannable))

    def next_gap_direction(self, last_direction=None):
        """(e, a, direction) of the nearest not-yet-filled, not-yet-given-
        up cell to last_direction (minimizes robot travel), or
        (None, None, None) if none remain."""
        open_mask = ~self.filled_mask() & ~self.unscannable
        cells = np.argwhere(open_mask)
        if len(cells) == 0:
            return None, None, None
        dirs = np.array([self.cell_direction(e, a) for e, a in cells])

        if last_direction is None:
            e, a = cells[0]
            return int(e), int(a), dirs[0]

        last_direction = np.asarray(last_direction, dtype=float)
        last_direction = last_direction / np.linalg.norm(last_direction)
        sims = dirs @ last_direction
        idx = int(np.argmax(sims))
        e, a = cells[idx]
        return int(e), int(a), dirs[idx]

    def recovery_attempts(self, direction, max_attempts):
        for radius_scale, elev_delta, az_delta in RECOVERY_OFFSETS[:max_attempts]:
            perturbed = perturb_direction(
                direction, elev_delta, az_delta,
                self.min_elevation_deg, self.max_elevation_deg)
            yield perturbed, radius_scale

    def all_cells_with_status(self):
        """For RViz: every cell's representative direction + (filled,
        unscannable) status."""
        dirs, filled, unscannable = [], [], []
        filled_mask = self.filled_mask()
        for e in range(self.n_elev):
            for a in range(self.n_az):
                dirs.append(self.cell_direction(e, a))
                filled.append(bool(filled_mask[e, a]))
                unscannable.append(bool(self.unscannable[e, a]))
        return np.array(dirs), np.array(filled), np.array(unscannable)
