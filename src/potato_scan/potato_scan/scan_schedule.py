"""Phase A of the scan: a deterministic raster orbit around the potato.

The scan runs in two phases, because the two halves of the problem have
very different character:

  Phase A (this module) -- the predictable bulk. Park at a fixed radius,
    face the potato, sweep the camera up and down one vertical column,
    rotate a fixed azimuth step around the potato, sweep the next column,
    and so on until the orbit closes. Every potato gets the same sweep;
    nothing here depends on what the current potato looks like, so a
    fixed schedule is exactly right and a learned policy would have
    nothing to learn.

  Phase B (surface_coverage.SurfaceCoverageGrid + scan_controller) --
    whatever the sweep MISSED. Which cells those are, and what motion
    recovers them, genuinely differs per potato: a deep eye pocket, a
    lumpy waist that shadows itself, the pin the potato is mounted on.
    That is the part worth learning (scan_controller's `view_policy`).

Columns alternate direction (up, then down, then up...) so the camera
ends each column where the next one starts, instead of flying back to
the bottom every time.
"""
import numpy as np

from potato_scan.surface_coverage import spherical_to_direction


class RasterOrbitSchedule:
    """Fixed (elevation, azimuth) view sequence -- the phase A sweep.

    min/max_elevation_deg bound the vertical sweep. The lower bound is
    what keeps the camera off the pin/fixture the potato is mounted on;
    the upper bound stops just short of straight overhead, where a
    look-at pose degenerates (the camera's up-vector becomes ambiguous --
    see pose_utils.look_at_rotation).
    """

    def __init__(self, min_elevation_deg=-15.0, max_elevation_deg=85.0,
                 azimuth_step_deg=45.0, elevation_step_deg=25.0):
        if azimuth_step_deg <= 0 or elevation_step_deg <= 0:
            raise ValueError('azimuth_step_deg and elevation_step_deg must be positive')
        if max_elevation_deg <= min_elevation_deg:
            raise ValueError('max_elevation_deg must exceed min_elevation_deg')

        self.min_elevation_deg = float(min_elevation_deg)
        self.max_elevation_deg = float(max_elevation_deg)
        self.azimuth_step_deg = float(azimuth_step_deg)
        self.elevation_step_deg = float(elevation_step_deg)

        self._views = self._build_views()
        self._index = 0

    def _build_views(self):
        # +1e-9 so an exactly-divisible range includes its top row rather
        # than dropping it to floating-point bad luck.
        elevations = np.arange(self.min_elevation_deg,
                               self.max_elevation_deg + 1e-9,
                               self.elevation_step_deg)
        azimuths = np.arange(0.0, 360.0, self.azimuth_step_deg)

        views = []
        for column, azimuth in enumerate(azimuths):
            # boustrophedon: sweep up, rotate, sweep back down
            sweep = elevations if column % 2 == 0 else elevations[::-1]
            for elevation in sweep:
                views.append((float(elevation), float(azimuth)))
        return views

    def __len__(self):
        return len(self._views)

    @property
    def done(self):
        return self._index >= len(self._views)

    @property
    def progress(self):
        return f'{self._index}/{len(self._views)}'

    def next_view(self):
        """(elevation_deg, azimuth_deg, direction) of the next view in the
        sweep, or (None, None, None) once the orbit has closed."""
        if self.done:
            return None, None, None
        elevation_deg, azimuth_deg = self._views[self._index]
        self._index += 1
        return elevation_deg, azimuth_deg, spherical_to_direction(elevation_deg, azimuth_deg)
