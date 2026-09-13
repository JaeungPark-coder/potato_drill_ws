"""Checks the raster orbit is the motion described: face the potato,
sweep up/down a column, rotate a fixed step, repeat."""
import numpy as np
import pytest

from potato_scan.scan_schedule import RasterOrbitSchedule
from potato_scan.surface_coverage import direction_to_spherical

AZIMUTHS = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
ELEVATIONS = [-15.0, 10.0, 35.0, 60.0, 85.0]
ROWS = len(ELEVATIONS)


@pytest.fixture(scope='module')
def views():
    """Every (elevation, azimuth, direction) the shipped raster produces."""
    schedule = RasterOrbitSchedule()
    out = []
    while not schedule.done:
        out.append(schedule.next_view())
    return out


def test_shipped_raster_is_eight_columns_of_five(views):
    assert len(views) == 40
    assert sorted({az for _, az, _ in views}) == AZIMUTHS
    assert sorted({el for el, _, _ in views}) == ELEVATIONS


def test_columns_are_contiguous_and_alternate_direction(views):
    """Boustrophedon: one azimuth per column, sweeping up then down."""
    for c in range(len(AZIMUTHS)):
        column = views[c * ROWS:(c + 1) * ROWS]
        assert len({az for _, az, _ in column}) == 1, 'a column must hold one azimuth'
        elevations = [el for el, _, _ in column]
        expected = ELEVATIONS if c % 2 == 0 else ELEVATIONS[::-1]
        assert elevations == expected, (c, elevations)


def test_no_flyback_at_column_seams(views):
    """The end of one column is adjacent in elevation to the start of the next.

    A non-zero jump here means the arm travels the full height of the sweep
    between every column, which is the cost the boustrophedon exists to avoid.
    """
    seams = [abs(views[c * ROWS + ROWS - 1][0] - views[(c + 1) * ROWS][0])
             for c in range(len(AZIMUTHS) - 1)]
    assert all(seam == 0.0 for seam in seams), seams


def test_directions_are_unit_vectors_that_round_trip(views):
    for el, az, direction in views:
        assert abs(np.linalg.norm(direction) - 1.0) < 1e-12
        el_back, az_back = direction_to_spherical(direction)
        az_error = abs((az_back - az + 180.0) % 360.0 - 180.0)   # circular distance
        assert abs(el_back - el) < 1e-9, (el, el_back)
        assert az_error < 1e-9, (az, az_back)


def test_consecutive_views_are_small_moves(views):
    """No wild jumps across the sphere -- every step is one grid cell."""
    angles = [np.degrees(np.arccos(np.clip(
        np.dot(views[i][2], views[i + 1][2]), -1.0, 1.0)))
        for i in range(len(views) - 1)]
    assert max(angles) <= 46.0, f'largest step {max(angles):.1f} deg'


def test_step_sizes_are_configurable():
    assert len(RasterOrbitSchedule(azimuth_step_deg=30.0,
                                   elevation_step_deg=20.0)) == 12 * 6
