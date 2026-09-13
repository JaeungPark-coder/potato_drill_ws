"""How many raster views does the scan actually need?

Sweeps the raster's two step sizes against the coverage they achieve on a set
of procedurally generated potatoes, using the same SurfaceCoverageGrid the
robot uses and the sampled visibility model in potato_surface.py. No robot,
no simulator, no GPU.

WHY ASK

The scan's cost is dominated by its view count -- each view is a move plus a
settle -- and the shipped default is 40 (45 degrees of azimuth by 25 of
elevation). Nothing chose that number against a measurement; it came from
picking round step sizes. Cycle time is the metric this cell had no way to
report at all until recently, and this is the one part of it that can be
sized before any hardware exists.

WHAT IT FOUND, AND THE CAVEAT THAT MATTERS

Twelve views (90 x 50 degrees) reached 100% coverage on every potato tried,
against 40 for the default -- a 3.3x reduction in the dominant cost. Nine
views reached 99.7%; six fell to 85.8%.

The caveat is the whole point of reading this carefully. The model is
GEOMETRIC. It has no sensor noise, no specularity, no dropout in the dark
pockets an eye actually is, and no arm -- so it cannot tell you that a view
was unreachable, or that a surface came back empty because it was shiny and
in shadow. Every one of those argues for more views than the geometric
minimum. So the honest reading is not "set it to 12": it is that 12 is what
the geometry requires, the other 28 are buying robustness against effects
this model does not contain, and that margin should be a decision someone
makes rather than an accident of round numbers.

    python -m potato_scan.scan_budget --potatoes 8
"""
import argparse

import numpy as np

from potato_scan.potato_surface import PotatoSurface
from potato_scan.scan_schedule import RasterOrbitSchedule
from potato_scan.surface_coverage import SurfaceCoverageGrid

# Coarse to fine. Below about nine views coverage falls away quickly, which
# is also the regime where a gap-filling policy would have something to do --
# see rl/cpu_scan_env.py.
STEP_GRID = ((180.0, 50.0), (120.0, 50.0), (90.0, 50.0), (90.0, 33.0),
             (60.0, 33.0), (60.0, 25.0), (45.0, 25.0), (36.0, 20.0))


def coverage_for(surface, potato_center, scan_radius, azimuth_step_deg,
                 elevation_step_deg, camera=None):
    """Grid coverage a raster of these step sizes reaches on one potato."""
    camera = camera or {}
    grid = SurfaceCoverageGrid()
    schedule = RasterOrbitSchedule(
        min_elevation_deg=grid.min_elevation_deg,
        max_elevation_deg=grid.max_elevation_deg,
        azimuth_step_deg=azimuth_step_deg,
        elevation_step_deg=elevation_step_deg)
    n_views = len(schedule)

    seen = np.zeros(len(surface.points), dtype=bool)
    while not schedule.done:
        _, _, direction = schedule.next_view()
        seen |= surface.visible(potato_center + direction * scan_radius, **camera)
    grid.set_from_points(surface.points[seen], potato_center)
    return n_views, grid.coverage_ratio()


def sweep(n_potatoes=5, potato_center=(0.50, 0.0, 0.15), scan_radius=0.15,
          n_surface_points=12000, seed=0, camera=None):
    """[(views, azimuth_step, elevation_step, mean, worst)] over the grid."""
    center = np.asarray(potato_center, dtype=float)
    surfaces = [PotatoSurface(center, np.random.default_rng(seed + i),
                              n_points=n_surface_points)
                for i in range(n_potatoes)]

    rows = []
    for azimuth_step, elevation_step in STEP_GRID:
        coverages, views = [], 0
        for surface in surfaces:
            views, coverage = coverage_for(surface, center, scan_radius,
                                           azimuth_step, elevation_step, camera)
            coverages.append(coverage)
        coverages = np.asarray(coverages)
        rows.append((views, azimuth_step, elevation_step,
                     float(coverages.mean()), float(coverages.min())))
    return rows


def smallest_sufficient(rows, required=1.0):
    """Fewest views whose WORST potato still clears `required`.

    Worst rather than mean on purpose: a view budget that covers the average
    potato and fails the awkward one is a budget that fails in production,
    and the awkward one is not rare.
    """
    passing = [r for r in rows if r[4] >= required]
    return min(passing, key=lambda r: r[0]) if passing else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--potatoes', type=int, default=5)
    parser.add_argument('--points', type=int, default=12000)
    parser.add_argument('--scan-radius', type=float, default=0.15)
    parser.add_argument('--required', type=float, default=1.0)
    args = parser.parse_args()

    rows = sweep(n_potatoes=args.potatoes, scan_radius=args.scan_radius,
                 n_surface_points=args.points)

    print(f'{args.potatoes} potatoes, scan_radius {args.scan_radius} m\n')
    print(f"{'views':>6} {'azimuth':>8} {'elevation':>10} {'mean':>8} {'worst':>8}")
    print('-' * 44)
    for views, azimuth, elevation, mean, worst in rows:
        print(f'{views:>6} {azimuth:>7.0f}d {elevation:>9.0f}d '
              f'{mean * 100:>7.1f}% {worst * 100:>7.1f}%')

    best = smallest_sufficient(rows, args.required)
    print()
    if best is None:
        print(f'  nothing on this grid reached {args.required * 100:.0f}% on every potato')
    else:
        views, azimuth, elevation, _, worst = best
        print(f'  fewest views clearing {args.required * 100:.0f}% on EVERY potato: '
              f'{views} ({azimuth:.0f}d x {elevation:.0f}d)')
        print(f'  the shipped default is 40 (45d x 25d), so the geometry asks for '
              f'{40 / views:.1f}x fewer')
    print('\n  Geometric only: no sensor noise, no specularity, no dropout in dark')
    print('  eye pockets, no arm to refuse a pose. All of those argue for more')
    print('  views than this, so treat the number as the floor and the rest as a')
    print('  robustness margin to choose deliberately.')


if __name__ == '__main__':
    main()
