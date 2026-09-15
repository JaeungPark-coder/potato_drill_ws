"""Run the eye detector against the Isaac Sim potato, with no Isaac Sim.

    # what the detector does on the simulated potato, over many of them
    python -m potato_scan.sim_detection_check --seeds 30

    # one particular potato -- the seed isaac_scene.py printed at startup --
    # with every candidate tagged against its true eyes and its bumps
    python -m potato_scan.sim_detection_check --seed 1234567

    # the same, at a different scanner noise level or threshold
    python -m potato_scan.sim_detection_check --seeds 30 --noise-mm 0.2 --mean-curvature-min 100

    # the kappa gate the detector shipped with until 2026-09-16
    python -m potato_scan.sim_detection_check --seeds 30 --mean-curvature-min -1 --curvature-min 0.015

    # a PARTIAL scan: only what the first N views of the shipped raster orbit
    # can see (field of view, range, grazing, self-occlusion), with the
    # coverage the robot's own grid would report for it
    python -m potato_scan.sim_detection_check --seeds 30 --views 5
    python -m potato_scan.sim_detection_check --seeds 30 --view-sweep

The potato is procedural_potato.generate -- the identical geometry
isaac_sim_common.make_potato_mesh wraps in USD -- sampled on its flat faces
at pointcloud_accumulator's voxel size, plus Gaussian scanner noise. The
detector is surface_curvature.find_eye_candidates with config/params.yaml's
defaults, and the scoring is detection_accuracy.match_detections, the same
code detection_accuracy_check runs live. With `--views` the cloud is cut
down to what the shipped raster orbit's first N views reach, by the same
range / field-of-view / grazing / self-occlusion tests scan_budget uses,
and scored alongside the coverage the robot's own SurfaceCoverageGrid
reports for that cloud -- so a live run's "found X at Y% coverage" has an
off-line counterpart. What is still missing is the sensor: no dropout in
the dark pocket an eye is, no specularity, no registration error between
views. A candidate that shows up here is in the geometry; one that shows
up only in Isaac is in the camera or the merge.

WHAT IT SETTLED, 2026-09-15 (30 potatoes, the kappa gate then in use)

Written to check the README's hypothesis for 8 spurious detections at 67%
coverage -- that two of the mesh's random bumps meet in a concave valley --
and it does not hold: with noise off, 30 potatoes produce 0 spurious
candidates, with the eye pits carved AND with them removed. The bumps on
their own make nothing the detector accepts. What does is the noise floor.
kappa is a covariance ratio, so isotropic noise lifts it everywhere; at
0.15mm the 90th percentile is ~0.007 and spurious stays at 0, at 0.20mm it
is ~0.012 and a dozen appear, at 0.25mm it is ~0.018 -- past curvature_min
-- and every point on the potato is a candidate. eye_detector logs those
percentiles on every run: compare the 67% run's against the 43% one's
before touching a threshold, because a merged cloud that thickens as views
accumulate is the same thing as rising noise to this detector.

The recall side has a cause too. A nominal eye (3.5mm deep, sigma 0.09rad)
on a noiseless 1mm cloud measures kappa 0.010-0.016 -- straddling
curvature_min=0.015 -- so which eyes clear it is decided by the per-potato
depth/sigma jitter and by the noise: recall ran 6% for the shallowest-
widest potatoes to 100% for the sharpest, 13% with no noise and 55% at
0.20mm. Four of seven eyes missed at 67% coverage is what this threshold
does at this density, not evidence the scan had not reached them.

And the ground truth was off the surface: bumps were ignored when placing
it, so it sat a median 6mm inside, 37% of eyes beyond the 8mm matching
tolerance -- each scored as one miss plus one spurious detection. Fixed in
procedural_potato.generate; this tool scores against the corrected truth.

WHAT CHANGED BECAUSE OF IT, 2026-09-16

The gate moved from kappa to mean curvature H (surface_curvature's
docstring has the reasoning). The comparison that decided it, same 30
potatoes, same clustering and size window, only the curvature axis swapped:

    noise    kappa > 0.015          H > 100/m        H > 150/m        H > 200/m
    0.00mm   11% found, 0 spur.     75%, 0           42%, 0           18%, 0
    0.15mm   32%, 0                 93%, 1           57%, 0           25%, 0
    0.20mm   59%, 17                90%, 10          66%, 0           30%, 0
    0.25mm   35%, 2232              73%, 309         75%, 0           41%, 0
    0.30mm    1%, 405               40%, 1729        85%, 15          49%, 0

(this tool, `--seeds 30`, each gate at each `--noise-mm`.) Mean position
error stayed 0.6-1.1mm for every H setting where kappa's grew to 3-7mm as
noise rose. 150/m is the default: the largest recall that holds 0
spurious through 0.25mm. Both gates remain available here and in
eye_detector, so this table can be re-run when a real camera's noise level
is known -- which the `surface thickness` line, printed per potato, reads
straight off a cloud.

PARTIAL SCANS (`--view-sweep`, 30 potatoes, 0.15mm noise, H > 150/m)

The live runs reported recall AT a coverage -- 2/7 at 43-49%, 3/7 at 67%
-- and the question was always whether the missing eyes were unscanned or
under the bar. This is the same raster orbit, cut at N views:

    views   coverage   H > 150/m           kappa > 0.015 (old gate)
    3       ~39%       14% found, 0 spur.  10%, 0
    5       ~66%       19%, 0              11%, 0
    10      ~73%       30%, 0              15%, 0
    20      ~86%       39%, 0              21%, 0
    40      ~100%      56%, 0              32%, 0

The coverage column lands where the live runs did (43-49% after 3 views,
82% after 10), so the rows are comparable: 2/7 at 43-49% and 3/7 at 67%
are what the old gate does at that coverage, and finishing the orbit
would have roughly doubled it, not found every eye. Recall tracks
coverage almost proportionally under either gate -- the missing eyes at a
partial coverage are mostly unscanned, and the ones missing at 100% are
under the bar. The ragged edge of a partial cloud is where a curvature
fit is least trustworthy, and the H gate produced no spurious candidates
there at any cut.
"""
import argparse

import numpy as np

from potato_scan.detection_accuracy import format_report, match_detections, summarize
from potato_scan.procedural_potato import generate, sample_surface, visible_from
from potato_scan.scan_schedule import RasterOrbitSchedule
from potato_scan.surface_coverage import SurfaceCoverageGrid
from potato_scan.surface_curvature import describe_surface, find_eye_candidates

# config/params.yaml, eye_detector block -- kept as the one place the offline
# check and the live node can disagree, so it is worth checking they do not
DEFAULTS = dict(knn=30, mean_curvature_min=150.0, curvature_min=None, shape_index_max=0.35,
                cluster_eps=0.003, cluster_min_points=8,
                min_diameter=0.002, max_diameter=0.015, max_center_distance=0.07)
POTATO_CENTER = np.array([0.50, 0.00, 0.15])
VOXEL_SIZE = 0.001
SCAN_RADIUS = 0.15   # scan_controller's scan_radius
VIEW_SWEEP = (3, 5, 10, 20, 40)


def partial_scan(geometry, points, n_views, scan_radius=SCAN_RADIUS):
    """What the shipped raster orbit (45 x 25 degrees, 40 views, the
    params.yaml defaults) sees in its first `n_views`, and the coverage the
    robot's grid would report for it. Returns (mask, coverage_ratio)."""
    schedule = RasterOrbitSchedule()
    seen = np.zeros(len(points), dtype=bool)
    for _ in range(min(int(n_views), len(schedule))):
        _, _, direction = schedule.next_view()
        seen |= visible_from(geometry, points, geometry.center + direction * scan_radius)
    grid = SurfaceCoverageGrid()
    grid.set_from_points(points[seen], geometry.center)
    return seen, grid.coverage_ratio()


def detect(geometry, noise, params, voxel_size=VOXEL_SIZE, sample_seed=0, views=None):
    points = sample_surface(geometry, voxel_size=voxel_size, noise=noise, rng=sample_seed)
    coverage = None
    if views is not None:
        seen, coverage = partial_scan(geometry, points, views)
        points = points[seen]
    if len(points) < params['knn'] + 1:
        empty = match_detections(np.empty((0, 3)), geometry.eye_points)
        return points, None, [], empty, coverage
    surface = describe_surface(points, geometry.center, knn=params['knn'])
    candidates = find_eye_candidates(points, geometry.center, **params)
    result = match_detections(
        [c['position'] for c in candidates], geometry.eye_points,
        detected_normals=[c['normal'] for c in candidates],
        truth_normals=geometry.eye_normals)
    return points, surface, candidates, result, coverage


def _gate_label(params):
    parts = []
    if params['mean_curvature_min'] is not None:
        parts.append(f"H>{params['mean_curvature_min']:g}/m")
    if params['curvature_min'] is not None:
        parts.append(f"kappa>{params['curvature_min']:g}")
    return ' & '.join(parts) if parts else 'shape index only'


def report_one(seed, noise, params, with_eyes=True, views=None):
    geometry = generate(POTATO_CENTER, seed=seed, with_eyes=with_eyes)
    points, surface, candidates, result, coverage = detect(geometry, noise, params, views=views)
    if surface is None:
        print(f'potato seed {geometry.seed}: {len(points)} points after {views} views -- '
              f'too few to describe')
        return result
    q = np.percentile(surface.kappa, [50, 90, 99])
    h = np.percentile(surface.mean_curvature, [50, 90, 99])
    thickness = np.percentile(surface.thickness, [50, 90])

    print(f'potato seed {geometry.seed}: {len(geometry.eye_dirs)} eyes '
          f'(depth {geometry.eye_depth * 1000:.2f}mm, sigma {np.degrees(geometry.eye_sigma):.1f}deg), '
          f'{len(geometry.bump_dirs)} bumps; {len(points)} points at {VOXEL_SIZE * 1000:.0f}mm '
          f'+ {noise * 1000:.2f}mm noise'
          + ('' if coverage is None else
             f' | first {views} raster views, coverage {coverage * 100:.0f}%'))
    for i, (bd, amp, w) in enumerate(zip(geometry.bump_dirs, geometry.bump_amp, geometry.bump_width)):
        print(f'  bump {i}: dir {np.round(bd, 3)} amp {amp * 1000:.1f}mm width {w:.2f}')
    print(f'  gate: {_gate_label(params)}')
    print(f'  H p50/p90/p99 (1/m): {h[0]:.0f} / {h[1]:.0f} / {h[2]:.0f}'
          + (' -- mean_curvature_min BELOW p90, expect spurious'
             if params['mean_curvature_min'] is not None and params['mean_curvature_min'] < h[1]
             else ''))
    print(f'  kappa p50/p90/p99: {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}'
          + (' -- curvature_min BELOW p90, expect spurious'
             if params['curvature_min'] is not None and params['curvature_min'] < q[1]
             else ''))
    print(f'  surface thickness (noise floor) p50/p90: '
          f'{thickness[0] * 1000:.3f} / {thickness[1] * 1000:.3f}mm')
    print()
    print(format_report(result))

    if candidates:
        print()
        print('  each candidate, against the geometry it sits on:')
        offsets, bump_index = geometry.bump_edge_offset_deg([c['position'] for c in candidates])
        matched = {d: t for t, d, _, _ in result['matches']}
        for d, c in enumerate(candidates):
            tag = f'eye {matched[d]}' if d in matched else 'SPURIOUS'
            print(f"    detection {d} [{tag:8s}] diam {c['diameter'] * 1000:4.1f}mm "
                  f"pts {c['points']:3d} H {c['mean_curvature']:4.0f}/m S {c['shape_index']:.2f} "
                  f"consistency {c['normal_consistency']:.3f} | "
                  f"{offsets[d]:4.1f}deg from bump {bump_index[d]}'s rim "
                  f"(width {geometry.bump_width[bump_index[d]]:.2f})")
    return result


def report_many(seeds, noise, params, with_eyes=True, views=None):
    found = truth = spurious = 0
    positions, normals, kappa90, coverages = [], [], [], []
    for seed in seeds:
        geometry = generate(POTATO_CENTER, seed=seed, with_eyes=with_eyes)
        _, surface, _, result, coverage = detect(geometry, noise, params, views=views)
        if coverage is not None:
            coverages.append(coverage)
        s = summarize(result)
        found += s['found']
        truth += s['of']
        spurious += s['spurious']
        positions += [m[2] for m in result['matches']]
        normals += [m[3] for m in result['matches'] if m[3] is not None]
        if surface is not None:
            kappa90.append(np.percentile(surface.kappa, 90))

    label = 'eyes carved' if with_eyes else 'eyes REMOVED (bumps only)'
    line = f'{len(seeds)} potatoes, {label}, {noise * 1000:.2f}mm noise, '
    if views is not None:
        line += f'first {views} views (coverage ~{np.mean(coverages) * 100:.0f}%), '
    line += (f'{_gate_label(params)}: kappa p90 ~{np.mean(kappa90):.4f} | '
             f'found {found}/{truth}')
    if truth:
        line += f' ({100.0 * found / truth:.0f}%)'
    line += f', spurious {spurious}'
    if positions:
        line += (f' | position mean {np.mean(positions) * 1000:.2f}mm '
                 f'worst {np.max(positions) * 1000:.2f}mm')
    if normals:
        line += f' | normal mean {np.mean(normals):.1f}deg worst {np.max(normals):.1f}deg'
    print(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--seed', type=int, default=None,
                        help='one potato, reported candidate by candidate (the seed '
                             'isaac_scene.py printed)')
    parser.add_argument('--seeds', type=int, default=30,
                        help='how many potatoes (seeds 0..N-1) for the summary line')
    parser.add_argument('--noise-mm', type=float, default=0.15,
                        help='scanner noise, 1 sigma; the test suite uses 0.15')
    parser.add_argument('--no-eyes', action='store_true',
                        help='carve no pits: anything found is the bumps')
    parser.add_argument('--views', type=int, default=None,
                        help='keep only what the first N views of the shipped raster orbit '
                             'see (default: the whole surface, no camera)')
    parser.add_argument('--view-sweep', action='store_true',
                        help=f'the summary line at each of {VIEW_SWEEP} views')
    parser.add_argument('--mean-curvature-min', type=float, default=DEFAULTS['mean_curvature_min'],
                        help='the H gate, 1/m; negative turns it off')
    parser.add_argument('--curvature-min', type=float, default=-1.0,
                        help='the old kappa gate; negative (default) turns it off')
    for name in ('shape_index_max', 'cluster_eps', 'max_diameter'):
        parser.add_argument('--' + name.replace('_', '-'), type=float, default=DEFAULTS[name])
    parser.add_argument('--knn', type=int, default=DEFAULTS['knn'])
    parser.add_argument('--cluster-min-points', type=int, default=DEFAULTS['cluster_min_points'])
    args = parser.parse_args(argv)

    params = dict(DEFAULTS)
    for name in ('knn', 'shape_index_max', 'cluster_eps', 'cluster_min_points', 'max_diameter'):
        params[name] = getattr(args, name)
    params['mean_curvature_min'] = None if args.mean_curvature_min < 0 else args.mean_curvature_min
    params['curvature_min'] = None if args.curvature_min < 0 else args.curvature_min
    noise = args.noise_mm / 1000.0

    if args.view_sweep:
        for views in VIEW_SWEEP:
            report_many(range(args.seeds), noise, params, with_eyes=not args.no_eyes, views=views)
    elif args.seed is not None:
        report_one(args.seed, noise, params, with_eyes=not args.no_eyes, views=args.views)
    else:
        report_many(range(args.seeds), noise, params, with_eyes=not args.no_eyes, views=args.views)


if __name__ == '__main__':
    main()
