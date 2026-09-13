"""Train the gap-filling view policy on a laptop.

No Isaac Sim, no GPU, no mesh library: a star-shaped potato and a sampled
visibility test (potato_surface.py) drive the same SurfaceCoverageGrid the
robot uses, through the same observation and action encoding the deployed
policy uses (scan_policy_spec). What changes between here and the robot is
only where the points come from.

IT TRAINS PHASE B ONLY, AND THAT IS THE POINT

scan_controller scans in two phases. Phase A is a fixed raster orbit that
every potato gets -- identical regardless of shape, so there is nothing in it
to learn. Phase B fills whatever that sweep missed, and what it missed
depends entirely on this potato's lumps, eye pockets and mounting pin.

So reset() runs the raster itself and hands the policy a grid that is already
mostly filled. Two things follow. The policy never wastes capacity
rediscovering a sweep that is already written down. And its observation --
the pooled coverage map -- arrives meaning something specific: these are the
places the standard sweep could not reach. Training from an empty grid would
instead ask it to learn the sweep first and the interesting part second.

READ THIS BEFORE TRAINING ANYTHING

At the shipped raster this environment has no problem to pose. Measured with
scan_budget.py: a 40-view raster reaches 100% grid coverage on every potato
tried, and so does 12 views. Phase B starts with nothing left to fill, the
episode ends on the first step, and a policy trained here would be learning
from a task that is already solved.

That is a property of the surface model, and worth being precise about. The
potato here is star-shaped -- its radius is a single value per direction --
and a star-shaped body seen from outside has essentially no self-occlusion
once grazing views are discarded. Deepening the dents does not change that,
because a dent is still single-valued; genuine self-occlusion needs an
overhang, which this representation cannot express. Adding the mounting pin
as an occluder was tried too and left one cell of 540 empty at a 25 mm
radius.

So the environment is only meaningful where the raster is coarse enough to
leave real gaps, which the same sweep locates: below about nine views,
coverage falls away (6 views reach 85.8%). Train here with a deliberately
coarse raster to study that regime; do not train against the shipped 40 and
expect to have learned anything.

The broader reading is about the pipeline, not the model: the value of the
Phase B policy is unproven, and where it will actually earn its place is on
effects this model omits -- dropout in the dark pockets an eye is,
specularity, and views the arm refuses -- not on geometry.

WHAT THIS MODEL IS NOT

Visibility here is geometric. It has no sensor noise, no specularity, and no
arm: a view the policy asks for is always executed, where on the robot the
reachability search may reject it at every roll. A policy trained here will
therefore be optimistic about awkward angles. That is a known gap to close by
fine-tuning against the real grid, not something to paper over by adding a
guessed reachability model to a surface model that has no arm in it.

    python -m potato_scan.rl.cpu_scan_env --timesteps 200000
"""
import argparse

import gymnasium as gym
import numpy as np

from potato_scan.potato_surface import PotatoSurface
from potato_scan.scan_schedule import RasterOrbitSchedule
from potato_scan.surface_coverage import SurfaceCoverageGrid
from potato_scan.rl import scan_policy_spec


class CpuScanEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, potato_center=(0.50, 0.0, 0.15), scan_radius=0.15,
                 max_views=60, camera_min_range=0.07, camera_max_range=0.50,
                 camera_fov_deg=60.0, n_surface_points=6000,
                 azimuth_step_deg=45.0, elevation_step_deg=25.0, seed=None):
        super().__init__()
        self.potato_center = np.asarray(potato_center, dtype=float)
        self.scan_radius = float(scan_radius)
        self.max_views = int(max_views)
        self.camera = dict(min_range=camera_min_range, max_range=camera_max_range,
                           fov_deg=camera_fov_deg)
        self.n_surface_points = int(n_surface_points)
        self.azimuth_step_deg = azimuth_step_deg
        self.elevation_step_deg = elevation_step_deg

        self.observation_space = scan_policy_spec.observation_space()
        self.action_space = scan_policy_spec.action_space()
        self._rng = np.random.default_rng(seed)

    # --- helpers ----------------------------------------------------------

    def _capture(self, direction, radius_scale=1.0):
        """Take one view and fold whatever it saw into the grid."""
        camera = self.potato_center + np.asarray(direction) * self.scan_radius * radius_scale
        self._seen |= self.surface.visible(camera, **self.camera)
        self.grid.set_from_points(self.surface.points[self._seen], self.potato_center)
        self._views_taken += 1

    def _observation(self):
        return scan_policy_spec.build_observation(
            self.grid, self._last_direction, self._views_taken, self.max_views)

    # --- gym API ----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.surface = PotatoSurface(self.potato_center, self._rng,
                                     n_points=self.n_surface_points)
        self.grid = SurfaceCoverageGrid()
        self._seen = np.zeros(len(self.surface.points), dtype=bool)
        self._views_taken = 0
        self._last_direction = None

        # Phase A, run here rather than learned -- see the module docstring
        raster = RasterOrbitSchedule(
            min_elevation_deg=self.grid.min_elevation_deg,
            max_elevation_deg=self.grid.max_elevation_deg,
            azimuth_step_deg=self.azimuth_step_deg,
            elevation_step_deg=self.elevation_step_deg)
        while not raster.done:
            _, _, direction = raster.next_view()
            self._capture(direction)
            self._last_direction = direction

        self._raster_views = self._views_taken
        return self._observation(), {'raster_views': self._raster_views,
                                     'coverage_after_raster': self.grid.coverage_ratio()}

    def step(self, action):
        direction, radius_scale = scan_policy_spec.decode_action(
            action, self.grid.min_elevation_deg, self.grid.max_elevation_deg)

        cell = self.grid.direction_to_cell(direction)
        # "redundant" means aiming at a cell the scan has already resolved --
        # the cheapest mistake to make and the one worth naming, since the
        # coverage gain alone would score it merely as zero rather than wrong
        redundant = bool(self.grid.is_filled(*cell) or self.grid.unscannable[cell])

        before = int(self.grid.filled_mask().sum())
        self._capture(direction, radius_scale)
        after = int(self.grid.filled_mask().sum())

        # a cell aimed at and still empty is given up on, exactly as the
        # controller's recovery does, so the episode cannot stall on a spot
        # no view can reach
        if not self.grid.is_filled(*cell):
            self.grid.mark_unscannable(*cell)

        self._last_direction = direction
        reward = scan_policy_spec.step_reward(before, after, redundant)

        coverage = self.grid.coverage_ratio()
        resolved = self.grid.resolved_ratio()
        terminated = bool(coverage >= 0.95 or resolved >= 1.0)
        truncated = bool(self._views_taken >= self.max_views)
        if terminated or truncated:
            reward += scan_policy_spec.terminal_bonus(
                coverage, resolved, self._views_taken, self.max_views)

        return self._observation(), reward, terminated, truncated, {
            'coverage': coverage,
            'resolved': resolved,
            'views_taken': self._views_taken,
            'gap_filling_views': self._views_taken - self._raster_views,
            'eye_points_seen': float(self._seen[self.surface.is_eye].mean()),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--timesteps', type=int, default=200_000)
    parser.add_argument('--out', default='models/scan_policy_cpu')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    from stable_baselines3 import PPO           # lazy: only training needs it

    env = CpuScanEnv(seed=args.seed)
    model = PPO('MlpPolicy', env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps)
    model.save(args.out)
    print(f'saved {args.out}.zip -- point scan_controller at it with '
          f'view_policy:=rl rl_model_path:={args.out}.zip')


if __name__ == '__main__':
    main()
