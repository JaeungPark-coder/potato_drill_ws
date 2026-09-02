"""Shared observation/action/reward encoding for the RL scan-view policy.

Deliberately has NO Isaac Sim / rclpy imports -- this module is loaded both
by the Isaac-side training env (isaac/rl_scan_train_env.py) and by the ROS2
inference wrapper (rl/scan_policy_backend.py, used from scan_controller.py),
so training and deployment always agree on what the policy sees and how its
actions are decoded. Only needs numpy + gymnasium.

The policy replaces SurfaceCoverageGrid.next_gap_direction (nearest-gap
heuristic) with a learned choice of where to look next. It still operates on
the same live coverage grid (surface_coverage.SurfaceCoverageGrid) -- nothing
about how coverage is judged from the real/simulated point cloud changes.
"""
import numpy as np
from gymnasium import spaces

from ..surface_coverage import spherical_to_direction

# Fixed-size pooled coverage map, independent of the grid's configured
# elevation_bin_deg/azimuth_bin_deg -- keeps the observation shape stable
# even if those params are retuned.
POOLED_ELEV_BINS = 8
POOLED_AZ_BINS = 16

# decode_action's radius_scale maps into this range -- matches
# surface_coverage.RECOVERY_OFFSETS' radius_scale range, so the RL policy
# has at least as much reach as the heuristic's recovery motions.
RADIUS_SCALE_MIN = 0.75
RADIUS_SCALE_MAX = 1.35

OBS_DIM = POOLED_ELEV_BINS * POOLED_AZ_BINS + 3 + 1  # pooled map + last_direction(3) + views_frac(1)
ACTION_DIM = 3  # elevation_norm, azimuth_norm, radius_scale_norm


def observation_space():
    return spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)


def action_space():
    return spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)


def pooled_coverage_map(grid):
    """Downsample `grid`'s filled/unscannable status onto a fixed
    (POOLED_ELEV_BINS, POOLED_AZ_BINS) map via nearest-index pooling, so the
    observation size doesn't depend on the grid's actual resolution.
    Values: 0.0 = open (not yet filled, not given up on), 0.5 = unscannable,
    1.0 = filled."""
    filled = grid.filled_mask()
    unscannable = grid.unscannable
    n_elev, n_az = filled.shape

    e_src = np.clip((np.arange(POOLED_ELEV_BINS) * n_elev) // POOLED_ELEV_BINS, 0, n_elev - 1)
    a_src = np.clip((np.arange(POOLED_AZ_BINS) * n_az) // POOLED_AZ_BINS, 0, n_az - 1)

    pooled_filled = filled[np.ix_(e_src, a_src)]
    pooled_unscannable = unscannable[np.ix_(e_src, a_src)]

    out = np.zeros((POOLED_ELEV_BINS, POOLED_AZ_BINS), dtype=np.float32)
    out[pooled_unscannable] = 0.5
    out[pooled_filled] = 1.0
    return out


def build_observation(grid, last_direction, views_taken, max_views):
    pooled = pooled_coverage_map(grid).reshape(-1)

    if last_direction is None:
        direction_part = np.zeros(3, dtype=np.float32)
    else:
        d = np.asarray(last_direction, dtype=np.float32)
        direction_part = d / (np.linalg.norm(d) + 1e-9)

    views_frac = np.array([min(views_taken / max(max_views, 1), 1.0)], dtype=np.float32)

    obs = np.concatenate([pooled, direction_part, views_frac]).astype(np.float32)
    return obs


def decode_action(action, min_elevation_deg, max_elevation_deg):
    """action: (elevation_norm, azimuth_norm, radius_scale_norm), each in
    [-1, 1]. Returns (direction: unit vector (3,), radius_scale: float)."""
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)

    elevation_deg = min_elevation_deg + (action[0] + 1.0) / 2.0 * (max_elevation_deg - min_elevation_deg)
    azimuth_deg = (action[1] + 1.0) / 2.0 * 360.0
    radius_scale = RADIUS_SCALE_MIN + (action[2] + 1.0) / 2.0 * (RADIUS_SCALE_MAX - RADIUS_SCALE_MIN)

    direction = spherical_to_direction(elevation_deg, azimuth_deg)
    return direction, float(radius_scale)


def step_reward(prev_filled_count, new_filled_count, is_redundant_pick,
                 coverage_gain_weight=5.0, step_cost=0.02, redundant_penalty=0.05):
    """Reward for one view attempt. `prev_filled_count`/`new_filled_count`
    are the number of filled cells in the grid before/after the attempt."""
    gain = max(new_filled_count - prev_filled_count, 0)
    reward = gain * coverage_gain_weight - step_cost
    if is_redundant_pick:
        reward -= redundant_penalty
    return float(reward)


def terminal_bonus(coverage_ratio, resolved_ratio, views_taken, max_views, bonus_weight=10.0):
    """One-off bonus at episode end, rewarding finishing with fewer views
    (encourages an efficient view policy, not just an eventually-complete
    one)."""
    if resolved_ratio < 1.0 and views_taken >= max_views:
        # ran out of views without resolving every cell -- no efficiency
        # bonus, just the coverage_ratio reached so far as partial credit.
        return float(coverage_ratio * bonus_weight * 0.5)
    efficiency = 1.0 - min(views_taken / max(max_views, 1), 1.0)
    return float(coverage_ratio * bonus_weight * (0.5 + 0.5 * efficiency))
