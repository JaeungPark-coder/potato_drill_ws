"""ROS2-side inference wrapper for a trained scan-view RL policy.

Loads a stable_baselines3 PPO checkpoint (trained via
isaac/train_scan_policy.py) and exposes next_view(...), a drop-in
replacement for SurfaceCoverageGrid.next_gap_direction's role in
scan_controller.py -- see the `view_policy` parameter there.
"""
from . import scan_policy_spec


class RLViewPolicy:
    def __init__(self, model_path, min_elevation_deg, max_elevation_deg):
        from stable_baselines3 import PPO  # lazy import: only needed when view_policy == 'rl'
        if not model_path:
            raise ValueError("view_policy is 'rl' but rl_model_path is empty -- set it in params.yaml")
        self.model = PPO.load(model_path)
        self.min_elevation_deg = min_elevation_deg
        self.max_elevation_deg = max_elevation_deg

    def next_view(self, coverage_grid, last_direction, views_taken, max_views):
        """Returns (direction: unit vector (3,), radius_scale: float),
        matching what scan_controller._attempt_view needs to move to the
        chosen view."""
        obs = scan_policy_spec.build_observation(coverage_grid, last_direction, views_taken, max_views)
        action, _ = self.model.predict(obs, deterministic=True)
        return scan_policy_spec.decode_action(action, self.min_elevation_deg, self.max_elevation_deg)
