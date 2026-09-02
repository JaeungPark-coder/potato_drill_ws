"""ROS2-side inference wrapper for a trained drill-approach RL policy.

Loads a stable_baselines3 PPO checkpoint (trained via
isaac/train_drill_policy.py) and exposes find_approach(...), a drop-in
replacement for drill_controller._find_reachable_approach's ROLL_SEARCH_DEG
sweep -- see the `approach_policy` parameter there.
"""
from . import drill_policy_spec


class RLApproachPolicy:
    def __init__(self, model_path, max_attempts=5):
        from stable_baselines3 import PPO  # lazy import: only needed when approach_policy == 'rl'
        if not model_path:
            raise ValueError("approach_policy is 'rl' but rl_model_path is empty -- set it in params.yaml")
        self.model = PPO.load(model_path)
        self.max_attempts = max_attempts

    def find_approach(self, robot, position, normal, standoff):
        """Tries up to `max_attempts` policy-proposed approach poses via
        robot.move_to_pose. Returns (approach_position, rotvec) of the pose
        actually reached, or (None, None) if every attempt was rejected --
        same contract as drill_controller._find_reachable_approach."""
        for _ in range(self.max_attempts):
            tcp_pos, _ = robot.get_tcp_pose()
            if tcp_pos is None:
                # isaac_sim backend before TF is up yet -- fall back to the
                # eye position itself as a stand-in "current pose" observation.
                tcp_pos = position
            obs = drill_policy_spec.build_observation(position, normal, tcp_pos)
            action, _ = self.model.predict(obs, deterministic=True)
            roll_deg, lateral_xy = drill_policy_spec.decode_action(action)
            approach, rotvec = drill_policy_spec.compose_approach_pose(
                position, normal, standoff, roll_deg, lateral_xy)
            if robot.move_to_pose(approach, rotvec):
                return approach, rotvec
        return None, None
