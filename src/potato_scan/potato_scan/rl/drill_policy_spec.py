"""Shared observation/action/reward encoding for the RL drill-approach
policy.

Deliberately has NO Isaac Sim / rclpy imports -- loaded both by the
Isaac-side training env (isaac/rl_drill_train_env.py) and by the ROS2
inference wrapper (rl/drill_policy_backend.py, used from drill_controller.py).

The policy replaces drill_controller.ROLL_SEARCH_DEG's fixed roll sweep with
a learned choice of (roll, small lateral offset) around the nominal
normal-aligned approach pose -- same +Z-is-outward-normal convention as
eye_detector.normal_to_quat / drill_controller.normal_rotation, reimplemented
here (not imported from drill_controller.py) to avoid a circular import
(drill_controller -> rl.drill_policy_backend -> this module).
"""
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as Rot

from ..drill_task_planner import approach_pose

# Lateral offset action maps into +/- this many meters, in the tangent plane
# of the approach -- small enough to stay "the same eye", large enough to
# dodge a locally-unreachable approach point.
LATERAL_OFFSET_MAX_M = 0.005

OBS_DIM = 9  # eye_position(3) + eye_normal(3) + current_tcp_position(3), base frame, meters
ACTION_DIM = 3  # roll_norm, lateral_x_norm, lateral_y_norm


def observation_space():
    return spaces.Box(low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32)


def action_space():
    return spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)


def build_observation(eye_position, eye_normal, tcp_position):
    eye_position = np.asarray(eye_position, dtype=np.float32)
    normal = np.asarray(eye_normal, dtype=np.float32)
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    tcp_position = np.asarray(tcp_position, dtype=np.float32)
    return np.concatenate([eye_position, normal, tcp_position]).astype(np.float32)


def decode_action(action):
    """action: (roll_norm, lateral_x_norm, lateral_y_norm), each in [-1, 1].
    Returns (roll_deg, lateral_xy: np.ndarray(2,) in meters)."""
    action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    roll_deg = float(action[0] * 180.0)
    lateral_xy = action[1:3] * LATERAL_OFFSET_MAX_M
    return roll_deg, lateral_xy


def normal_rotation(normal):
    """Rotation matrix whose +Z axis is `normal` -- must match
    eye_detector.normal_to_quat / drill_controller.normal_rotation exactly,
    so approach orientations line up with the rest of the pipeline."""
    z = np.asarray(normal, dtype=float)
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


def compose_approach_pose(position, normal, standoff, roll_deg, lateral_xy):
    """Nominal standoff point along `normal` (drill_task_planner.approach_pose),
    nudged by `lateral_xy` in the approach's own tangent plane, with `roll_deg`
    applied about the insertion axis -- same roll convention
    drill_controller._find_reachable_approach uses. Returns
    (approach_position (3,), rotvec (3,))."""
    base_rotation = normal_rotation(normal)
    x_axis, y_axis = base_rotation[:, 0], base_rotation[:, 1]

    approach = approach_pose(position, normal, standoff)
    approach = approach + x_axis * lateral_xy[0] + y_axis * lateral_xy[1]

    roll = Rot.from_euler('z', roll_deg, degrees=True).as_matrix()
    rotvec = Rot.from_matrix(base_rotation @ roll).as_rotvec()
    return approach, rotvec


def deviation_norm(roll_deg, lateral_xy):
    """0 at the nominal (no-roll, no-offset) pose, growing toward 1 as the
    action pushes further from it -- penalizes unnecessary deviation so the
    policy prefers the simplest approach that still works."""
    roll_term = abs(roll_deg) / 180.0
    lateral_term = float(np.linalg.norm(lateral_xy)) / (LATERAL_OFFSET_MAX_M * np.sqrt(2))
    return float(np.clip(0.5 * roll_term + 0.5 * lateral_term, 0.0, 1.0))


def attempt_reward(reached, force_overshoot_ratio, roll_deg, lateral_xy, unreachable=False,
                    success_bonus=10.0, force_penalty_weight=5.0, deviation_weight=1.0,
                    unreachable_penalty=8.0):
    """One-shot reward for a single drill approach + insertion attempt.

    reached: True if force_drill reached max_depth without hitting max_force.
    force_overshoot_ratio: how far force went past max_force relative to
      max_force (0 if it never got close), only meaningful when not reached.
    unreachable: True if the approach pose itself couldn't be reached
      (move_to_pose failed) -- insertion was never attempted.
    """
    if unreachable:
        return -unreachable_penalty - deviation_weight * deviation_norm(roll_deg, lateral_xy)

    reward = success_bonus if reached else -force_penalty_weight * min(force_overshoot_ratio, 3.0)
    reward -= deviation_weight * deviation_norm(roll_deg, lateral_xy)
    return float(reward)
