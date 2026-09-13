"""Shared observation/action/reward encoding for the RL drill-approach
policy.

Deliberately has NO Isaac Sim / rclpy imports -- loaded both by the
Isaac-side training env (isaac/rl_drill_train_env.py) and by the ROS2
inference wrapper (rl/drill_policy_backend.py, used from drill_controller.py).

The policy replaces drill_task_planner.ROLL_SEARCH_DEG's fixed roll sweep
with a learned choice of (roll, small lateral offset) around the nominal
normal-aligned approach pose -- the same +Z-is-outward-normal convention as
eye_detector.normal_to_quat, imported from drill_task_planner rather than
copied. It used to be copied, to break a circular import through
drill_controller; the convention has since moved into drill_task_planner,
which imports nothing from this package, so the copy was removable -- and
worth removing, because two identical copies of a rotation convention are
two copies that can drift apart without any test noticing.
"""
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as Rot

from ..drill_task_planner import approach_pose, normal_rotation

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


def compose_approach_pose(position, normal, standoff, roll_deg, lateral_xy):
    """Nominal standoff point along `normal` (drill_task_planner.approach_pose),
    nudged by `lateral_xy` in the approach's own tangent plane, with `roll_deg`
    applied about the insertion axis -- same roll convention
    drill_controller._find_reachable_approach uses, from
    drill_task_planner.ROLL_SEARCH_DEG. Returns
    (approach_position (3,), rotvec (3,))."""
    # The tool frame points INTO the surface (+Z = -normal), while the
    # approach POSITION stays standoff metres out along the outward normal.
    # normal_rotation's own "+Z is the outward normal" convention is left
    # alone -- it is the perception-side contract shared with
    # eye_detector.normal_to_quat -- and only the tool command is flipped.
    #
    # MEASURED (2026-09-08, RMPflow against a real potato mesh): a UR5e's
    # wrist extends back along the tool's -Z, so commanding +Z = +normal
    # asks the wrist to occupy the potato's own volume. Across three real
    # eyes that pose was never reachable (position error 78/102/154mm,
    # rotation error 17/27/88deg), while +Z = -normal reached every one of
    # them (18/19/18mm, 2/1/6deg). The drill bit, which add_drill_tip
    # extends along the tool's +Z, correspondingly now points into the
    # surface rather than away from it.
    base_rotation = normal_rotation(-np.asarray(normal, dtype=float))
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
