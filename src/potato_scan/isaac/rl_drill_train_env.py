"""Gymnasium env for training the drill-approach RL policy inside Isaac
Sim's own physics -- one real robot move + (if reached) one real
force-limited insertion per env.step(), so this is slow by design (see
the plan this was built from: training realism was chosen over training
speed).

NOT a ROS2 node -- run only via `<isaac-sim-install-dir>/python.sh`, same
as isaac_scene.py (needs Isaac's own Kit runtime). Not imported directly;
see train_drill_policy.py.

Ground-truth eye positions/normals come straight from
isaac_sim_common.make_potato_mesh's procedural generation (the same pit
locations eye_detector.py would otherwise have to find from a point cloud)
-- this env is downstream of detection, so it skips the vision pipeline
entirely and trains against known eyes.

Deliberately duplicates a small amount of RMPflow move/settle logic from
rl_scan_train_env.py rather than importing it: both modules construct
their own module-level Isaac Sim `SimulationApp` at import time (mirroring
isaac_scene.py's own top-level `simulation_app = SimulationApp(...)`), and
Isaac Sim only supports one `SimulationApp` per process, so the two env
modules must stay import-independent of each other.

Written and reasoned about WITHOUT the ability to run Isaac Sim in the
environment this was authored in -- treat this as a solid first draft,
not verified to run. "ADJUST:" marks the spots most likely to need
on-machine tweaking, same convention isaac_scene.py uses.
"""
import os
import time

import numpy as np
import gymnasium as gym
from scipy.spatial.transform import Rotation as Rot

from isaacsim import SimulationApp

# Headless is the sane default for training (potentially thousands of
# episodes) -- set ISAAC_DRILL_ENV_HEADLESS=0 before launching
# train_drill_policy.py to open the Kit GUI and watch training instead.
HEADLESS = os.environ.get("ISAAC_DRILL_ENV_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.nucleus import get_assets_root_path  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402

# Bridges the ROS2-installed potato_scan package onto sys.path -- same
# mechanism isaac_scene.py already relies on for `import rclpy`. Requires
# the workspace to be colcon-built and `source install/setup.bash`'d in
# the terminal BEFORE launching this via python.sh.
enable_extension("isaacsim.ros2.bridge")  # noqa: E402

from isaac_sim_common import (  # noqa: E402
    UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH, TOOL_LINK_PRIM_PATH,
    make_potato_mesh, add_drill_tip, ContactForceReader, setup_rmpflow, prim_world_pose,
)
from potato_scan.rl import drill_policy_spec  # noqa: E402


class IsaacDrillEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, potato_center=(0.50, 0.00, 0.15), standoff=0.03,
                 feed_force=15.0, max_force=40.0, max_depth=0.015,
                 max_attempts_per_episode=5, move_timeout_s=6.0, settle_steps=60,
                 insertion_step_size_m=0.0005, insertion_timeout_s=8.0):
        super().__init__()
        self.simulation_app = simulation_app

        self.potato_center = np.array(potato_center, dtype=float)
        self.standoff = standoff
        self.feed_force = feed_force
        self.max_force = max_force
        self.max_depth = max_depth
        self.max_attempts_per_episode = max_attempts_per_episode
        self.move_timeout_s = move_timeout_s
        self.settle_steps = settle_steps
        self.insertion_step_size_m = insertion_step_size_m
        self.insertion_timeout_s = insertion_timeout_s
        self.physics_dt = 1.0 / 60.0
        self._rng = np.random.default_rng()

        self.observation_space = drill_policy_spec.observation_space()
        self.action_space = drill_policy_spec.action_space()

        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Could not resolve Isaac Sim assets root -- check Nucleus connection.")

        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        self.stage = get_current_stage()

        add_reference_to_stage(assets_root + UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH)
        self.robot = Articulation(ROBOT_PRIM_PATH)
        self.world.reset()  # initializes physics handles for the articulation

        drill_tip_path = add_drill_tip(self.stage, TOOL_LINK_PRIM_PATH)
        self.contact_reader = ContactForceReader(drill_tip_path)

        self.rmpflow, self.articulation_policy = setup_rmpflow(self.robot)

        self.eye_position = None
        self.eye_normal = None
        self.attempts = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mesh_seed = seed if seed is not None else int(self._rng.integers(0, 2**31 - 1))

        potato_prim_path = "/World/potato"
        if self.stage.GetPrimAtPath(potato_prim_path).IsValid():
            self.stage.RemovePrim(potato_prim_path)
        _, eye_points, eye_normals = make_potato_mesh(
            self.stage, potato_prim_path, self.potato_center, seed=mesh_seed)

        self.world.reset()  # re-homes the robot articulation

        idx = int(self._rng.integers(0, len(eye_points)))
        self.eye_position = eye_points[idx]
        self.eye_normal = eye_normals[idx]
        self.attempts = 0

        tcp_pos, _ = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        obs = drill_policy_spec.build_observation(self.eye_position, self.eye_normal, tcp_pos)
        return obs, {}

    def step(self, action):
        roll_deg, lateral_xy = drill_policy_spec.decode_action(action)
        approach, rotvec = drill_policy_spec.compose_approach_pose(
            self.eye_position, self.eye_normal, self.standoff, roll_deg, lateral_xy)

        self.attempts += 1
        moved = self._move_and_settle(approach, rotvec)

        if not moved:
            reward = drill_policy_spec.attempt_reward(
                reached=False, force_overshoot_ratio=0.0,
                roll_deg=roll_deg, lateral_xy=lateral_xy, unreachable=True)
            terminated = False
            truncated = self.attempts >= self.max_attempts_per_episode
        else:
            reached, force_overshoot_ratio = self._force_insert(approach, rotvec)
            reward = drill_policy_spec.attempt_reward(
                reached=reached, force_overshoot_ratio=force_overshoot_ratio,
                roll_deg=roll_deg, lateral_xy=lateral_xy)
            self._move_and_settle(approach, rotvec)  # retract back to the standoff pose
            terminated = True
            truncated = False

        tcp_pos, _ = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        obs = drill_policy_spec.build_observation(self.eye_position, self.eye_normal, tcp_pos)
        info = {"attempts": self.attempts}
        return obs, reward, terminated, truncated, info

    def _move_and_settle(self, target_pos, target_rotvec, pos_tol_m=0.005, rot_tol_deg=3.0):
        """Same tolerance-polling contract as
        IsaacSimRobotInterface.move_to_pose (over ROS2/TF at deployment),
        done here with direct in-process prim reads. Returns True if the
        tool0 link settled within tolerance before move_timeout_s."""
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = Rot.from_rotvec(np.asarray(target_rotvec, dtype=float))
        target_quat_wxyz = target_rot.as_quat()[[3, 0, 1, 2]]
        rot_tol_rad = np.radians(rot_tol_deg)

        t0 = time.time()
        reached = False
        while time.time() - t0 < self.move_timeout_s:
            self.rmpflow.set_end_effector_target(target_pos, target_quat_wxyz)
            self.rmpflow.update_world()
            action = self.articulation_policy.get_next_articulation_action(self.physics_dt)
            self.robot.apply_action(action)
            self.world.step(render=True)

            cur_pos, cur_quat = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
            cur_rotvec = Rot.from_quat(cur_quat).as_rotvec()
            pos_err = float(np.linalg.norm(cur_pos - target_pos))
            rot_err = (Rot.from_rotvec(cur_rotvec).inv() * target_rot).magnitude()
            if pos_err <= pos_tol_m and rot_err <= rot_tol_rad:
                reached = True
                break

        for _ in range(self.settle_steps):
            self.world.step(render=True)
        return reached

    def _force_insert(self, approach_pos, rotvec):
        """Feeds along the negative Z axis of `rotvec` (the approach's own
        insertion axis, i.e. INTO the surface) in small position steps,
        reading simulated contact force each step -- same step-based
        approximation isaac_robot_interface.IsaacSimRobotInterface.force_drill
        uses at deployment (axis_index=2, same step_size_m default).

        Returns (reached: bool, force_overshoot_ratio: float) -- reached
        is True if max_depth was hit before max_force; force_overshoot_ratio
        is how far the peak observed force went past max_force, as a
        fraction of max_force (0 if it never got close)."""
        rot_matrix = Rot.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
        insertion_axis = rot_matrix[:, 2]  # +Z = outward normal-ish; feed INTO the surface = -axis
        target_quat_wxyz = Rot.from_rotvec(rotvec).as_quat()[[3, 0, 1, 2]]

        depth = 0.0
        reached = False
        max_force_ratio_seen = 0.0
        t0 = time.time()
        while time.time() - t0 < self.insertion_timeout_s:
            depth = min(depth + self.insertion_step_size_m, self.max_depth)
            target_pos = np.asarray(approach_pos, dtype=float) - insertion_axis * depth

            self.rmpflow.set_end_effector_target(target_pos, target_quat_wxyz)
            self.rmpflow.update_world()
            action = self.articulation_policy.get_next_articulation_action(self.physics_dt)
            self.robot.apply_action(action)
            self.world.step(render=True)

            force_mag = float(np.linalg.norm(self.contact_reader.read()))
            max_force_ratio_seen = max(max_force_ratio_seen, force_mag / self.max_force)
            if force_mag >= self.max_force:
                reached = False
                break
            if depth >= self.max_depth:
                reached = True
                break

        force_overshoot_ratio = max(max_force_ratio_seen - 1.0, 0.0)
        return reached, force_overshoot_ratio

    def close(self):
        self.simulation_app.close()
