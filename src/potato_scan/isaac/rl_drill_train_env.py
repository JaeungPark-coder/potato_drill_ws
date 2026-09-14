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
from isaacsim.core.prims import SingleArticulation  # noqa: E402

# Bridges the ROS2-installed potato_scan package onto sys.path. Requires the
# workspace to be colcon-built and `source install/setup.bash`'d in the
# terminal BEFORE launching this via python.sh.
#
# Sourcing is safe HERE and is not safe everywhere: this file never imports
# rclpy, so it only wants potato_scan on the path. isaac_scene.py does import
# rclpy, and sourcing system ROS 2 puts its Python-3.10 build ahead of the
# Python-3.11 one Kit ships -- which is why the README tells you to run THAT
# script under `env -i` with nothing sourced. Do not carry this line's advice
# over to it.
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
        # SingleArticulation (unbatched), not the vectorized Articulation --
        # CONFIRMED (2026-09-07): RmpFlow's ArticulationMotionPolicy needs
        # get_articulation_controller(), which only SingleArticulation
        # provides (checked directly against the installed Isaac Sim API).
        # Same class rl_scan_train_env.py / vla_ur5e_ws use for their own
        # RMPflow-driven arm.
        self.robot = SingleArticulation(ROBOT_PRIM_PATH, name="ur5e_arm")
        self.world.reset()  # initializes physics handles for the articulation
        self.robot.initialize()

        self.rmpflow, self.articulation_policy = setup_rmpflow(self.robot)

        # Poses are commanded to RMPflow, which drives the frame its config
        # names ("tool0"), but the drill tip has to be parented to a real USD
        # prim (TOOL_LINK_PRIM_PATH). MEASURED (2026-09-07): the two share a
        # position exactly but differ in orientation by about (-90, -90, 0)
        # degrees on this asset, so mounting the bit straight onto the flange
        # would aim it well away from the insertion axis force_drill drives.
        # Same correction rl_scan_train_env.py applies to its camera.
        q0 = np.asarray(self.robot.get_joint_positions())[:6]
        _, tool0_rot = self.rmpflow.get_end_effector_pose(q0)
        tool0_rot = np.asarray(tool0_rot)
        r_tool0 = (Rot.from_matrix(tool0_rot) if tool0_rot.shape == (3, 3)
                   else Rot.from_quat(tool0_rot[[1, 2, 3, 0]]))
        _, flange_quat = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        self.r_flange_to_tool0 = Rot.from_quat(flange_quat).inv() * r_tool0

        drill_tip_path = add_drill_tip(self.stage, TOOL_LINK_PRIM_PATH,
                                        r_parent_to_tool=self.r_flange_to_tool0)
        self.contact_reader = ContactForceReader(drill_tip_path)

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

        # See __init__'s comment -- a non-soft world.reset() invalidates
        # self.robot's physics handles every time, not just once.
        self.world.reset()  # re-homes the robot articulation
        self.robot.initialize()

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

        # Console progress readout -- this env renders nothing task-specific
        # in the GUI, so without this there is no way to tell an insertion
        # that reached depth from one that stalled on force.
        if not moved:
            _pe, _re, _cur, _tgt = self._last_move_err
            print(f"  attempt {self.attempts}/{self.max_attempts_per_episode}: "
                  f"UNREACHABLE approach pose, reward={reward:+.2f} "
                  f"[pos_err={_pe * 1000:.0f}mm rot_err={_re:.0f}deg "
                  f"target={np.round(_tgt, 3)} actual={np.round(_cur, 3)} "
                  f"eye={np.round(self.eye_position, 3)}]")
        else:
            print(f"  attempt {self.attempts}/{self.max_attempts_per_episode}: "
                  f"{'REACHED depth' if reached else 'stopped on force'} "
                  f"(force_overshoot={force_overshoot_ratio:.2f}, roll={roll_deg:+.0f}deg) "
                  f"reward={reward:+.2f}")
        return obs, reward, terminated, truncated, info

    def _move_and_settle(self, target_pos, target_rotvec, pos_tol_m=0.02, rot_tol_deg=8.0):
        """Same tolerance-polling contract as
        IsaacSimRobotInterface.move_to_pose (over ROS2/TF at deployment),
        done here with direct in-process prim reads. Returns True if the
        tool0 link settled within tolerance before move_timeout_s.

        MEASURED (2026-09-07) against this scene: RMPflow is a reactive
        controller and lands roughly 2-20mm / 2-12deg from these targets, so
        the original 5mm/3deg tolerance was effectively never met. That
        matters far more here than in the scan env: step() treats a False
        return as "unreachable" and skips the insertion entirely, so with the
        old tolerance the policy would have been scored on attempts it never
        actually drilled."""
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

            # Compare against RMPflow's OWN end-effector frame, not the USD
            # flange prim. Both are commanded the same target, but the two
            # frames differ in orientation by about (-90, -90, 0) degrees on
            # this asset (see __init__), so reading the flange's rotation made
            # rot_err permanently ~127deg and this loop could never report
            # success -- which step() then scored as "unreachable", skipping
            # every single insertion.
            cur_pos, cur_rot = self.rmpflow.get_end_effector_pose(
                np.asarray(self.robot.get_joint_positions())[:6])
            cur_rot = np.asarray(cur_rot)
            r_cur = (Rot.from_matrix(cur_rot) if cur_rot.shape == (3, 3)
                     else Rot.from_quat(cur_rot[[1, 2, 3, 0]]))
            pos_err = float(np.linalg.norm(np.asarray(cur_pos) - target_pos))
            rot_err = (r_cur.inv() * target_rot).magnitude()
            if pos_err <= pos_tol_m and rot_err <= rot_tol_rad:
                reached = True
                break

        self._last_move_err = (pos_err, float(np.degrees(rot_err)),
                               np.asarray(cur_pos), target_pos)

        for _ in range(self.settle_steps):
            self.world.step(render=True)
        return reached

    def _force_insert(self, approach_pos, rotvec):
        """Feeds along the +Z axis of `rotvec` -- which compose_approach_pose
        now aims INTO the surface -- in small position steps, reading
        simulated contact force each step.

        The deployment-side force_drill implementations
        (isaac_robot_interface.py and robot_interface.py) now feed along +Z
        too, and drill_controller builds its tool rotation from -normal, so
        sim and deployment drill in the same direction. They differ only in
        where depth is measured from: deployment references it to a
        detected contact point (the standoff gap is unknown there), while
        this env starts already at the approach pose with the surface
        position known exactly.

        Returns (reached: bool, force_overshoot_ratio: float) -- reached
        is True if max_depth was hit before max_force; force_overshoot_ratio
        is how far the peak observed force went past max_force, as a
        fraction of max_force (0 if it never got close)."""
        rot_matrix = Rot.from_rotvec(np.asarray(rotvec, dtype=float)).as_matrix()
        # compose_approach_pose now aims the tool INTO the surface (+Z =
        # -normal, see its comment: the outward-facing pose is kinematically
        # unreachable because the wrist would have to sit inside the potato),
        # so feeding in is +axis. The drill bit add_drill_tip mounts along the
        # tool's +Z leads the way, as it should.
        insertion_axis = rot_matrix[:, 2]  # +Z now points INTO the surface
        target_quat_wxyz = Rot.from_rotvec(rotvec).as_quat()[[3, 0, 1, 2]]

        depth = 0.0
        reached = False
        max_force_ratio_seen = 0.0
        t0 = time.time()
        while time.time() - t0 < self.insertion_timeout_s:
            depth = min(depth + self.insertion_step_size_m, self.max_depth)
            target_pos = np.asarray(approach_pos, dtype=float) + insertion_axis * depth

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
