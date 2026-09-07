"""Gymnasium env for training the scan-view RL policy inside Isaac Sim's
own physics -- one real robot move + settle per env.step(), so this is
slow compared to a GPU-vectorized simulator, by design (see the plan this
was built from: training realism was chosen over training speed).

NOT a ROS2 node -- run only via `<isaac-sim-install-dir>/python.sh`, same
as isaac_scene.py (needs Isaac's own Kit runtime). Not imported directly;
see train_scan_policy.py.

Reuses potato_scan.surface_coverage.SurfaceCoverageGrid and
potato_scan.pose_utils exactly as scan_controller.py does at deployment,
and potato_scan.rl.scan_policy_spec for the observation/action/reward
encoding shared with the ROS2-side inference wrapper
(potato_scan/rl/scan_policy_backend.py) -- so what gets trained here
matches what runs at deployment. Only the Isaac-specific scene plumbing
(potato mesh, camera render, RMPflow stepping) is new, via
isaac_sim_common.py (shared with isaac_scene.py).

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
# episodes) -- set ISAAC_SCAN_ENV_HEADLESS=0 before launching
# train_scan_policy.py to open the Kit GUI and watch training instead.
HEADLESS = os.environ.get("ISAAC_SCAN_ENV_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
import carb  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from pxr import UsdGeom, Gf  # noqa: E402

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.nucleus import get_assets_root_path  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402

# Bridges the ROS2-installed potato_scan package onto sys.path -- same
# mechanism isaac_scene.py already relies on for `import rclpy`. Requires
# the workspace to be colcon-built and `source install/setup.bash`'d in
# the terminal BEFORE launching this via python.sh.
enable_extension("isaacsim.ros2.bridge")  # noqa: E402

from isaac_sim_common import (  # noqa: E402
    UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH, TOOL_LINK_PRIM_PATH,
    make_potato_mesh, setup_rmpflow, prim_world_pose,
)
from potato_scan.pose_utils import look_at_rotation, camera_pose_to_tcp_pose, rotmat_to_rotvec  # noqa: E402
from potato_scan.surface_coverage import SurfaceCoverageGrid  # noqa: E402
from potato_scan.rl import scan_policy_spec  # noqa: E402


class IsaacScanEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, potato_center=(0.50, 0.00, 0.15), scan_radius=0.15,
                 elevation_bin_deg=8.0, azimuth_bin_deg=8.0, min_elevation_deg=-15.0,
                 max_elevation_deg=85.0, min_hits_to_fill=3, coverage_threshold=0.95,
                 max_views=40, settle_steps=90, move_timeout_s=6.0,
                 tcp_cam_translation=(0.0, -0.05, 0.05), tcp_cam_quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
        super().__init__()
        self.simulation_app = simulation_app

        self.potato_center = np.array(potato_center, dtype=float)
        self.scan_radius = scan_radius
        self.elevation_bin_deg = elevation_bin_deg
        self.azimuth_bin_deg = azimuth_bin_deg
        self.min_elevation_deg = min_elevation_deg
        self.max_elevation_deg = max_elevation_deg
        self.min_hits_to_fill = min_hits_to_fill
        self.coverage_threshold = coverage_threshold
        self.max_views = max_views
        self.settle_steps = settle_steps
        self.move_timeout_s = move_timeout_s
        self.r_tcp_cam = Rot.from_quat(tcp_cam_quat_xyzw).as_matrix()
        self.t_tcp_cam = np.array(tcp_cam_translation, dtype=float)
        self.physics_dt = 1.0 / 60.0
        self._rng = np.random.default_rng()

        self.observation_space = scan_policy_spec.observation_space()
        self.action_space = scan_policy_spec.action_space()

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
        # provides (checked directly against the installed Isaac Sim API:
        # hasattr(Articulation, "get_articulation_controller") is False,
        # hasattr(SingleArticulation, ...) is True) -- same class vla_ur5e_ws
        # (sibling project) uses for its own RMPflow-driven arm.
        self.robot = SingleArticulation(ROBOT_PRIM_PATH, name="ur5e_arm")
        self.world.reset()  # initializes physics handles for the articulation
        self.robot.initialize()

        camera_path = f"{TOOL_LINK_PRIM_PATH}/camera"
        camera = UsdGeom.Camera.Define(self.stage, camera_path)
        camera.AddTranslateOp().Set(Gf.Vec3d(*tcp_cam_translation))
        # CONFIRMED (2026-09-07, actual Isaac Sim run) root cause of coverage
        # staying at 0.0% forever: a USD camera images along its local -Z
        # (OpenGL convention), but pose_utils.look_at_rotation -- and the
        # r_tcp_cam hand-eye rotation it composes with -- uses the
        # OpenCV/robotics convention where the optical axis is +Z (see that
        # function's own docstring). With no rotation op here the camera
        # inherits the flange's orientation verbatim, so once _move_and_settle
        # aimed the flange's +Z at the potato the camera was looking exactly
        # 180 degrees the OTHER way: the arm reached its target to within
        # 1.7mm while the point cloud contained nothing but floor/background
        # (nearest point ~0.9m from a potato sitting 0.13m away, zero points
        # ever inside the expected 0.015-0.07m radius band). RotateX(180)
        # maps the camera's -Z view direction onto the flange's +Z.
        camera.AddRotateXOp().Set(180.0)
        # CONFIRMED (2026-09-07, distance sweep against this exact scene): a
        # USD camera's default clippingRange is (1.0, 1000000), i.e. a 1 METRE
        # near plane -- so a potato orbited at scan_radius=0.15m was entirely
        # clipped away and the point cloud contained nothing but floor and
        # background beyond 1m. Measured directly: with the default range,
        # points within the potato's 0.015-0.07m radius band = 0 at every
        # camera distance tried (0.15/0.35/0.8/1.5/3.0m); with near=0.01 the
        # same poses return 76389/23001/4370/1239/310 in-band points and the
        # nearest returned point sits at 0.030m from the potato centre,
        # exactly its surface radius.
        camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
        render_product = rep.create.render_product(camera_path, (640, 480))
        # CONFIRMED (2026-09-07, actual Isaac Sim run): the "pointcloud"
        # annotator's OGN node filters by semantic segmentation ID by
        # default (threshold=1 unless includeUnlabelled=True -- see
        # OgnPointCloudGenerator.py), so with no semantic label on the
        # potato mesh this returned an empty (0,) point array every single
        # step -- coverage stayed at 0% for the entire episode. Verified
        # the fix directly: same scene/camera setup returned (0,) points
        # without this param, 27120 points with it.
        self.pointcloud_annotator = rep.AnnotatorRegistry.get_annotator(
            "pointcloud", init_params={"includeUnlabelled": True})
        self.pointcloud_annotator.attach([render_product])

        self.rmpflow, self.articulation_policy = setup_rmpflow(self.robot)

        self.coverage = None
        self.views_taken = 0
        self.last_direction = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mesh_seed = seed if seed is not None else int(self._rng.integers(0, 2**31 - 1))

        potato_prim_path = "/World/potato"
        if self.stage.GetPrimAtPath(potato_prim_path).IsValid():
            self.stage.RemovePrim(potato_prim_path)
        make_potato_mesh(self.stage, potato_prim_path, self.potato_center, seed=mesh_seed)

        # A non-soft world.reset() is a hard reset (Stop+Play) under the
        # hood, which invalidates self.robot's physics handles (it's a
        # SingleArticulation not registered via world.scene) -- CONFIRMED
        # (2026-09-07, same fix already needed and verified in vla_ur5e_ws):
        # skipping this re-initialize() causes an "is_homogeneous"
        # AttributeError / stale physics view the next time apply_action or
        # RMPflow's get_next_articulation_action touches self.robot.
        self.world.reset()  # re-homes the robot articulation
        self.robot.initialize()

        self.coverage = SurfaceCoverageGrid(
            elevation_bin_deg=self.elevation_bin_deg, azimuth_bin_deg=self.azimuth_bin_deg,
            min_elevation_deg=self.min_elevation_deg, max_elevation_deg=self.max_elevation_deg,
            min_hits_to_fill=self.min_hits_to_fill)
        self.views_taken = 0
        self.last_direction = None
        # CONFIRMED (2026-09-07, actual run): SurfaceCoverageGrid.set_from_points
        # is idempotent -- it does self.hits[:] = 0 and rebuilds entirely from
        # whatever `points` it's given each call (by design, matching the ROS2
        # deployment side's pointcloud_accumulator.py, which merges points
        # across views itself before calling set_from_points once per merge).
        # This env has no such accumulator -- _read_pointcloud_world() only
        # returns the CURRENT frame's points -- so without accumulating them
        # here across the episode's views, coverage stayed at 0% forever:
        # every step() overwrote the previous view's hits with just the new
        # view's, never letting any cell reach min_hits_to_fill from repeated
        # views.
        self._accumulated_points = np.empty((0, 3), dtype=np.float32)

        obs = scan_policy_spec.build_observation(self.coverage, None, 0, self.max_views)
        return obs, {}

    def step(self, action):
        direction, radius_scale = scan_policy_spec.decode_action(
            action, self.min_elevation_deg, self.max_elevation_deg)

        e, a = self.coverage.direction_to_cell(direction)
        was_filled_before = self.coverage.is_filled(e, a)

        cam_pos = self.potato_center + direction * self.scan_radius * radius_scale
        cam_rot = look_at_rotation(cam_pos, self.potato_center)
        tcp_pos, tcp_rot = camera_pose_to_tcp_pose(cam_pos, cam_rot, self.r_tcp_cam, self.t_tcp_cam)
        tcp_rotvec = rotmat_to_rotvec(tcp_rot)
        final_pos_err = self._move_and_settle(tcp_pos, tcp_rotvec)

        prev_filled_count = int(self.coverage.filled_mask().sum())
        new_points = self._read_pointcloud_world()
        if new_points.size:
            self._accumulated_points = np.concatenate([self._accumulated_points, new_points], axis=0)
        self.coverage.set_from_points(self._accumulated_points, self.potato_center)
        new_filled_count = int(self.coverage.filled_mask().sum())

        self.views_taken += 1
        self.last_direction = direction

        reward = scan_policy_spec.step_reward(prev_filled_count, new_filled_count, was_filled_before)

        coverage_ratio = self.coverage.coverage_ratio()
        resolved_ratio = self.coverage.resolved_ratio()
        terminated = bool(coverage_ratio >= self.coverage_threshold or resolved_ratio >= 1.0)
        truncated = bool(self.views_taken >= self.max_views)
        if terminated or truncated:
            reward += scan_policy_spec.terminal_bonus(
                coverage_ratio, resolved_ratio, self.views_taken, self.max_views)

        obs = scan_policy_spec.build_observation(
            self.coverage, self.last_direction, self.views_taken, self.max_views)
        info = {"coverage_ratio": coverage_ratio, "resolved_ratio": resolved_ratio,
                "views_taken": self.views_taken}

        # Console progress readout -- nothing in this env is otherwise
        # visible (no RViz/MarkerArray, that's the separate ROS2 deployment
        # path in scan_controller.py): print coverage after every view so
        # progress is visible alongside the GUI while watching training.
        print(f"  view {self.views_taken}/{self.max_views}: "
              f"coverage={coverage_ratio * 100:.1f}% resolved={resolved_ratio * 100:.1f}% "
              f"reward={reward:+.2f} (points={len(new_points)}, "
              f"reach_err={final_pos_err * 1000:.0f}mm)")
        if terminated or truncated:
            print(f"episode done -- {'SOLVED' if terminated else 'max views reached'}: "
                  f"{self.views_taken} views, coverage={coverage_ratio * 100:.1f}%, "
                  f"resolved={resolved_ratio * 100:.1f}%")

        return obs, reward, terminated, truncated, info

    def _move_and_settle(self, target_pos, target_rotvec, pos_tol_m=0.005, rot_tol_deg=3.0):
        """Drives RMPflow toward (target_pos, target_rotvec) and steps
        physics/render until the tool0 link settles within tolerance or
        move_timeout_s elapses -- same tolerance-polling contract
        IsaacSimRobotInterface.move_to_pose uses over ROS2/TF, just done
        with direct in-process prim reads since we're already inside the
        sim process. A few extra settle_steps afterward mirror
        scan_controller's settle_time_s (letting the pointcloud/physics
        state stabilize before the caller reads it)."""
        target_pos = np.asarray(target_pos, dtype=float)
        target_rot = Rot.from_rotvec(np.asarray(target_rotvec, dtype=float))
        target_quat_wxyz = target_rot.as_quat()[[3, 0, 1, 2]]
        rot_tol_rad = np.radians(rot_tol_deg)

        t0 = time.time()
        while time.time() - t0 < self.move_timeout_s:
            self.rmpflow.set_end_effector_target(target_pos, target_quat_wxyz)
            self.rmpflow.update_world()
            action = self.articulation_policy.get_next_articulation_action(self.physics_dt)
            self.robot.apply_action(action)
            # render=True even headless -- the replicator pointcloud
            # annotator needs a rendered frame, GUI window or not.
            self.world.step(render=True)

            cur_pos, cur_quat = prim_world_pose(self.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
            cur_rotvec = Rot.from_quat(cur_quat).as_rotvec()
            pos_err = float(np.linalg.norm(cur_pos - target_pos))
            rot_err = (Rot.from_rotvec(cur_rotvec).inv() * target_rot).magnitude()
            if pos_err <= pos_tol_m and rot_err <= rot_tol_rad:
                break

        for _ in range(self.settle_steps):
            self.world.step(render=True)
        return pos_err

    def _read_pointcloud_world(self):
        # ADJUST: same caveat as isaac_scene.py's publish_cloud -- assumes
        # the "pointcloud" annotator returns world-space points on this
        # Isaac Sim version.
        data = self.pointcloud_annotator.get_data()
        return np.asarray(data.get("data", []), dtype=np.float32).reshape(-1, 3)

    def close(self):
        self.simulation_app.close()
