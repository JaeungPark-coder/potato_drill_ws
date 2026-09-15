"""Standalone Isaac Sim scene for the potato_scan pipeline.

NOT a ROS2 node run via `ros2 run` -- this needs Isaac Sim's own Kit
Python runtime (for the physics/rendering engine), so it's launched
directly with Isaac Sim's bundled interpreter:

    <isaac-sim-install-dir>/python.sh /path/to/isaac_scene.py

Written and reasoned about against Isaac Sim 5.1 (the `isaacsim.*`
extension/module namespace introduced in the 4.5+ restructuring) and
ROS2 Humble, WITHOUT the ability to actually run Isaac Sim in this
environment -- so treat this as a solid first draft, not a verified
working script. The handful of spots most likely to need adjustment on
your machine are marked "ADJUST:" below (asset path, robot link names,
RMPflow robot policy name). Isaac Sim's own bundled example scenes
(Window -> Examples -> ROS2 in the Isaac Sim GUI) are the fastest way to
confirm exact extension/API names if something here doesn't import.

Design choice: almost everything ROS2-facing here (publishing the point
cloud, TF, and drill-tip wrench; subscribing to the Cartesian target
pose) is done with PLAIN rclpy rather than Isaac Sim's OmniGraph ROS2
bridge nodes. OmniGraph node type names are the part of the Isaac Sim
API that changes most between versions; raw rclpy/tf2/pxr code is far
more stable and, since this file already needs Isaac-specific APIs for
the physics/rendering side regardless, keeping the ROS2 side plain
Python minimizes the surface area that can silently break on upgrade.

What this scene provides to the rest of the potato_scan pipeline (run
normally via `ros2 launch potato_scan potato_drill.launch.py
robot_backend:=isaac_sim`, in a separate terminal from this script):
  - /camera/depth/color/points (sensor_msgs/PointCloud2) -- what
    pointcloud_accumulator.py already expects by default
  - TF: base_link -> camera_link, base_link -> tool0
  - /isaac_sim/drill_tip/wrench (geometry_msgs/WrenchStamped) -- what
    isaac_robot_interface.IsaacSimRobotInterface reads for force_drill
  - subscribes /isaac_sim/cartesian_target (geometry_msgs/PoseStamped)
    and drives the arm there via RMPflow

The potato itself is a procedurally generated bumpy mesh (see
isaac_sim_common.make_potato_mesh) with a few random deeper pits standing
in for eyes -- not a photoreal asset, just enough irregularity to exercise
the shape-agnostic coverage grid and the concavity-based eye detector with
a different "potato" every run (seed below). The same mesh/robot/sensor
setup is reused (not reimplemented) by the RL training envs in
rl_scan_train_env.py / rl_drill_train_env.py, via isaac_sim_common.py.
"""
import os
import sys
import time

import numpy as np

from isaacsim import SimulationApp

# Same toggle as vla_ur5e_ws/isaac/*.py: default headless (works over SSH /
# no X session), set ISAAC_PICK_PLACE_HEADLESS=0 for an interactive window.
HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
import carb
import omni.replicator.core as rep
from pxr import UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.prims import SingleArticulation

enable_extension("isaacsim.ros2.bridge")  # makes rclpy importable/usable in this process

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import (
    Pose, PoseArray, PoseStamped, WrenchStamped, TransformStamped)
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
import tf2_ros
from scipy.spatial.transform import Rotation as Rot

# The single definition of the +Z-is-outward-normal convention; the detector
# publishes eyes in it too, which is what makes the two PoseArrays comparable
# without either side restating the rule.
#
# The path insert is required, not tidiness. This script runs under Isaac
# Sim's own interpreter and, per the README, deliberately WITHOUT sourcing the
# ROS 2 workspace -- so the installed potato_scan package is not importable,
# and a plain `python isaac/isaac_scene.py` puts only isaac/ on sys.path, not
# its parent. Without this the import fails before the scene is built at all.
# Same idiom collect_rlds_episodes.py uses to share EULER_SEQ across trees.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from potato_scan.drill_task_planner import normal_rotation  # noqa: E402

# Scene-building helpers (potato mesh, drill tip, contact sensor, RMPflow
# setup, world-pose readout) live in isaac_sim_common.py, shared with the
# two RL training envs (rl_scan_train_env.py, rl_drill_train_env.py) so
# training never simulates a different potato/robot/sensor model than this
# deployment script actually runs.
from isaac_sim_common import (
    UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH, TOOL_LINK_PRIM_PATH,
    make_potato_mesh, add_drill_tip, ContactForceReader, setup_rmpflow, prim_world_pose,
    generate_potato_geometry,
)

# ---------------------------------------------------------------------------
# Config -- match these to config/params.yaml (potato_center, base_frame, ...)
# ---------------------------------------------------------------------------
BASE_FRAME = "base_link"
CAMERA_FRAME = "camera_link"
TCP_FRAME = "tool0"
POTATO_CENTER = np.array([0.50, 0.00, 0.15])   # world == base_link here (robot at world origin)
# None -> a fresh random potato each run. Whichever seed is used is printed
# at startup as `potato seed N`: `python -m potato_scan.sim_detection_check
# --seed N` then rebuilds this exact potato (same draws, same vertices) with
# no Isaac Sim, which is how a detection this scene cannot explain gets
# checked against the geometry it was made on. Set it here to re-run one.
POTATO_SEED = None
CAMERA_TOPIC = "/camera/depth/color/points"
TARGET_POSE_TOPIC = "/isaac_sim/cartesian_target"
WRENCH_TOPIC = "/isaac_sim/drill_tip/wrench"
GROUND_TRUTH_EYES_TOPIC = "/potato_scan/ground_truth_eyes"
CLOUD_PUBLISH_PERIOD_S = 1.0


class IsaacSceneBridge(Node):
    """Plain ROS2 node living inside the Isaac Sim process: owns the
    publishers/subscriber the rest of the pipeline talks to. Kept
    separate from the physics-callback logic in main() for clarity."""

    def __init__(self):
        super().__init__('isaac_scene_bridge')
        self.cloud_pub = self.create_publisher(PointCloud2, CAMERA_TOPIC, 10)
        self.wrench_pub = self.create_publisher(WrenchStamped, WRENCH_TOPIC, 10)
        # Latched: the potato is carved once at startup, long before
        # eye_detector or detection_accuracy_check are up. Without
        # TRANSIENT_LOCAL the one message would be sent to nobody.
        self.truth_pub = self.create_publisher(
            PoseArray, GROUND_TRUTH_EYES_TOPIC,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.latest_target_pos = POTATO_CENTER.copy()
        self.latest_target_rotvec = np.array([0.0, np.pi, 0.0])  # pointing down, arbitrary default
        self.create_subscription(PoseStamped, TARGET_POSE_TOPIC, self._on_target, 10)

    def publish_ground_truth_eyes(self, eye_points, eye_normals):
        """Where the pits actually are, straight from the mesh that carved them.

        make_potato_mesh has always returned this and isaac_scene has always
        discarded it, which left "is a detected eye in the right place?"
        answerable only with a real robot and callipers (bring-up step 5).
        It is answerable here, now, for free -- and unlike the callipers it
        also gives the true outward NORMAL, which is what the 84-degree
        failure of 2026-09-14 turned on.

        "Where the pits actually are" was itself wrong until 2026-09-15:
        the position came out as base_radius - eye_depth along the pit's
        direction, ignoring the bumps, which put it a median 6mm inside
        the surface and further than the 8mm matching tolerance for 37%
        of eyes -- each of which then scored as a miss plus a spurious
        detection no matter what the detector did. It is now the point
        where the pit meets the real surface, bumps included (see
        procedural_potato.generate).
        """
        msg = PoseArray()
        msg.header.frame_id = BASE_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        for position, normal in zip(eye_points, eye_normals):
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in position)
            # same +Z-is-the-outward-normal convention eye_detector publishes,
            # from the one definition of it, so the two are comparable at all
            q = Rot.from_matrix(normal_rotation(normal)).as_quat()   # x, y, z, w
            pose.orientation.x, pose.orientation.y = float(q[0]), float(q[1])
            pose.orientation.z, pose.orientation.w = float(q[2]), float(q[3])
            msg.poses.append(pose)
        self.truth_pub.publish(msg)
        self.get_logger().info(
            f'published {len(msg.poses)} ground-truth eyes on {GROUND_TRUTH_EYES_TOPIC}')

    def _on_target(self, msg: PoseStamped):
        self.latest_target_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = msg.pose.orientation
        self.latest_target_rotvec = Rot.from_quat([q.x, q.y, q.z, q.w]).as_rotvec()

    def broadcast_tf(self, translation, quat_xyzw, child_frame):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = BASE_FRAME
        t.child_frame_id = child_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (
            float(translation[0]), float(translation[1]), float(translation[2]))
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (
            float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]), float(quat_xyzw[3]))
        self.tf_broadcaster.sendTransform(t)

    def publish_wrench(self, force_xyz):
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = TCP_FRAME
        msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = (
            float(force_xyz[0]), float(force_xyz[1]), float(force_xyz[2]))
        self.wrench_pub.publish(msg)

    def publish_cloud(self, points_xyz):
        # ADJUST: Replicator's "pointcloud" annotator's output frame has
        # varied across Isaac Sim versions -- assuming world-space points
        # here (world == base_link in this scene, robot at the origin), so
        # frame_id=BASE_FRAME makes pointcloud_accumulator's TF lookup a
        # trivial identity, sidestepping any camera-TF-accuracy error
        # entirely. If the scanned cloud looks offset/warped in RViz, the
        # annotator may instead be returning camera-local points -- switch
        # frame_id to CAMERA_FRAME (the broadcast TF for it is already set
        # up below) if so.
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = BASE_FRAME
        msg = pc2.create_cloud_xyz32(header, points_xyz.astype(np.float32))
        self.cloud_pub.publish(msg)


def main():
    assets_root = get_assets_root_path()
    if assets_root is None:
        carb.log_error("Could not resolve Isaac Sim assets root -- check Nucleus connection.")
        simulation_app.close()
        sys.exit(1)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    stage = get_current_stage()
    add_reference_to_stage(assets_root + UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH)
    # SingleArticulation (unbatched), not the vectorized Articulation --
    # RmpFlow's ArticulationMotionPolicy needs get_articulation_controller(),
    # which only SingleArticulation provides (same fix the two RL envs carry).
    robot = SingleArticulation(ROBOT_PRIM_PATH, name="ur5e_arm")
    world.reset()  # initializes physics handles for the articulation
    robot.initialize()

    potato = generate_potato_geometry(POTATO_CENTER, seed=POTATO_SEED)
    make_potato_mesh(stage, "/World/potato", POTATO_CENTER, geometry=potato)
    eye_points, eye_normals = potato.eye_points, potato.eye_normals
    # The seed is what makes this potato reproducible off-line, and the
    # bumps are the geometry the 2026-09-15 spurious detections were
    # suspected of sitting on -- neither was visible anywhere before.
    print(f"potato seed {potato.seed}: {len(eye_points)} eyes "
          f"(depth {potato.eye_depth * 1000:.2f}mm, sigma {np.degrees(potato.eye_sigma):.1f}deg), "
          f"{len(potato.bump_dirs)} bumps", flush=True)
    for i, (bd, amp, w) in enumerate(zip(potato.bump_dirs, potato.bump_amp, potato.bump_width)):
        print(f"  bump {i}: dir {np.round(bd, 3)} amp {amp * 1000:.1f}mm width {w:.2f}", flush=True)

    tool_prim = stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH)
    if not tool_prim.IsValid():
        carb.log_warn(f"{TOOL_LINK_PRIM_PATH} not found -- check the robot's actual link names "
                       f"in the Stage window and update TOOL_LINK_PRIM_PATH.")

    rmpflow, articulation_policy = setup_rmpflow(robot)

    # RMPflow drives the frame its config names ("tool0"); the camera and
    # drill tip have to hang off a real USD prim (TOOL_LINK_PRIM_PATH).
    # MEASURED (2026-09-07): same position, but orientations differ by about
    # (-90, -90, 0) degrees on this asset -- see rl_scan_train_env.py, where
    # mounting straight onto the flange left the point cloud empty on every
    # view. Measured here rather than hard-coded.
    q0 = np.asarray(robot.get_joint_positions())[:6]
    _, tool0_rot = rmpflow.get_end_effector_pose(q0)
    tool0_rot = np.asarray(tool0_rot)
    r_tool0 = (Rot.from_matrix(tool0_rot) if tool0_rot.shape == (3, 3)
               else Rot.from_quat(tool0_rot[[1, 2, 3, 0]]))
    _, flange_quat = prim_world_pose(stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
    r_flange_to_tool0 = Rot.from_quat(flange_quat).inv() * r_tool0

    drill_tip_path = add_drill_tip(stage, TOOL_LINK_PRIM_PATH,
                                    r_parent_to_tool=r_flange_to_tool0)

    camera_path = f"{TOOL_LINK_PRIM_PATH}/camera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    # ADJUST: this offset is a placeholder for the eye-in-hand mount --
    # replace with your real hand-eye calibration translation/rotation once
    # measured (see handeye_calibration.py for the real-hardware equivalent).
    # It is authored in the flange frame, so the tool-frame offset is rotated.
    camera.AddTranslateOp().Set(
        Gf.Vec3d(*r_flange_to_tool0.apply(np.array([0.0, -0.05, 0.05]))))
    # A USD camera images along local -Z while pose_utils.look_at_rotation
    # puts the optical axis on +Z, so the mount carries a 180-degree flip
    # about X on top of the frame correction above.
    _q_cam = (r_flange_to_tool0 * Rot.from_euler("x", 180.0, degrees=True)).as_quat()  # xyzw
    camera.AddOrientOp().Set(
        Gf.Quatf(float(_q_cam[3]), float(_q_cam[0]), float(_q_cam[1]), float(_q_cam[2])))
    # See rl_scan_train_env.py's matching comment: a USD camera's default
    # near clipping plane is 1.0m, which silently clips away anything this
    # eye-in-hand camera is actually scanning (confirmed by measurement).
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
    render_product = rep.create.render_product(camera_path, (640, 480))
    # See rl_scan_train_env.py's matching comment -- includeUnlabelled=True
    # is required or this annotator silently returns zero points for any
    # prim without an explicit semantic label (confirmed against an actual
    # Isaac Sim run).
    pointcloud_annotator = rep.AnnotatorRegistry.get_annotator(
        "pointcloud", init_params={"includeUnlabelled": True})
    pointcloud_annotator.attach([render_product])

    contact_reader = ContactForceReader(drill_tip_path)
    # CONFIRMED 2026-09-14 (drill_contact_probe.py): without this, the
    # sensor's readings never leave their constructor default
    # ({"time": 0, "physics_step": 0}) for the rest of the run, no matter
    # how hard/long the tip presses into the potato -- the underlying
    # get_sensor_reading() stays is_valid=False forever. PhysX parses
    # contact-report registrations (including the one
    # IsaacSensorCreateContactSensor already applies to drill_tip) when the
    # physics scene (re)starts; the sensor prim was created well after the
    # FIRST world.reset() above, so PhysX never saw it. A second reset here,
    # after every physics-relevant prim for this scene already exists, is
    # what actually binds it -- re-verified by direct query against
    # isaacsim.sensors.physics._sensor's own interface, bypassing this
    # project's ContactForceReader wrapper entirely, so this isn't a
    # wrapper-level bug. re-initialize the articulation for the same reason
    # __init__'s own world.reset() above needed a paired robot.initialize().
    world.reset()
    robot.initialize()

    rclpy.init()
    bridge = IsaacSceneBridge()
    # Latched, so it is published once here and still reaches eye_detector
    # or detection_accuracy_check whenever they start.
    bridge.publish_ground_truth_eyes(eye_points, eye_normals)

    last_cloud_time = 0.0
    physics_dt = 1.0 / 60.0

    print("isaac_scene.py running -- Ctrl+C in this terminal (or close the Isaac Sim "
          "window) to stop. Launch the rest of the pipeline in another terminal with "
          "robot_backend:=isaac_sim.")

    try:
        while simulation_app.is_running():
            world.step(render=True)
            rclpy.spin_once(bridge, timeout_sec=0.0)

            rmpflow.set_end_effector_target(
                bridge.latest_target_pos,
                Rot.from_rotvec(bridge.latest_target_rotvec).as_quat()[[3, 0, 1, 2]])  # wxyz for RmpFlow
            rmpflow.update_world()
            action = articulation_policy.get_next_articulation_action(physics_dt)
            robot.apply_action(action)

            tool_pos, tool_quat = prim_world_pose(stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
            bridge.broadcast_tf(tool_pos, tool_quat, TCP_FRAME)
            cam_pos, cam_quat = prim_world_pose(stage.GetPrimAtPath(camera_path))
            bridge.broadcast_tf(cam_pos, cam_quat, CAMERA_FRAME)

            bridge.publish_wrench(contact_reader.read())

            now = time.time()
            if now - last_cloud_time >= CLOUD_PUBLISH_PERIOD_S:
                last_cloud_time = now
                data = pointcloud_annotator.get_data()
                pts = np.asarray(data.get("data", []), dtype=np.float32).reshape(-1, 3)
                if len(pts) > 0:
                    bridge.publish_cloud(pts)
    except (KeyboardInterrupt, Exception) as e:
        # A plain SIGTERM (e.g. from `timeout`, or a supervisor stopping this
        # process) is not a KeyboardInterrupt -- CONFIRMED 2026-09-14: rclpy
        # installs its own SIGTERM handler that calls context.shutdown() from
        # inside the signal handler, racing whatever line happens to be
        # executing. That left rclpy.spin_once() mid-call raising RCLError
        # ("the given context is not valid"), uncaught here before this fix,
        # which skipped every cleanup step below and crashed the process
        # (native SIGSEGV in omni.graph.core.plugin/omni.syntheticdata.plugin
        # during Py_FinalizeEx) rather than exiting cleanly.
        if not isinstance(e, KeyboardInterrupt):
            print(f"isaac_scene.py: shutting down after {e!r}", flush=True)
    finally:
        # Each step guarded independently: rclpy.shutdown() raising (context
        # already shut down by the race above) must not skip the Replicator
        # cleanup below it -- CONFIRMED 2026-09-14: leaving the pointcloud
        # annotator attached and the render product alive when
        # simulation_app.close() tears down the extension stack is what
        # produced the native crash, not the rclpy exception itself.
        for cleanup in (bridge.destroy_node, rclpy.shutdown,
                        pointcloud_annotator.detach, render_product.destroy):
            try:
                cleanup()
            except Exception as cleanup_error:
                print(f"isaac_scene.py: cleanup step {cleanup!r} failed: {cleanup_error!r}",
                      flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
