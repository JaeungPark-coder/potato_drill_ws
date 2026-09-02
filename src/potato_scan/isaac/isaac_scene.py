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
import sys
import time

import numpy as np

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
import carb
import omni.replicator.core as rep
from pxr import UsdGeom, Gf

from isaacsim.core.api import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.prims import Articulation

enable_extension("isaacsim.ros2.bridge")  # makes rclpy importable/usable in this process

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, WrenchStamped, TransformStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
import tf2_ros
from scipy.spatial.transform import Rotation as Rot

# Scene-building helpers (potato mesh, drill tip, contact sensor, RMPflow
# setup, world-pose readout) live in isaac_sim_common.py, shared with the
# two RL training envs (rl_scan_train_env.py, rl_drill_train_env.py) so
# training never simulates a different potato/robot/sensor model than this
# deployment script actually runs.
from isaac_sim_common import (
    UR5E_ASSET_RELATIVE_PATH, ROBOT_PRIM_PATH, TOOL_LINK_PRIM_PATH,
    make_potato_mesh, add_drill_tip, ContactForceReader, setup_rmpflow, prim_world_pose,
)

# ---------------------------------------------------------------------------
# Config -- match these to config/params.yaml (potato_center, base_frame, ...)
# ---------------------------------------------------------------------------
BASE_FRAME = "base_link"
CAMERA_FRAME = "camera_link"
TCP_FRAME = "tool0"
POTATO_CENTER = np.array([0.50, 0.00, 0.15])   # world == base_link here (robot at world origin)
POTATO_SEED = None                             # None -> random potato shape each run
CAMERA_TOPIC = "/camera/depth/color/points"
TARGET_POSE_TOPIC = "/isaac_sim/cartesian_target"
WRENCH_TOPIC = "/isaac_sim/drill_tip/wrench"
CLOUD_PUBLISH_PERIOD_S = 1.0


class IsaacSceneBridge(Node):
    """Plain ROS2 node living inside the Isaac Sim process: owns the
    publishers/subscriber the rest of the pipeline talks to. Kept
    separate from the physics-callback logic in main() for clarity."""

    def __init__(self):
        super().__init__('isaac_scene_bridge')
        self.cloud_pub = self.create_publisher(PointCloud2, CAMERA_TOPIC, 10)
        self.wrench_pub = self.create_publisher(WrenchStamped, WRENCH_TOPIC, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.latest_target_pos = POTATO_CENTER.copy()
        self.latest_target_rotvec = np.array([0.0, np.pi, 0.0])  # pointing down, arbitrary default
        self.create_subscription(PoseStamped, TARGET_POSE_TOPIC, self._on_target, 10)

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
    robot = Articulation(ROBOT_PRIM_PATH)
    world.reset()  # initializes physics handles for the articulation

    make_potato_mesh(stage, "/World/potato", POTATO_CENTER, seed=POTATO_SEED)

    tool_prim = stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH)
    if not tool_prim.IsValid():
        carb.log_warn(f"{TOOL_LINK_PRIM_PATH} not found -- check the robot's actual link names "
                       f"in the Stage window and update TOOL_LINK_PRIM_PATH.")
    drill_tip_path = add_drill_tip(stage, TOOL_LINK_PRIM_PATH)

    camera_path = f"{TOOL_LINK_PRIM_PATH}/camera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    # ADJUST: this offset is a placeholder for the eye-in-hand mount --
    # replace with your real hand-eye calibration translation/rotation once
    # measured (see handeye_calibration.py for the real-hardware equivalent).
    camera.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.05, 0.05))
    render_product = rep.create.render_product(camera_path, (640, 480))
    pointcloud_annotator = rep.AnnotatorRegistry.get_annotator("pointcloud")
    pointcloud_annotator.attach([render_product])

    rmpflow, articulation_policy = setup_rmpflow(robot)
    contact_reader = ContactForceReader(drill_tip_path)

    rclpy.init()
    bridge = IsaacSceneBridge()

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
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
