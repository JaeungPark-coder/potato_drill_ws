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
make_potato_mesh below) with a few random deeper pits standing in for
eyes -- not a photoreal asset, just enough irregularity to exercise the
shape-agnostic coverage grid and the concavity-based eye detector with a
different "potato" every run (seed below).

--- REVIEW FIXES APPLIED (see inline "FIX:" comments) ---
1. make_potato_mesh: the collision now explicitly uses a non-convex
   triangle-mesh approximation. Without this, PhysX's default convex-hull
   approximation would fill in the concave eye pits, so the drill tip
   would never actually be able to contact the "inside" of an eye even
   though the mesh looks visually correct.
2. add_drill_tip: dropped RigidBodyAPI. The drill tip is a child prim of
   tool0 (an articulation link); giving it its own RigidBodyAPI made it a
   physically independent free body instead of a fixed attachment to the
   link, with nothing (no fixed joint) actually welding it to the arm.
   Collision alone is enough for the tip to inherit tool0's rigid-body
   motion and still register PhysX contacts.
"""
import sys
import time

import numpy as np

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
import carb
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdPhysics, Gf

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

# ADJUST: relative path under the Isaac asset root for the UR5e USD. If this
# raises, open the Isaac Sim Asset Browser (Window -> Browsers -> Assets) and
# search "ur5e" to find the correct path for your install, then paste it here.
UR5E_ASSET_RELATIVE_PATH = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"

# ADJUST: exact prim names inside the loaded UR5e USD. Check the Stage window
# after the robot loads if TF/camera/drill placement looks wrong.
ROBOT_PRIM_PATH = "/World/ur5e"
TOOL_LINK_PRIM_PATH = "/World/ur5e/tool0"


def make_potato_mesh(stage, prim_path, center, base_radius=0.035, bumpiness=0.35,
                      n_lat=24, n_lon=36, seed=None):
    """Procedural bumpy-blob mesh standing in for a real potato: a
    perturbed sphere with a handful of random low-frequency bumps
    (irregular overall shape -- exercises the shape-agnostic coverage
    grid) plus a few sharper inward pits (stand-in eyes -- exercises
    eye_detector's concavity clustering). Re-seed for a different
    "potato" each run."""
    rng = np.random.default_rng(seed)
    lats = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lons = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)

    n_bumps = int(rng.integers(4, 7))
    bump_dirs = rng.normal(size=(n_bumps, 3))
    bump_dirs /= np.linalg.norm(bump_dirs, axis=1, keepdims=True)
    bump_amp = rng.uniform(0.1, 1.0, size=n_bumps) * bumpiness * base_radius
    bump_width = rng.uniform(0.4, 1.0, size=n_bumps)

    n_eyes = int(rng.integers(3, 8))
    eye_dirs = rng.normal(size=(n_eyes, 3))
    eye_dirs /= np.linalg.norm(eye_dirs, axis=1, keepdims=True)
    eye_depth = rng.uniform(0.15, 0.3) * base_radius

    points = []
    for lat in lats:
        for lon in lons:
            d = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
            r = base_radius
            for bd, amp, w in zip(bump_dirs, bump_amp, bump_width):
                r += amp * max(0.0, float(np.dot(d, bd))) ** (1.0 / w)
            for ed in eye_dirs:
                dot = float(np.dot(d, ed))
                if dot > 0.85:
                    r -= eye_depth * (dot - 0.85) / 0.15
            points.append(center + d * r)
    points = np.array(points)

    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            a, b = i * n_lon + j, i * n_lon + j2
            c, dd = (i + 1) * n_lon + j2, (i + 1) * n_lon + j
            faces.append((a, b, c, dd))

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([4] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([idx for f in faces for idx in f])
    mesh.CreateSubdivisionSchemeAttr("none")

    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)  # static collider so the drill tip can contact it

    # FIX 1: without this, PhysX's default convex-hull approximation would
    # fill in the concave eye pits (eye_dirs above), so the drill tip could
    # never actually reach "inside" an eye even though the mesh looks right
    # visually. The potato is a static (non-articulated, non-rigid-body)
    # collider, so a full non-convex triangle mesh approximation is allowed
    # and keeps the pits geometrically real for contact purposes.
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision_api.CreateApproximationAttr("none")

    return mesh


def add_drill_tip(stage, parent_path, prim_path="drill_tip", length=0.02, radius=0.0015):
    """A small collider rigidly attached to the tool link, standing in
    for the physical drill bit -- gives PhysX something to report
    contact force on during force_drill's insertion.

    FIX 2: previously this also applied RigidBodyAPI to the tip prim.
    Since drill_tip is a child prim of tool0 (an articulation link), giving
    it its own RigidBodyAPI made PhysX treat it as an independent free body
    rather than a fixed attachment -- with no fixed joint actually welding
    it to the arm, it could visually follow the parent transform in the
    Stage hierarchy while being physically inconsistent (or not follow the
    arm's motion at all in the physics simulation). Collision alone is
    sufficient here: as a child prim under the tool0 link with no rigid-body
    API of its own, it's treated as rigidly fixed to tool0 and still
    registers PhysX contacts for the ContactSensor to read.
    """
    full_path = f"{parent_path}/{prim_path}"
    cyl = UsdGeom.Cylinder.Define(stage, full_path)
    cyl.CreateHeightAttr(length)
    cyl.CreateRadiusAttr(radius)
    cyl.AddTranslateOp().Set(Gf.Vec3d(0, 0, length / 2))  # extends out along tool +Z
    prim = cyl.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    return full_path


class ContactForceReader:
    """Reads contact force at the drill tip via Isaac Sim's higher-level
    ContactSensor wrapper (a scalar normal-force reading, like a physical
    force/bump sensor -- sufficient here since IsaacSimRobotInterface only
    needs a force MAGNITUDE for its max_force check, same as the real
    UR5e's getActualTCPForce() magnitude).

    ADJUST: if isaacsim.sensors.physics.ContactSensor doesn't construct on
    your install (namespace moved again, or the sensor prim schema needs a
    slightly different setup), this prints one warning and falls back to
    always reporting zero force. That's still enough to validate the
    scan/detect/visit-order/roll-search parts of the pipeline -- it just
    means force_drill will always run to max_depth in sim instead of ever
    genuinely exercising the max_force stop path, so don't mistake a clean
    sim drilling pass for having validated the force-limit safety logic.
    """

    def __init__(self, prim_path, radius=0.002):
        self._ok = False
        try:
            from isaacsim.sensors.physics import ContactSensor
            self._sensor = ContactSensor(
                prim_path=f"{prim_path}/contact_sensor",
                name="drill_tip_contact",
                frequency=60,
                translation=np.array([0.0, 0.0, 0.0]),
                min_threshold=0.0,
                max_threshold=1000.0,
                radius=radius,
            )
            self._ok = True
        except Exception as exc:  # noqa: BLE001 -- best-effort sim sensor setup
            carb.log_warn(
                f"ContactSensor setup failed ({exc}); drill_tip wrench will report zero force -- "
                f"force_drill will always reach max_depth in sim, never stop on max_force.")

    def read(self):
        if not self._ok:
            return np.zeros(3)
        frame = self._sensor.get_current_frame()
        force_mag = float(frame.get("force", 0.0) or 0.0)
        return np.array([force_mag, 0.0, 0.0])


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


def prim_world_pose(prim):
    xform = UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = mat.ExtractTranslation()
    quat = mat.ExtractRotationQuat()
    quat_xyzw = [quat.imaginary[0], quat.imaginary[1], quat.imaginary[2], quat.real]
    return np.array([translation[0], translation[1], translation[2]]), np.array(quat_xyzw)


def setup_rmpflow(robot_articulation):
    """ADJUST: the exact robot-name string load_supported_motion_policy_config
    expects for the UR5e may differ from "UR5e" on your install -- if this
    raises/returns None, call get_supported_robot_policy_pairs() (same module)
    to list the exact registered names and swap it in below."""
    from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
    from isaacsim.robot_motion.motion_generation.interface_config_loader import (
        load_supported_motion_policy_config)

    rmp_config = load_supported_motion_policy_config("UR5e", "RMPflow")
    rmpflow = RmpFlow(**rmp_config)
    physics_dt = 1.0 / 60.0
    return rmpflow, ArticulationMotionPolicy(robot_articulation, rmpflow, physics_dt)


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