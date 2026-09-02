"""Stage 3: visits each detected potato eye in an efficient order and
removes it with the UR5e + drill end-effector, using force_mode to feed
along the surface normal so the (initially unknown) exact insertion
depth is force-limited rather than blindly position-commanded.

Triggered by publishing True on /potato_scan/start_drilling once the
eye positions in RViz look correct.

SAFETY: max_force / max_depth / feed_force below are placeholders. They
MUST be tuned on the real setup (drill bit, potato size, UR5e safety
limits) before running unattended -- start with a low feed_force and a
conservative max_depth, and be ready to hit the pendant e-stop.

Different eyes on different potatoes end up needing very different
approach angles (whatever the local surface normal happens to be), and
not every one of those is guaranteed reachable -- some may sit past a
joint limit or through a wrist singularity for the default orientation.
Rather than just failing on those, this exploits the fact that a drill
bit is rotationally symmetric about its own insertion axis: rotation
about that axis (roll) doesn't change the actual drilling geometry at
all, so it's a free parameter that gets swept (ROLL_SEARCH_DEG) to find
an orientation the arm can actually reach before giving up on an eye.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseArray
from scipy.spatial.transform import Rotation as Rot

from potato_scan.robot_interface import UR5eInterface
from potato_scan.isaac_robot_interface import IsaacSimRobotInterface
from potato_scan.drill_task_planner import plan_visit_order, approach_pose

# Roll offsets (degrees, about the insertion axis) tried in order when the
# default approach orientation is unreachable. Smallest deviation from
# the default first, then wider swings.
ROLL_SEARCH_DEG = [0, 45, -45, 90, -90, 135, -135, 180]


def normal_rotation(normal):
    """Rotation whose +Z axis is `normal` -- must match eye_detector's
    normal_to_quat convention so orientations line up."""
    z = np.asarray(normal, dtype=float)
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


class DrillController(Node):
    def __init__(self):
        super().__init__('drill_controller')

        self.declare_parameter('robot_ip', '192.168.1.100')
        self.declare_parameter('robot_backend', 'rtde')  # 'rtde' or 'isaac_sim' -- see scan_controller.py
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tcp_frame', 'tool0')  # only used by isaac_sim backend
        self.declare_parameter('drill_output_pin', 0)
        self.declare_parameter('standoff', 0.03)
        self.declare_parameter('max_depth', 0.015)
        self.declare_parameter('feed_force', 15.0)
        self.declare_parameter('max_force', 40.0)
        self.declare_parameter('approach_speed', 0.2)
        self.declare_parameter('approach_acceleration', 0.5)
        self.declare_parameter('drill_timeout_s', 8.0)
        # 'heuristic' (default) = sweep ROLL_SEARCH_DEG until one orientation
        # is reachable. 'rl' = a trained policy (see potato_scan/rl/ and
        # isaac/train_drill_policy.py) picks the approach roll + a small
        # lateral offset instead -- requires rl_model_path.
        self.declare_parameter('approach_policy', 'heuristic')
        self.declare_parameter('rl_model_path', '')

        self.standoff = self.get_parameter('standoff').value
        self.max_depth = self.get_parameter('max_depth').value
        self.feed_force = self.get_parameter('feed_force').value
        self.max_force = self.get_parameter('max_force').value
        self.drill_timeout_s = self.get_parameter('drill_timeout_s').value

        approach_policy_mode = self.get_parameter('approach_policy').value
        if approach_policy_mode == 'rl':
            from potato_scan.rl.drill_policy_backend import RLApproachPolicy
            self.approach_policy = RLApproachPolicy(self.get_parameter('rl_model_path').value)
            self.get_logger().info(
                f"approach_policy=rl, loaded {self.get_parameter('rl_model_path').value}")
        else:
            self.approach_policy = None

        backend = self.get_parameter('robot_backend').value
        if backend == 'isaac_sim':
            self.robot = IsaacSimRobotInterface(
                self, base_frame=self.get_parameter('base_frame').value,
                tcp_frame=self.get_parameter('tcp_frame').value)
        else:
            self.robot = UR5eInterface(
                self.get_parameter('robot_ip').value,
                speed=self.get_parameter('approach_speed').value,
                acceleration=self.get_parameter('approach_acceleration').value,
                drill_output_pin=self.get_parameter('drill_output_pin').value)

        self._eyes = None  # list of (position, normal)
        self.create_subscription(PoseArray, '/potato_scan/eye_poses', self._on_eye_poses, 10)
        self.create_subscription(Bool, '/potato_scan/start_drilling', self._on_start, 10)
        self.status_pub = self.create_publisher(Bool, '/potato_scan/drilling_complete', 10)

    def _on_eye_poses(self, msg: PoseArray):
        eyes = []
        for pose in msg.poses:
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            normal = Rot.from_quat(quat).as_matrix()[:, 2]  # +Z axis = outward normal
            eyes.append((position, normal))
        self._eyes = eyes
        self.get_logger().info(f'received {len(eyes)} eye poses')

    def _on_start(self, msg: Bool):
        if not msg.data:
            return
        if not self._eyes:
            self.get_logger().warn('start_drilling received but no eye poses available')
            return
        self.run_drilling()

    def _find_reachable_approach(self, position, normal):
        """Try the default approach orientation, then sweep roll about the
        (rotationally symmetric) drill axis until one is actually
        reachable. Returns (approach_position, rotvec) of the pose the
        robot successfully moved to, or (None, None) if every roll offset
        was rejected. When approach_policy == 'rl', delegates to the
        trained policy instead (see potato_scan/rl/drill_policy_backend.py)."""
        if self.approach_policy is not None:
            return self.approach_policy.find_approach(self.robot, position, normal, self.standoff)

        approach = approach_pose(position, normal, self.standoff)
        base_rotation = normal_rotation(normal)
        for roll_deg in ROLL_SEARCH_DEG:
            roll = Rot.from_euler('z', roll_deg, degrees=True).as_matrix()
            rotvec = Rot.from_matrix(base_rotation @ roll).as_rotvec()
            if self.robot.move_to_pose(approach, rotvec):
                if roll_deg != 0:
                    self.get_logger().info(f'reached via roll offset {roll_deg}deg from default')
                return approach, rotvec
            self.get_logger().warn(f'approach unreachable at roll={roll_deg}deg, trying next offset')
        return None, None

    def run_drilling(self):
        positions = np.array([p for p, _ in self._eyes])
        start_pos, _ = self.robot.get_tcp_pose()
        order = plan_visit_order(positions, start_position=start_pos)
        self.get_logger().info(f'visiting {len(order)} eyes in order {order}')

        for count, idx in enumerate(order):
            position, normal = self._eyes[idx]

            self.get_logger().info(f'[{count + 1}/{len(order)}] approaching eye {idx} at {position}')
            approach, rotvec = self._find_reachable_approach(position, normal)
            if approach is None:
                self.get_logger().error(
                    f'eye {idx}: no reachable approach angle found after full roll search -- '
                    'skipping (check reachability / fixture placement / potato_center)')
                continue

            task_frame = list(approach) + list(rotvec)
            self.robot.drill_on()
            reached = self.robot.force_drill(
                task_frame, axis_index=2,
                feed_force=self.feed_force, max_force=self.max_force,
                max_depth=self.max_depth, timeout_s=self.drill_timeout_s)
            if not reached:
                self.get_logger().warn(f'eye {idx}: force limit hit before target depth (check tuning)')

            self.robot.move_to_pose(approach, rotvec)  # retract
            self.robot.drill_off()

        self.get_logger().info('drilling pass complete')
        self.status_pub.publish(Bool(data=True))


def main():
    rclpy.init()
    node = DrillController()
    try:
        rclpy.spin(node)
    finally:
        node.robot.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
