"""Score eye_detector against the eyes isaac_scene says are really there.

    ros2 run potato_scan detection_accuracy_check

Then run the pipeline as usual. This node listens to both sides and prints a
report each time the detector publishes:

    /potato_scan/ground_truth_eyes   isaac_scene (latched, once at startup)
    /potato_scan/eye_poses           eye_detector

WHY THIS EXISTS, AND WHAT IT IS NOT

Bring-up step 5 measures the same thing with a real robot and callipers, and
still has to: this sees the simulator's own depth rendering, which has no
sensor noise, no specularity, and no dropout in the dark pocket an eye
actually is. Clearing the 2 mm target here is necessary and nowhere near
sufficient.

What it buys is that the question stops needing hardware to ask at all. The
detector produced six plausible-looking candidates on 2026-09-14 and two of
them could not be drilled -- and whether that was a position error, a normal
error, or a real eye the arm simply could not reach took a separate
investigation to establish. It was the normals: up to 84 degrees off. That
now comes out of one run, as a number, before anyone touches a potato.

Sim only. Nothing publishes ground truth on real hardware, and the node says
so rather than sitting silent if the topic never arrives.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as Rot

from potato_scan.detection_accuracy import format_report, match_detections

GROUND_TRUTH_TOPIC = '/potato_scan/ground_truth_eyes'
DETECTED_TOPIC = '/potato_scan/eye_poses'


def poses_to_arrays(msg):
    """PoseArray -> (positions (N,3), outward normals (N,3)).

    Both sides encode the normal as the pose's local +Z, which is the
    convention drill_task_planner.normal_rotation defines and eye_detector
    publishes in.
    """
    positions, normals = [], []
    for pose in msg.poses:
        positions.append([pose.position.x, pose.position.y, pose.position.z])
        q = pose.orientation
        normals.append(Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()[:, 2])
    return (np.asarray(positions, dtype=float).reshape(-1, 3),
            np.asarray(normals, dtype=float).reshape(-1, 3))


class DetectionAccuracyCheck(Node):
    def __init__(self):
        super().__init__('detection_accuracy_check')
        self.declare_parameter('match_tolerance', 0.008)
        self.declare_parameter('warn_after_s', 20.0)
        self.tolerance = self.get_parameter('match_tolerance').value

        self.truth = None
        # must match isaac_scene's publisher exactly, or the latched message
        # is never delivered and this looks like a scene that published nothing
        self.create_subscription(
            PoseArray, GROUND_TRUTH_TOPIC, self._on_truth,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(PoseArray, DETECTED_TOPIC, self._on_detected, 10)

        self._warned = False
        self.create_timer(self.get_parameter('warn_after_s').value, self._warn_if_silent)
        self.get_logger().info(
            f'waiting for {GROUND_TRUTH_TOPIC} (isaac_scene) and {DETECTED_TOPIC}')

    def _warn_if_silent(self):
        if self.truth is None and not self._warned:
            self._warned = True
            self.get_logger().warn(
                f'no ground truth on {GROUND_TRUTH_TOPIC} yet. It is published once, '
                f'latched, by isaac/isaac_scene.py -- so either the scene is not '
                f'running, it predates this feature, or ROS_DOMAIN_ID differs '
                f'between it and this node. There is no real-hardware equivalent: '
                f'on a real potato, bring-up step 5 (calliper_check) is the '
                f'measurement instead.')

    def _on_truth(self, msg):
        self.truth = poses_to_arrays(msg)
        self.get_logger().info(f'ground truth: {len(msg.poses)} eyes')

    def _on_detected(self, msg):
        if self.truth is None:
            self.get_logger().warn(
                f'{len(msg.poses)} eyes detected but no ground truth to score against')
            return

        truth_positions, truth_normals = self.truth
        positions, normals = poses_to_arrays(msg)
        result = match_detections(
            positions, truth_positions, tolerance=self.tolerance,
            detected_normals=normals, truth_normals=truth_normals)
        self.get_logger().info('\n' + format_report(result))


def main():
    rclpy.init()
    node = DetectionAccuracyCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
