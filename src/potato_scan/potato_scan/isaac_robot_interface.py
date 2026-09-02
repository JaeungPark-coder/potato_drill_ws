"""ROS2-facing robot backend for Isaac Sim, matching robot_interface.
UR5eInterface's method surface (move_to_pose, get_tcp_pose, drill_on/off,
force_drill, stop, close) so scan_controller.py and drill_controller.py
run against either backend unchanged -- only which class gets
constructed differs (see the `robot_backend` parameter added to both).

Isaac Sim doesn't speak the UR RTDE protocol that UR5eInterface uses, so
this talks to isaac/isaac_scene.py (a separate standalone script run
inside Isaac Sim itself) over plain ROS2 topics instead:

  publish  geometry_msgs/PoseStamped  -> cartesian_target_topic
           (the sim's RMPflow controller tracks this as the TCP target,
           the same "give me a pose, IK figures out the joints" contract
           moveL has on the real controller)
  subscribe geometry_msgs/WrenchStamped <- wrench_topic
           (simulated force at the drill tip, from a PhysX ContactSensor
           in the scene)
  TF        base_frame -> tcp_frame
           (read via tf2, exactly like pointcloud_accumulator.py already
           does for the camera -- the sim script broadcasts this TF)

force_drill here is a simplified approximation, NOT true admittance
control: it steps the target pose along the insertion axis a small
increment at a time and stops on max_force (from the simulated wrench)
or max_depth, matching the real force_drill's external contract. It
isn't compliant on the other 5 axes the way UR's real force_mode is --
fine for validating the scan/detect/visit-order/roll-search pipeline in
sim, not a substitute for tuning real insertion dynamics.
"""
import time
import numpy as np
import rclpy
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped, WrenchStamped
from std_msgs.msg import Bool
import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation as Rot


class IsaacSimRobotInterface:
    def __init__(self, node, base_frame='base_link', tcp_frame='tool0',
                 target_pose_topic='/isaac_sim/cartesian_target',
                 wrench_topic='/isaac_sim/drill_tip/wrench',
                 drill_state_topic='/isaac_sim/drill_state',
                 pose_reach_tolerance_m=0.005, pose_reach_tolerance_deg=3.0,
                 settle_timeout_s=6.0):
        self.node = node
        self.base_frame = base_frame
        self.tcp_frame = tcp_frame
        self.pose_reach_tolerance_m = pose_reach_tolerance_m
        self.pose_reach_tolerance_rad = np.radians(pose_reach_tolerance_deg)
        self.settle_timeout_s = settle_timeout_s

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, node)

        self.target_pub = node.create_publisher(PoseStamped, target_pose_topic, 10)
        self.drill_state_pub = node.create_publisher(Bool, drill_state_topic, 10)

        self._latest_wrench = None
        node.create_subscription(WrenchStamped, wrench_topic, self._on_wrench, 10)

    def _on_wrench(self, msg: WrenchStamped):
        self._latest_wrench = msg

    def _current_force_mag(self):
        if self._latest_wrench is None:
            return None
        w = self._latest_wrench.wrench.force
        return float(np.linalg.norm([w.x, w.y, w.z]))

    def _publish_target(self, position, rotvec):
        quat = Rot.from_rotvec(np.asarray(rotvec, dtype=float)).as_quat()
        msg = PoseStamped()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        position = np.asarray(position, dtype=float)
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = (
            float(position[0]), float(position[1]), float(position[2]))
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = (
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
        self.target_pub.publish(msg)

    def get_tcp_pose(self):
        """Returns (position, rotvec) read from TF, or (None, None) if the
        transform isn't available yet (e.g. sim not fully up)."""
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.tcp_frame, Time())
        except TransformException as ex:
            self.node.get_logger().warn(f'TF lookup {self.base_frame}->{self.tcp_frame} failed: {ex}',
                                         throttle_duration_sec=2.0)
            return None, None
        t = tf.transform.translation
        q = tf.transform.rotation
        position = np.array([t.x, t.y, t.z])
        rotvec = Rot.from_quat([q.x, q.y, q.z, q.w]).as_rotvec()
        return position, rotvec

    def move_to_pose(self, position, rotvec, speed=None, acceleration=None):
        """Publish the target pose and poll TF until the sim's RMPflow
        controller settles there (within tolerance) or settle_timeout_s
        elapses. `speed`/`acceleration` are accepted for interface
        compatibility with UR5eInterface but unused -- RMPflow's motion
        profile is configured in isaac_scene.py, not per-call.

        Returns False on timeout, standing in for RTDE's "target
        rejected as unreachable" signal: RMPflow is a reactive
        controller that doesn't hard-fail on an unreachable target, it
        just never gets there, so "didn't converge in time" is the
        practical equivalent drill_controller's roll search relies on.
        """
        self._publish_target(position, rotvec)
        target_pos = np.asarray(position, dtype=float)
        target_rot = Rot.from_rotvec(np.asarray(rotvec, dtype=float))

        t0 = time.time()
        while time.time() - t0 < self.settle_timeout_s:
            time.sleep(0.05)
            cur_pos, cur_rotvec = self.get_tcp_pose()
            if cur_pos is None:
                continue
            pos_err = float(np.linalg.norm(cur_pos - target_pos))
            rot_err = (Rot.from_rotvec(cur_rotvec).inv() * target_rot).magnitude()
            if pos_err <= self.pose_reach_tolerance_m and rot_err <= self.pose_reach_tolerance_rad:
                return True
        return False

    def drill_on(self):
        self.drill_state_pub.publish(Bool(data=True))

    def drill_off(self):
        self.drill_state_pub.publish(Bool(data=False))

    def force_drill(self, task_frame, axis_index=2, feed_force=15.0, max_force=40.0,
                     max_depth=0.015, timeout_s=8.0, poll_dt=0.05,
                     free_axis_speed_limit=0.05, held_axis_deviation_limit=0.005,
                     step_size_m=0.0005):
        """Feed along the negative direction of `axis_index` of
        `task_frame` in small position steps, reading simulated contact
        force each step. `feed_force`/`free_axis_speed_limit`/
        `held_axis_deviation_limit` are accepted for interface
        compatibility with UR5eInterface.force_drill (real force_mode
        parameters) but unused here -- see module docstring on why this
        is a simplified stand-in, not true admittance control.

        Returns True if max_depth was reached, False if stopped on
        max_force.
        """
        base_pos = np.array(task_frame[:3], dtype=float)
        base_rotvec = np.array(task_frame[3:], dtype=float)
        base_rot_matrix = Rot.from_rotvec(base_rotvec).as_matrix()
        insertion_axis = base_rot_matrix[:, axis_index]  # outward normal direction

        depth = 0.0
        reached = False
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            depth = min(depth + step_size_m, max_depth)
            target_pos = base_pos - insertion_axis * depth  # feed INTO the surface
            self._publish_target(target_pos, base_rotvec)
            time.sleep(poll_dt)

            force_mag = self._current_force_mag()
            if force_mag is not None and force_mag >= max_force:
                reached = False
                break
            if depth >= max_depth:
                reached = True
                break

        return reached

    def stop(self):
        pass  # no in-flight trajectory queue to cancel with a pose-target interface

    def close(self):
        pass
