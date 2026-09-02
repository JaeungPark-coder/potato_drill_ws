"""Step 3 of the recommended bring-up order: tune drill_controller's
force-mode safety parameters (feed_force / max_force / max_depth) on a
single point, without running the full scan -> detect -> visit-order
pipeline.

Usage: freedrive the UR5e (this script enables teach mode) so the tool
tip sits `standoff` back from a real potato surface, oriented with its
+Z axis pointing OUT of the surface (same convention as
eye_detector.normal_to_quat / drill_controller.normal_rotation). Press
Enter to run one probe with the current feed_force/max_force/max_depth
from config/params.yaml's drill_controller block (reused here via the
`force_drill_tuner` block so drill_controller's own tuning stays
untouched until you're confident). Edit params.yaml and restart between
trials to change values -- ROS2 params aren't hot-reloaded here.

START LOW: begin with a small feed_force and shallow max_depth, watch
the reported depth/force/duration, and increase gradually. Keep a hand
on the pendant e-stop.

Run: ros2 run potato_scan force_drill_tuner --ros-args --params-file config/params.yaml
"""
import numpy as np
import rclpy
from rclpy.node import Node

from potato_scan.robot_interface import UR5eInterface


class ForceDrillTuner(Node):
    def __init__(self):
        super().__init__('force_drill_tuner')

        self.declare_parameter('robot_ip', '192.168.1.100')
        self.declare_parameter('drill_output_pin', 0)
        self.declare_parameter('feed_force', 8.0)
        self.declare_parameter('max_force', 25.0)
        self.declare_parameter('max_depth', 0.015)
        self.declare_parameter('drill_timeout_s', 8.0)

        self.feed_force = self.get_parameter('feed_force').value
        self.max_force = self.get_parameter('max_force').value
        self.max_depth = self.get_parameter('max_depth').value
        self.drill_timeout_s = self.get_parameter('drill_timeout_s').value

        self.robot = UR5eInterface(
            self.get_parameter('robot_ip').value,
            drill_output_pin=self.get_parameter('drill_output_pin').value)

    def probe_once(self):
        pos, rotvec = self.robot.get_tcp_pose()
        task_frame = list(pos) + list(rotvec)

        self.get_logger().info(
            f'probing at {np.round(pos, 4)} with feed_force={self.feed_force}N '
            f'max_force={self.max_force}N max_depth={self.max_depth * 1000:.1f}mm')

        self.robot.drill_on()
        try:
            reached = self.robot.force_drill(
                task_frame, axis_index=2,
                feed_force=self.feed_force, max_force=self.max_force,
                max_depth=self.max_depth, timeout_s=self.drill_timeout_s)
        finally:
            self.robot.drill_off()

        end_pos, _ = self.robot.get_tcp_pose()
        depth = float(np.linalg.norm(end_pos - pos))
        force = self.robot.receive.getActualTCPForce()
        force_mag = float(np.linalg.norm(force[:3]))

        outcome = 'reached max_depth' if reached else 'stopped on max_force'
        self.get_logger().info(
            f'result: {outcome} -- depth={depth * 1000:.2f}mm force={force_mag:.1f}N')

        # retract back to the pre-probe pose
        self.robot.move_to_pose(pos, rotvec)


def main():
    rclpy.init()
    node = ForceDrillTuner()

    print('\n=== force_drill tuner ===')
    print(f'Loaded: feed_force={node.feed_force}N max_force={node.max_force}N '
          f'max_depth={node.max_depth * 1000:.1f}mm')
    print('Enabling freedrive -- position the tool tip at standoff distance from the '
          'potato surface, +Z pointing outward along the intended insertion axis.\n')
    node.robot.control.teachMode()

    try:
        while True:
            cmd = input("Press Enter to run one probe from the current pose ('q' to quit): ").strip().lower()
            if cmd == 'q':
                break
            node.robot.control.endTeachMode()
            node.probe_once()
            node.robot.control.teachMode()
    finally:
        node.robot.control.endTeachMode()
        node.robot.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
