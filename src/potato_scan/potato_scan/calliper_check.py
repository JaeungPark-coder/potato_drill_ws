"""Measure where the bit actually lands, without drilling anything.

The cheapest real measurement this cell can make, and the one it currently
has no equivalent of: every number on record is from synthetic data. This
drives the bit tip to each detected eye with the drill OFF, pauses, and asks
for the distance from the tip to the eye's centre read off a pair of
callipers.

Nothing is cut, so the same potato can be measured again after a change --
which is what makes it a tuning loop rather than a one-shot test. The idea is
borrowed from a potato-eye study that validated its cutting geometry by
standing in a laser line for the blade and measuring to it with a 0.01 mm
calliper rather than cutting anything.

WHAT THE NUMBER MEANS

It is end-to-end: camera intrinsics, hand-eye calibration, the live potato
centre, detection, and the arm's own tracking, all folded into one distance.
That is a feature -- it is the quantity that decides whether a hole lands on
an eye -- but it means a bad number says "something upstream is wrong", not
which thing.

The bar to clear comes from the closest published system, which reports
1.84 mm mean positional error at the sampling site. Eyes run 2-15 mm across,
so under about 2 mm is usable and under 1 mm is comfortable.

    ros2 run potato_scan calliper_check --ros-args --params-file config/params.yaml

Run the scan and detector first; this waits for /potato_scan/eye_poses.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseArray
from scipy.spatial.transform import Rotation as Rot

from potato_scan.isaac_robot_interface import IsaacSimRobotInterface
from potato_scan.drill_task_planner import (
    approach_candidates, tilt_search_sequence, ROLL_SEARCH_DEG)

# Mean positional error reported by the closest published tissue-sampling
# system, used here only as the bar to compare against.
REFERENCE_POSITION_ERROR_M = 0.00184


class CalliperCheck(Node):
    def __init__(self):
        super().__init__('calliper_check')
        self._cb_group = ReentrantCallbackGroup()

        self.declare_parameter('robot_ip', '192.168.1.100')
        self.declare_parameter('robot_backend', 'rtde')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tcp_frame', 'tool0')
        self.declare_parameter('standoff', 0.03)
        self.declare_parameter('max_approach_tilt_deg', 15.0)
        self.declare_parameter('approach_tilt_step_deg', 7.5)
        self.declare_parameter('approach_speed', 0.05)   # slow: a hand is near the tool
        self.declare_parameter('approach_acceleration', 0.2)

        self.standoff = self.get_parameter('standoff').value
        self.tilt_search_deg = tilt_search_sequence(
            self.get_parameter('max_approach_tilt_deg').value,
            self.get_parameter('approach_tilt_step_deg').value)

        if self.get_parameter('robot_backend').value == 'isaac_sim':
            self.robot = IsaacSimRobotInterface(
                self, base_frame=self.get_parameter('base_frame').value,
                tcp_frame=self.get_parameter('tcp_frame').value,
                callback_group=self._cb_group)
        else:
            # Lazy import -- see scan_controller.py's matching comment:
            # robot_interface.py imports rtde_control at module level, which
            # is not installed for robot_backend:=isaac_sim.
            from potato_scan.robot_interface import UR5eInterface
            self.robot = UR5eInterface(
                self.get_parameter('robot_ip').value,
                speed=self.get_parameter('approach_speed').value,
                acceleration=self.get_parameter('approach_acceleration').value)

        self._eyes = None
        self.create_subscription(
            PoseArray, '/potato_scan/eye_poses', self._on_eye_poses, 10,
            callback_group=self._cb_group)

    def _on_eye_poses(self, msg: PoseArray):
        eyes = []
        for pose in msg.poses:
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            quat = [pose.orientation.x, pose.orientation.y,
                    pose.orientation.z, pose.orientation.w]
            eyes.append((position, Rot.from_quat(quat).as_matrix()[:, 2]))
        self._eyes = eyes
        self.get_logger().info(f'received {len(eyes)} eye poses')

    def _present_tip_at(self, position, normal):
        """Move the tip onto the eye, via the same approach the drill would
        use -- the point is to measure THAT pose, not a different one that
        happens to be easier to reach.

        Returns the rotvec used, or None if nothing was reachable.
        """
        for approach, rotvec, tilt_deg, roll_deg in approach_candidates(
                position, normal, self.standoff,
                tilt_search_deg=self.tilt_search_deg,
                roll_search_deg=ROLL_SEARCH_DEG):
            if not self.robot.move_to_pose(approach, rotvec):
                continue
            # from the standoff pose, straight down the insertion axis onto
            # the eye -- no force mode, no drill, just position
            if not self.robot.move_to_pose(position, rotvec):
                continue
            if tilt_deg or roll_deg:
                self.get_logger().info(
                    f'  reached with tilt={tilt_deg:.1f}deg roll={roll_deg}deg')
            return rotvec
        return None

    def run(self):
        if not self._eyes:
            print('no eye poses yet -- run the scan and detector first')
            return

        print(f'\n{len(self._eyes)} eyes. The drill is NEVER switched on, so nothing is')
        print('cut and this potato can be measured again after a change.\n')

        errors = []
        for index, (position, normal) in enumerate(self._eyes):
            print(f'--- eye {index}/{len(self._eyes) - 1} at {np.round(position, 4)} ---')
            rotvec = self._present_tip_at(position, normal)
            if rotvec is None:
                print('  unreachable at every roll and tilt -- skipped\n')
                continue

            reading = input('  calliper, tip to eye centre, in mm '
                            '(blank to skip, q to stop): ').strip().lower()
            if reading == 'q':
                break
            if reading:
                try:
                    errors.append(float(reading) / 1000.0)
                except ValueError:
                    print('  not a number, skipped')

            approach = position + normal * self.standoff
            self.robot.move_to_pose(approach, rotvec)   # back off before the next

        if not errors:
            print('\nno measurements recorded')
            return

        errors = np.asarray(errors)
        print(f'\n{len(errors)} measurements')
        print(f'  mean   {errors.mean() * 1000:.2f} mm')
        print(f'  median {np.median(errors) * 1000:.2f} mm')
        print(f'  worst  {errors.max() * 1000:.2f} mm')
        print(f'  vs the published reference of '
              f'{REFERENCE_POSITION_ERROR_M * 1000:.2f} mm mean: '
              f'{"better" if errors.mean() < REFERENCE_POSITION_ERROR_M else "worse"}')
        print('\n  This is end-to-end -- intrinsics, hand-eye, potato centre, detection')
        print('  and tracking all fold into it. A bad number says something upstream is')
        print('  wrong, not which thing; bisect by re-running after changing one.')


def main():
    rclpy.init()
    node = CalliperCheck()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    import threading
    threading.Thread(target=executor.spin, daemon=True).start()

    print('waiting for /potato_scan/eye_poses ... (Ctrl+C to give up)')
    try:
        import time
        while node._eyes is None:
            time.sleep(0.2)
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.robot.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
