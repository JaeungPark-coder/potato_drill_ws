"""Main orchestrator (stage 1): drives the UR5e through a data-driven
scan of a fixed potato until real coverage of its surface is complete.

Unlike a fixed orbit/candidate-sphere schedule, coverage here is judged
from the ACTUAL accumulated point cloud (/potato_scan/merged_cloud),
binned into a live (elevation, azimuth) grid around the potato center
(see surface_coverage.SurfaceCoverageGrid). This is what lets the scan
adapt automatically to whatever shape the current potato has: an odd
lump or a deep eye pocket that occludes itself from some angles shows up
directly as an empty cell in the real reconstruction, not as a guess.

Loop:
  1. pick the nearest not-yet-filled, not-yet-given-up grid cell
     (minimizes robot travel between views)
  2. compute a camera pose looking at the potato center from that cell's
     direction at `scan_radius`, convert to a TCP pose via the hand-eye
     extrinsic, and moveL there; settle
  3. re-check that exact cell against the freshly updated grid
  4a. if it's now filled: move on
  4b. if it's still empty (occlusion, glare, out of FOV, ...): retry that
      exact spot with an independent recovery motion (closer/farther
      radius, tilted approach angle; see surface_coverage.RECOVERY_OFFSETS)
      up to `max_local_retries` times. Only if every recovery attempt
      still leaves it empty is the cell given up on and flagged
      `unscannable`, rather than silently treated as done.
  5. stop when coverage_ratio >= threshold, no open cells remain, or
     max_views total robot moves is hit

Publishes /potato_scan/view_candidates (MarkerArray, red=empty /
green=filled / orange=unscannable after recovery failed) so progress is
visible in RViz, and /potato_scan/scan_complete (Bool) when done.
"""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as Rot

from potato_scan.pose_utils import look_at_rotation, rotmat_to_rotvec, camera_pose_to_tcp_pose
from potato_scan.surface_coverage import SurfaceCoverageGrid
from potato_scan.robot_interface import UR5eInterface
from potato_scan.isaac_robot_interface import IsaacSimRobotInterface


class ScanController(Node):
    def __init__(self):
        super().__init__('scan_controller')

        self.declare_parameter('robot_ip', '192.168.1.100')
        # 'rtde' talks to a real UR5e / URSim over RTDE (robot_interface.
        # UR5eInterface). 'isaac_sim' talks to isaac/isaac_scene.py over
        # plain ROS2 topics + TF instead (isaac_robot_interface.
        # IsaacSimRobotInterface) -- same move_to_pose/get_tcp_pose
        # contract either way, nothing else in this node changes.
        self.declare_parameter('robot_backend', 'rtde')
        self.declare_parameter('tcp_frame', 'tool0')  # only used by isaac_sim backend (TF lookup)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('potato_center', [0.5, 0.0, 0.15])
        # No per-potato size measurement: these are generic bounds wide
        # enough for any real potato this fixture could hold (and to
        # reject the table/fixture/gripper). The actual radius of
        # whichever potato is currently mounted is estimated live from
        # the scan itself -- see SurfaceCoverageGrid.
        self.declare_parameter('min_expected_radius', 0.015)
        self.declare_parameter('max_expected_radius', 0.07)
        self.declare_parameter('radius_band', 0.02)
        self.declare_parameter('scan_radius', 0.15)
        self.declare_parameter('elevation_bin_deg', 8.0)
        self.declare_parameter('azimuth_bin_deg', 8.0)
        self.declare_parameter('min_hits_to_fill', 3)
        self.declare_parameter('min_elevation_deg', -15.0)
        self.declare_parameter('max_elevation_deg', 85.0)
        self.declare_parameter('coverage_threshold', 0.95)
        self.declare_parameter('max_views', 60)
        self.declare_parameter('settle_time_s', 1.5)
        self.declare_parameter('max_local_retries', 4)
        # hand-eye calibration result (camera pose in TCP frame) -- REPLACE with your calibration.
        self.declare_parameter('tcp_cam_translation', [0.0, -0.05, 0.05])
        self.declare_parameter('tcp_cam_quat_xyzw', [0.0, 0.0, 0.0, 1.0])

        self.base_frame = self.get_parameter('base_frame').value
        self.potato_center = np.array(self.get_parameter('potato_center').value)
        self.scan_radius = self.get_parameter('scan_radius').value
        self.coverage_threshold = self.get_parameter('coverage_threshold').value
        self.max_views = self.get_parameter('max_views').value
        self.settle_time_s = self.get_parameter('settle_time_s').value
        self.max_local_retries = min(
            self.get_parameter('max_local_retries').value, 6)

        self.r_tcp_cam = Rot.from_quat(self.get_parameter('tcp_cam_quat_xyzw').value).as_matrix()
        self.t_tcp_cam = np.array(self.get_parameter('tcp_cam_translation').value)

        self.coverage = SurfaceCoverageGrid(
            min_expected_radius=self.get_parameter('min_expected_radius').value,
            max_expected_radius=self.get_parameter('max_expected_radius').value,
            radius_band=self.get_parameter('radius_band').value,
            elevation_bin_deg=self.get_parameter('elevation_bin_deg').value,
            azimuth_bin_deg=self.get_parameter('azimuth_bin_deg').value,
            min_elevation_deg=self.get_parameter('min_elevation_deg').value,
            max_elevation_deg=self.get_parameter('max_elevation_deg').value,
            min_hits_to_fill=self.get_parameter('min_hits_to_fill').value,
        )

        self.create_subscription(PointCloud2, '/potato_scan/merged_cloud', self._on_cloud, 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/potato_scan/view_candidates', 10)
        self.complete_pub = self.create_publisher(Bool, '/potato_scan/scan_complete', 10)

        backend = self.get_parameter('robot_backend').value
        if backend == 'isaac_sim':
            self.robot = IsaacSimRobotInterface(
                self, base_frame=self.base_frame,
                tcp_frame=self.get_parameter('tcp_frame').value)
        else:
            self.robot = UR5eInterface(self.get_parameter('robot_ip').value)

        self._last_direction = None
        self._views_taken = 0
        self._done = False
        self.create_timer(0.5, self._run_step)

    def _on_cloud(self, msg: PointCloud2):
        pts = np.array(list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)))
        self.coverage.set_from_points(pts, self.potato_center)

    def _publish_candidate_markers(self):
        dirs, filled, unscannable = self.coverage.all_cells_with_status()
        arr = MarkerArray()
        for i, (d, f, u) in enumerate(zip(dirs, filled, unscannable)):
            pos = self.potato_center + d * self.scan_radius
            m = Marker()
            m.header.frame_id = self.base_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'view_candidates'
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.008
            m.color.a = 0.8
            if u:
                m.color.r = 1.0
                m.color.g = 0.5  # orange: gave up after recovery attempts, model has a real gap here
            elif f:
                m.color.g = 1.0
            else:
                m.color.r = 1.0
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def _attempt_view(self, direction, radius):
        """Move to the camera pose looking at the potato center from
        `direction` at `radius`, and settle so /potato_scan/merged_cloud
        (and therefore the coverage grid) has a chance to update."""
        cam_pos = self.potato_center + direction * radius
        cam_rot = look_at_rotation(cam_pos, self.potato_center)
        tcp_pos, tcp_rot = camera_pose_to_tcp_pose(cam_pos, cam_rot, self.r_tcp_cam, self.t_tcp_cam)
        tcp_rotvec = rotmat_to_rotvec(tcp_rot)

        self.robot.move_to_pose(tcp_pos, tcp_rotvec)
        self._views_taken += 1
        time.sleep(self.settle_time_s)

    def _scan_with_recovery(self, e, a, direction):
        """Attempt the nominal view; if the targeted cell is still empty
        in the real reconstruction afterwards, retry that exact spot with
        independent recovery motions (closer/farther radius, tilted
        angle) before giving up on it. Returns True if any attempt
        (nominal or recovery) filled the cell."""
        self.get_logger().info(f'moving to cell ({e},{a}) direction={np.round(direction, 2)}')
        self._attempt_view(direction, self.scan_radius)
        if self.coverage.is_filled(e, a):
            return True

        self.get_logger().warn(
            f'cell ({e},{a}) still empty after nominal view -- retrying with '
            f'independent recovery motions')

        for attempt, (perturbed_dir, radius_scale) in enumerate(
                self.coverage.recovery_attempts(direction, self.max_local_retries), start=1):
            radius = self.scan_radius * radius_scale
            self.get_logger().info(
                f'cell ({e},{a}) recovery attempt {attempt}/{self.max_local_retries}: '
                f'direction={np.round(perturbed_dir, 2)} radius={radius:.3f}')
            self._attempt_view(perturbed_dir, radius)
            if self.coverage.is_filled(e, a):
                self.get_logger().info(f'cell ({e},{a}) recovered on attempt {attempt}')
                return True

        self.get_logger().warn(
            f'cell ({e},{a}) still empty after {self.max_local_retries} recovery attempts -- '
            'marking unscannable (likely persistent occlusion; check fixture/gripper geometry)')
        return False

    def _run_step(self):
        if self._done:
            return

        self._publish_candidate_markers()

        coverage = self.coverage.coverage_ratio()
        resolved = self.coverage.resolved_ratio()
        radius_str = (f'{self.coverage.estimated_radius * 1000:.1f}mm'
                      if self.coverage.estimated_radius is not None else 'unknown yet')
        self.get_logger().info(
            f'coverage={coverage:.2f} resolved={resolved:.2f} views_taken={self._views_taken} '
            f'estimated_potato_radius={radius_str}')

        if coverage >= self.coverage_threshold or resolved >= 1.0 or self._views_taken >= self.max_views:
            self._finish()
            return

        e, a, direction = self.coverage.next_gap_direction(self._last_direction)
        if e is None:
            self._finish()
            return

        ok = self._scan_with_recovery(e, a, direction)
        if not ok:
            self.coverage.mark_unscannable(e, a)
        self._last_direction = direction

    def _finish(self):
        self._done = True
        n_unscannable = int(np.sum(self.coverage.unscannable))
        coverage = self.coverage.coverage_ratio()
        if n_unscannable:
            self.get_logger().warn(
                f'scan complete: coverage={coverage:.2f}, {n_unscannable} unscannable spot(s) -- '
                'shown orange in /potato_scan/view_candidates. The model has real gaps there; if a '
                'potato eye could be hiding in one, reposition the potato/fixture and rescan before drilling.')
        else:
            self.get_logger().info(f'scan complete, coverage={coverage:.2f}')
        self.complete_pub.publish(Bool(data=True))
        self.robot.stop()


def main():
    rclpy.init()
    node = ScanController()
    try:
        rclpy.spin(node)
    finally:
        node.robot.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
