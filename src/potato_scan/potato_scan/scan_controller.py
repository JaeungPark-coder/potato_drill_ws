"""Main orchestrator (stage 1): drives the UR5e through a data-driven
scan of a fixed potato until real coverage of its surface is complete.

Unlike a fixed orbit/candidate-sphere schedule, coverage here is judged
from the ACTUAL accumulated point cloud (/potato_scan/merged_cloud),
binned into a live (elevation, azimuth) grid around the potato center
(see surface_coverage.SurfaceCoverageGrid). This is what lets the scan
adapt automatically to whatever shape the current potato has: an odd
lump or a deep eye pocket that occludes itself from some angles shows up
directly as an empty cell in the real reconstruction, not as a guess.

The scan runs in two phases, split by whether the motion is the same
for every potato or specific to this one:

PHASE A -- fixed raster orbit (scan_schedule.RasterOrbitSchedule,
`raster_orbit: true`). Park at `scan_radius` facing the potato, sweep the
camera up and down one vertical column, rotate `azimuth_step_deg` around
the potato, sweep the next column, and repeat until the orbit closes.
Identical for every potato, so it is a fixed schedule. It runs to
completion even if coverage_threshold is met partway: a threshold met on
the near side says nothing about the far side not yet visited.

PHASE B -- gap filling, driven by what phase A actually missed:
  1. pick the nearest not-yet-filled, not-yet-given-up grid cell
     (minimizes robot travel between views). This is the
     `view_policy: 'heuristic'` default; set `view_policy: 'rl'` +
     `rl_model_path` to have a trained policy choose instead -- see
     potato_scan/rl/scan_policy_backend.py and
     isaac/train_scan_policy.py -- reusing the same grid/recovery
     bookkeeping for steps 2-5. This is the half worth learning: WHICH
     cells are missing and what motion recovers them depends on the
     individual potato's lumps, eye pockets and mounting pin.
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
  5. stop when coverage_ratio >= threshold or no open cells remain

`max_views` caps total robot moves across BOTH phases.

Publishes /potato_scan/view_candidates (MarkerArray, red=empty /
green=filled / orange=unscannable after recovery failed) so progress is
visible in RViz, and /potato_scan/scan_complete (Bool) when done.
"""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import (QoSProfile, DurabilityPolicy, ReliabilityPolicy,
                       HistoryPolicy)
from std_msgs.msg import Bool, Int32
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, PointStamped
from scipy.spatial.transform import Rotation as Rot

from potato_scan.pose_utils import look_at_rotation, rotmat_to_rotvec, camera_pose_to_tcp_pose
from potato_scan.scan_schedule import RasterOrbitSchedule
from potato_scan.surface_coverage import SurfaceCoverageGrid
from potato_scan.robot_interface import UR5eInterface
from potato_scan.isaac_robot_interface import IsaacSimRobotInterface

# Camera roll offsets (degrees, about the optical axis) tried in order when a
# view pose is rejected as unreachable. Rolling the camera about its own
# optical axis spins the IMAGE without changing which surface patch is in
# frame, so roll is a free parameter -- the same spare-DOF trick
# drill_controller.ROLL_SEARCH_DEG uses for the rotationally symmetric bit.
# Smallest deviation from level first, then wider swings.
CAMERA_ROLL_SEARCH_DEG = [0, 45, -45, 90, -90, 135, -135, 180]


class ScanController(Node):
    def __init__(self):
        super().__init__('scan_controller')

        # See main()'s MultiThreadedExecutor for why this exists:
        # robot_backend=isaac_sim's move_to_pose polls TF, and
        # _wait_for_cloud_to_settle polls /potato_scan/point_count, both
        # from inside _run_step's own timer callback -- those subscriptions
        # need to run concurrently with _run_step, not queued behind it.
        self._cb_group = ReentrantCallbackGroup()

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
        # Phase A: the fixed raster orbit (scan_schedule.RasterOrbitSchedule)
        # every potato gets before any gap-filling starts -- park at
        # scan_radius facing the potato, sweep the camera up/down one
        # vertical column, rotate azimuth_step_deg around it, sweep the
        # next column, repeat until the orbit closes. Set false to skip
        # straight to gap-driven views (the old behaviour).
        # The potato sits on top of a mounting pin. The fixture repeats its
        # lateral position well, but the HEIGHT of the potato's centre moves
        # with every potato, since a bigger one's centre sits further above
        # the same pin. potato_center is what look_at aims the camera at and
        # what the coverage grid bins around, so a stale value tilts the
        # whole scan. With this on, it is re-fitted from the accumulated
        # cloud (surface_coverage.estimate_center) and the configured value
        # above becomes a starting guess plus a safety bound.
        self.declare_parameter('potato_center_auto', True)
        self.declare_parameter('potato_center_max_shift', 0.03)
        self.declare_parameter('potato_center_min_points', 800)
        self.declare_parameter('raster_orbit', True)
        self.declare_parameter('azimuth_step_deg', 45.0)
        self.declare_parameter('elevation_step_deg', 25.0)
        # hand-eye calibration result (camera pose in TCP frame) -- REPLACE with your calibration.
        self.declare_parameter('tcp_cam_translation', [0.0, -0.05, 0.05])
        self.declare_parameter('tcp_cam_quat_xyzw', [0.0, 0.0, 0.0, 1.0])
        # 'heuristic' (default) = SurfaceCoverageGrid.next_gap_direction picks
        # the nearest uncovered cell. 'rl' = a trained policy (see
        # potato_scan/rl/ and isaac/train_scan_policy.py) picks the next view
        # instead -- requires rl_model_path and an Isaac Sim bring-up
        # (robot_backend:=isaac_sim), since that's what the policy was
        # trained against.
        self.declare_parameter('view_policy', 'heuristic')
        self.declare_parameter('rl_model_path', '')

        self.base_frame = self.get_parameter('base_frame').value
        self.potato_center = np.array(self.get_parameter('potato_center').value)
        self.scan_radius = self.get_parameter('scan_radius').value
        self.coverage_threshold = self.get_parameter('coverage_threshold').value
        self.max_views = self.get_parameter('max_views').value
        self.settle_time_s = self.get_parameter('settle_time_s').value
        self.max_local_retries = min(
            self.get_parameter('max_local_retries').value, 6)

        # Kept separate from self.potato_center: the auto-fit is always
        # bounded against the value the operator configured, so repeated
        # updates cannot walk the centre away from the fixture over a run.
        self.configured_potato_center = self.potato_center.copy()
        self.auto_center = self.get_parameter('potato_center_auto').value
        self.potato_center_max_shift = self.get_parameter('potato_center_max_shift').value
        self.potato_center_min_points = self.get_parameter('potato_center_min_points').value

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

        if self.get_parameter('raster_orbit').value:
            self.raster = RasterOrbitSchedule(
                min_elevation_deg=self.get_parameter('min_elevation_deg').value,
                max_elevation_deg=self.get_parameter('max_elevation_deg').value,
                azimuth_step_deg=self.get_parameter('azimuth_step_deg').value,
                elevation_step_deg=self.get_parameter('elevation_step_deg').value)
            if len(self.raster) > self.max_views:
                self.get_logger().warn(
                    f'raster orbit needs {len(self.raster)} views but max_views is '
                    f'{self.max_views} -- the sweep will be cut short and gap-filling '
                    f'will never run. Raise max_views, or coarsen azimuth_step_deg/'
                    f'elevation_step_deg.')
            else:
                self.get_logger().info(
                    f'raster orbit: {len(self.raster)} views, leaving '
                    f'{self.max_views - len(self.raster)} of max_views for gap-filling')
        else:
            self.raster = None

        view_policy_mode = self.get_parameter('view_policy').value
        if view_policy_mode == 'rl':
            from potato_scan.rl.scan_policy_backend import RLViewPolicy
            self.view_policy = RLViewPolicy(
                self.get_parameter('rl_model_path').value,
                self.coverage.min_elevation_deg, self.coverage.max_elevation_deg)
            self.get_logger().info(
                f"view_policy=rl, loaded {self.get_parameter('rl_model_path').value}")
        else:
            self.view_policy = None

        self.create_subscription(
            PointCloud2, '/potato_scan/merged_cloud', self._on_cloud, 10, callback_group=self._cb_group)
        self._latest_point_count = None
        self.create_subscription(
            Int32, '/potato_scan/point_count', self._on_point_count, 10, callback_group=self._cb_group)

        self.marker_pub = self.create_publisher(MarkerArray, '/potato_scan/view_candidates', 10)
        self.complete_pub = self.create_publisher(Bool, '/potato_scan/scan_complete', 10)
        # drill_controller needs whatever centre the scan settled on: its
        # fixture keep-out cone is defined relative to it, and with
        # potato_center_auto on, the configured value is only a guess.
        # TRANSIENT_LOCAL so the drill node still receives it if it
        # subscribes after this was published.
        self.center_pub = self.create_publisher(
            PointStamped, '/potato_scan/potato_center',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE,
                       history=HistoryPolicy.KEEP_LAST))

        backend = self.get_parameter('robot_backend').value
        if backend == 'isaac_sim':
            self.robot = IsaacSimRobotInterface(
                self, base_frame=self.base_frame,
                tcp_frame=self.get_parameter('tcp_frame').value,
                callback_group=self._cb_group)
        else:
            self.robot = UR5eInterface(self.get_parameter('robot_ip').value)

        self._last_direction = None
        self._views_taken = 0
        self._unreachable_views = 0
        self._done = False
        self.create_timer(0.5, self._run_step, callback_group=self._cb_group)

    def _on_cloud(self, msg: PointCloud2):
        pts = np.array(list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)))
        if self.auto_center:
            self._update_potato_center(pts)
        self.coverage.set_from_points(pts, self.potato_center)

    def _update_potato_center(self, points):
        """Re-fit the potato's centre from the cloud so far, if the fit is
        trustworthy. Rebinning afterwards is free: set_from_points rebuilds
        the grid from scratch every time, so the coverage map stays
        consistent with whatever centre is current."""
        candidate = self.coverage.estimate_center(
            points, self.potato_center,
            max_shift=self.potato_center_max_shift,
            min_points=self.potato_center_min_points)
        if candidate is None:
            return

        drift = float(np.linalg.norm(candidate - self.configured_potato_center))
        if drift > self.potato_center_max_shift:
            self.get_logger().warn(
                f'potato_center fit landed {drift * 1000:.0f}mm from the configured '
                f'{np.round(self.configured_potato_center, 3)} (limit '
                f'{self.potato_center_max_shift * 1000:.0f}mm) -- ignoring it. The fit has '
                'probably latched onto the fixture or background rather than the potato.')
            return

        moved = float(np.linalg.norm(candidate - self.potato_center))
        self.potato_center = candidate
        self.get_logger().info(
            f'potato_center -> {np.round(candidate, 4)} (moved {moved * 1000:.1f}mm, '
            f'{drift * 1000:.1f}mm from configured)')

    def _on_point_count(self, msg: Int32):
        self._latest_point_count = msg.data

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
        (and therefore the coverage grid) has a chance to update.

        Returns True if the arm actually reached a pose for this view.
        A requested viewpoint can be unreachable (past a joint limit,
        through a wrist singularity, or simply outside the UR5e's
        envelope at this potato_center/scan_radius), and move_to_pose
        reports that by returning False rather than raising. Since camera
        roll is free, that is not the end of it: the same viewpoint is
        retried at each CAMERA_ROLL_SEARCH_DEG offset before giving up,
        which costs nothing optically and often finds a wrist
        configuration the arm accepts.

        Only reached views count against max_views -- a rejected pose
        moves no motor and consumes no scan budget.
        """
        cam_pos = self.potato_center + direction * radius
        for roll_deg in CAMERA_ROLL_SEARCH_DEG:
            cam_rot = look_at_rotation(cam_pos, self.potato_center, roll_deg=roll_deg)
            tcp_pos, tcp_rot = camera_pose_to_tcp_pose(
                cam_pos, cam_rot, self.r_tcp_cam, self.t_tcp_cam)
            if not self.robot.move_to_pose(tcp_pos, rotmat_to_rotvec(tcp_rot)):
                continue
            if roll_deg:
                self.get_logger().info(f'view reached via camera roll {roll_deg}deg')
            self._views_taken += 1
            self._wait_for_cloud_to_settle()
            return True

        self._unreachable_views += 1
        self.get_logger().warn(
            f'view direction={np.round(direction, 2)} radius={radius:.3f}m unreachable at every '
            f'camera roll -- skipping. If many views fail, potato_center/scan_radius likely put '
            f'the orbit outside the arm workspace.')
        return False

    def _wait_for_cloud_to_settle(self, poll_period_s=0.2, stable_reads_required=2, min_wait_s=1.1):
        """Waits for /potato_scan/point_count to stop growing (the merged
        cloud has caught up with this view) instead of always sleeping the
        full settle_time_s regardless of how long that actually takes --
        still capped at settle_time_s so a view whose count never
        stabilizes (stuck TF, dead camera) doesn't hang the scan. Relies on
        _on_point_count running concurrently with this poll (both in
        self._cb_group, spun via main()'s MultiThreadedExecutor) so the
        count read here is actually live, not whatever it was before this
        move started.

        min_wait_s: floor beneath which "stable" reads don't count, even if
        two consecutive polls happen to match -- pointcloud_accumulator.py's
        publish_status timer only fires once per second, so polling every
        poll_period_s=0.2s would otherwise see the same not-yet-updated
        count twice within ~0.4s almost every time and return before the
        new view's points have actually been merged/published."""
        t0 = time.time()
        stable_count = 0
        last_count = self._latest_point_count
        while time.time() - t0 < self.settle_time_s:
            time.sleep(poll_period_s)
            current = self._latest_point_count
            elapsed = time.time() - t0
            if current is not None and current == last_count:
                stable_count += 1
                if stable_count >= stable_reads_required and elapsed >= min_wait_s:
                    return
            else:
                stable_count = 0
            last_count = current

    def _scan_with_recovery(self, e, a, direction, radius_scale=1.0):
        """Attempt the nominal view (at self.scan_radius * radius_scale --
        radius_scale is 1.0 for the heuristic, policy-chosen when
        view_policy == 'rl'); if the targeted cell is still empty in the
        real reconstruction afterwards, retry that exact spot with
        independent recovery motions (closer/farther radius, tilted
        angle) before giving up on it. Returns True if any attempt
        (nominal or recovery) filled the cell."""
        self.get_logger().info(f'moving to cell ({e},{a}) direction={np.round(direction, 2)}')
        reached = self._attempt_view(direction, self.scan_radius * radius_scale)
        if reached and self.coverage.is_filled(e, a):
            return True

        reason = 'still empty after nominal view' if reached else 'nominal view was unreachable'
        self.get_logger().warn(
            f'cell ({e},{a}) {reason} -- retrying with independent recovery motions '
            f'(these change radius and angle, so they can also get around an '
            f'unreachable nominal pose)')

        for attempt, (perturbed_dir, recovery_radius_scale) in enumerate(
                self.coverage.recovery_attempts(direction, self.max_local_retries), start=1):
            radius = self.scan_radius * recovery_radius_scale
            self.get_logger().info(
                f'cell ({e},{a}) recovery attempt {attempt}/{self.max_local_retries}: '
                f'direction={np.round(perturbed_dir, 2)} radius={radius:.3f}')
            if not self._attempt_view(perturbed_dir, radius):
                continue
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
        in_raster = self.raster is not None and not self.raster.done
        phase = f'raster {self.raster.progress}' if in_raster else 'gap-filling'
        self.get_logger().info(
            f'[{phase}] coverage={coverage:.2f} resolved={resolved:.2f} '
            f'views_taken={self._views_taken} estimated_potato_radius={radius_str}')

        if self._views_taken >= self.max_views:
            self._finish()
            return

        # Phase A runs to completion even once coverage_threshold is met:
        # the sweep is what guarantees a baseline model of EVERY potato,
        # and a threshold met early on one side says nothing about the far
        # side that hasn't been visited yet. Only phase B stops on coverage.
        if in_raster:
            elevation_deg, azimuth_deg, direction = self.raster.next_view()
            self.get_logger().info(
                f'raster view {self.raster.progress}: elevation={elevation_deg:.0f}deg '
                f'azimuth={azimuth_deg:.0f}deg')
            self._attempt_view(direction, self.scan_radius)
            self._last_direction = direction
            return

        if coverage >= self.coverage_threshold or resolved >= 1.0:
            self._finish()
            return

        if self.view_policy is not None:
            direction, radius_scale = self.view_policy.next_view(
                self.coverage, self._last_direction, self._views_taken, self.max_views)
            e, a = self.coverage.direction_to_cell(direction)
        else:
            e, a, direction = self.coverage.next_gap_direction(self._last_direction)
            if e is None:
                self._finish()
                return
            radius_scale = 1.0

        ok = self._scan_with_recovery(e, a, direction, radius_scale)
        if not ok:
            self.coverage.mark_unscannable(e, a)
        self._last_direction = direction

    def _finish(self):
        self._done = True
        n_unscannable = int(np.sum(self.coverage.unscannable))
        coverage = self.coverage.coverage_ratio()
        if self._unreachable_views:
            self.get_logger().warn(
                f'{self._unreachable_views} view(s) were unreachable at every camera roll and '
                'were skipped -- check potato_center/scan_radius against the arm workspace')
        if n_unscannable:
            self.get_logger().warn(
                f'scan complete: coverage={coverage:.2f}, {n_unscannable} unscannable spot(s) -- '
                'shown orange in /potato_scan/view_candidates. The model has real gaps there; if a '
                'potato eye could be hiding in one, reposition the potato/fixture and rescan before drilling.')
        else:
            self.get_logger().info(f'scan complete, coverage={coverage:.2f}')
        center_msg = PointStamped()
        center_msg.header.frame_id = self.base_frame
        center_msg.header.stamp = self.get_clock().now().to_msg()
        center_msg.point = Point(x=float(self.potato_center[0]),
                                 y=float(self.potato_center[1]),
                                 z=float(self.potato_center[2]))
        self.center_pub.publish(center_msg)

        self.complete_pub.publish(Bool(data=True))
        self.robot.stop()


def main():
    rclpy.init()
    node = ScanController()
    # MultiThreadedExecutor -- see ScanController.__init__'s _cb_group
    # comment: robot_backend=isaac_sim's move_to_pose polls TF, and
    # _wait_for_cloud_to_settle polls /potato_scan/point_count, both from
    # inside _run_step's own timer callback. A single-threaded executor (or
    # leaving these on the node's default MutuallyExclusiveCallbackGroup)
    # can't service those other subscriptions while _run_step is still
    # running, so the polled values would never actually refresh and every
    # move/settle would just time out. Only matters for
    # robot_backend=isaac_sim -- the real UR5e path talks to RTDE directly.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.robot.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
