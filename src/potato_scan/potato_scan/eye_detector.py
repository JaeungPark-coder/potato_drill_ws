"""Stage 2: detect potato eyes (sprout buds) on the completed scan and
publish their 3D coordinates + surface normals for the drill task
planner (stage 3), plus RViz markers showing each eye's coordinate and
distance from the robot base.

Triggered once by /potato_scan/scan_complete; operates on a snapshot of
/potato_scan/merged_cloud taken at that moment.

Detection is classical, no training data needed -- see
potato_scan/surface_curvature.py, which holds all of the geometry and has
no ROS or Open3D in it so it can be tested against surfaces whose true
curvature is known. In outline:

  1. From each point's neighbourhood covariance: an outward normal
     (oriented away from the potato centre rather than by propagating
     orientation across the cloud) and the surface variation
     kappa = lambda0 / sum(lambda), a dimensionless "how curved".
  2. From a quadratic fit in the local frame: the two principal
     curvatures, and from them Chen & Bhanu's shape index S, a
     scale-invariant "which way curved" -- 0 is a cup, 1 is a dome.
  3. Keep points that are both curved enough AND cup-shaped, cluster
     them, and keep clusters the size of an eye.
  4. Measure how much darker each surviving cluster is than the skin
     immediately around it, and optionally reject on that.

The two-axis test replaced a single score -- the offset of each point's
neighbourhood centroid along its own normal, in metres. That score could
not separate a pit from the ridge beside it (both are curved), and being
a length it sat near the sensor's noise floor and had to be retuned
whenever scan density changed. Ablated on a synthetic potato, kappa is
what rejects non-eyes and the shape index is what localises them: without
the latter the position error grows from 0.7 mm to 2.4 mm, without the
former 9 of 10 candidates are false.

Step 4 exists because the geometry has a ceiling it cannot lift on its own:
a clod of soil sitting in a hollow IS a pit, with the same curvature and the
same shape index as an eye. Colour is the only axis that separates them, and
it is already arriving -- the camera publishes a coloured cloud and it was
being discarded. The contrast is measured relative to the surrounding
surface rather than as an absolute colour, since that survives changes in
lighting and skin tone, and it is reported but not enforced by default: the
visible band alone is a known-weak tuber-vs-soil discriminator, good on wet
material and doubtful when dry, so it earns a place as evidence before it
earns one as a gate. A learned keypoint/segmentation model, or a near-
infrared band, is the step beyond that if soil remains a problem.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, Pose, Point, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial.transform import Rotation as Rot

from potato_scan.cloud_rgb import unpack_rgb
from potato_scan.drill_task_planner import normal_rotation
from potato_scan.surface_curvature import describe_surface, find_eye_candidates

# A potato carries roughly 5-10 eyes. Counts far outside that say the
# thresholds are wrong for this scan rather than that this potato is
# unusual, and it is worth saying so before the drill acts on them.
PLAUSIBLE_EYE_COUNT = (2, 15)


def normal_to_quat(normal):
    """Quaternion whose local +Z axis aligns with `normal`.

    The rotation itself comes from drill_task_planner rather than being
    rebuilt here: this is the perception side of a convention the drill side
    also depends on (it builds the TOOL frame from -normal, so the bit points
    into the surface), and two copies of it could drift apart while every
    test still passed.
    """
    return Rot.from_matrix(normal_rotation(normal)).as_quat()  # x, y, z, w


class EyeDetector(Node):
    def __init__(self):
        super().__init__('eye_detector')
        self.declare_parameter('base_frame', 'base_link')
        # Needed to orient normals outward. Only a fallback: scan_controller
        # publishes the centre it actually settled on, which is the one the
        # drill's fixture keep-out cone is measured from too.
        self.declare_parameter('potato_center', [0.5, 0.0, 0.15])
        self.declare_parameter('knn', 30)
        # Minimum surface variation. Dimensionless and bounded [0, 1/3], but
        # NOT scale-invariant: with a fixed knn a denser scan gives a smaller
        # neighbourhood, which reads flatter. Tune against a real scan using
        # the percentiles this node logs.
        self.declare_parameter('curvature_min', 0.015)
        # Maximum shape index. 0.35 keeps cups and ruts, rejects saddles
        # (0.5), ridges (0.75) and domes (1.0). Scale-invariant, so unlike
        # curvature_min this should not need per-setup tuning.
        self.declare_parameter('shape_index_max', 0.35)
        # How much darker than the surrounding skin a candidate must be.
        # Curvature and shape index describe a pit, and a clod of soil in a
        # hollow is also a pit -- colour is the only axis that separates
        # them. Negative disables the gate, measuring and reporting the
        # contrast without rejecting anything, which is the right default
        # until the number has been seen on real potatoes: the visible band
        # is a weak tuber-vs-soil discriminator by itself, good on wet
        # material and doubtful when dry.
        self.declare_parameter('min_color_contrast', -1.0)
        # Directional agreement among the cluster's member normals, 0..1.
        # Negative (the default) measures and reports it without rejecting
        # anything -- deliberately, because no threshold here has been
        # earned yet: on synthetic clouds this number does NOT predict how
        # wrong the normal is (see surface_curvature.find_eye_candidates).
        # It is logged next to drill_controller's normal_vs_radial_deg so
        # one real run produces the pairs a threshold could be set from.
        self.declare_parameter('min_normal_consistency', -1.0)
        self.declare_parameter('cluster_eps', 0.003)
        self.declare_parameter('cluster_min_points', 8)
        self.declare_parameter('min_eye_diameter', 0.002)
        self.declare_parameter('max_eye_diameter', 0.015)
        # Same bound and same default as scan_controller's own
        # max_expected_radius ("larger than the largest potato you'd ever
        # load") -- reused here rather than invented fresh, since it already
        # means exactly what this filter needs: how far from potato_center a
        # real surface point can plausibly be. CONFIRMED 2026-09-14 this
        # filter is needed: with no distance gate, 4 of 11 "eyes" detected
        # against a real Isaac Sim cloud clustered near the ROBOT'S OWN BASE
        # (curvature+shape-index alone can't tell that apart from a real
        # eye) -- potato_center only orients normals, it was never used to
        # restrict which points get considered in the first place.
        self.declare_parameter('max_expected_radius', 0.07)

        self.base_frame = self.get_parameter('base_frame').value
        self.potato_center = np.array(self.get_parameter('potato_center').value, dtype=float)
        self.knn = self.get_parameter('knn').value
        self.curvature_min = self.get_parameter('curvature_min').value
        self.shape_index_max = self.get_parameter('shape_index_max').value
        min_contrast = self.get_parameter('min_color_contrast').value
        self.min_color_contrast = None if min_contrast < 0.0 else min_contrast
        min_consistency = self.get_parameter('min_normal_consistency').value
        self.min_normal_consistency = (
            None if min_consistency < 0.0 else min_consistency)
        self.cluster_eps = self.get_parameter('cluster_eps').value
        self.cluster_min_points = self.get_parameter('cluster_min_points').value
        self.min_eye_diameter = self.get_parameter('min_eye_diameter').value
        self.max_eye_diameter = self.get_parameter('max_eye_diameter').value
        self.max_expected_radius = self.get_parameter('max_expected_radius').value

        self._latest_cloud_msg = None
        self.create_subscription(PointCloud2, '/potato_scan/merged_cloud', self._on_cloud, 10)
        self.create_subscription(Bool, '/potato_scan/scan_complete', self._on_scan_complete, 10)
        self.create_subscription(
            PointStamped, '/potato_scan/potato_center', self._on_potato_center, 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/potato_scan/eye_markers', 10)
        self.pose_pub = self.create_publisher(PoseArray, '/potato_scan/eye_poses', 10)

    def _on_cloud(self, msg: PointCloud2):
        self._latest_cloud_msg = msg

    def _on_potato_center(self, msg: PointStamped):
        self.potato_center = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=float)
        self.get_logger().info(
            f'potato_center from scan: {np.round(self.potato_center, 4)} '
            '(normals are oriented outward from here)')

    def _on_scan_complete(self, msg: Bool):
        if not msg.data or self._latest_cloud_msg is None:
            return
        self.get_logger().info('scan_complete received, running eye detection')
        self.detect_and_publish(self._latest_cloud_msg)

    def _log_distributions(self, points):
        """What the two scores actually look like on THIS scan.

        curvature_min has to be set for the scan density in use, and these
        percentiles are how to set it: an eye occupies a small fraction of
        the surface, so a workable threshold sits far out in kappa's upper
        tail. Printed every run so the number can be checked against real
        data instead of carried over from a synthetic.
        """
        _, kappa, s_index = describe_surface(points, self.potato_center, knn=self.knn)
        q = [50, 90, 99, 99.9]
        kappa_q = np.percentile(kappa, q)
        selected = (kappa > self.curvature_min) & (s_index < self.shape_index_max)
        self.get_logger().info(
            'kappa percentiles ' + ', '.join(f'p{p}={v:.4f}' for p, v in zip(q, kappa_q))
            + f' | curvature_min={self.curvature_min}'
            + f' | cup-shaped (S<{self.shape_index_max}): '
              f'{100.0 * float((s_index < self.shape_index_max).mean()):.1f}% of points'
            + f' | both: {int(selected.sum())} points')
        if self.curvature_min < kappa_q[1]:
            self.get_logger().warn(
                f'curvature_min={self.curvature_min} sits below this scan\'s 90th '
                f'percentile ({kappa_q[1]:.4f}) -- that admits a large fraction of the '
                f'surface as "curved", which usually means it is set for a different '
                f'scan density than this one')

    def detect_and_publish(self, cloud_msg: PointCloud2):
        has_rgb = any(f.name == 'rgb' for f in cloud_msg.fields)
        fields = ('x', 'y', 'z', 'rgb') if has_rgb else ('x', 'y', 'z')
        # read_points returns a STRUCTURED array (named dtype fields), not a
        # plain (N, len(fields)) float array -- see pointcloud_accumulator.py's
        # matching fix/comment (2026-09-14).
        raw = np.array(list(pc2.read_points(
            cloud_msg, field_names=fields, skip_nans=True)))
        if len(raw) < self.knn + 1:
            self.get_logger().warn('not enough points for eye detection')
            return
        pts = np.column_stack([raw['x'], raw['y'], raw['z']])
        colors = unpack_rgb(raw['rgb']) if has_rgb else None
        if colors is None:
            self.get_logger().warn(
                'merged cloud has no colour, so detection is geometry-only and a soil '
                'clod in a hollow is indistinguishable from an eye')

        self._log_distributions(pts)

        candidates = find_eye_candidates(
            pts, self.potato_center, knn=self.knn,
            curvature_min=self.curvature_min,
            shape_index_max=self.shape_index_max,
            cluster_eps=self.cluster_eps,
            cluster_min_points=self.cluster_min_points,
            min_diameter=self.min_eye_diameter,
            max_diameter=self.max_eye_diameter,
            colors=colors, min_color_contrast=self.min_color_contrast,
            max_center_distance=self.max_expected_radius,
            min_normal_consistency=self.min_normal_consistency)

        eyes = [(c['position'], c['normal'], c['diameter']) for c in candidates]
        self.get_logger().info(f'detected {len(eyes)} potato eyes')
        for i, c in enumerate(candidates):
            self.get_logger().info(
                f"  #{i} at {np.round(c['position'], 4)} diameter "
                f"{c['diameter'] * 1000:.1f}mm shape_index {c['shape_index']:.3f} "
                f"({c['points']} points) "
                f"normal_consistency {c['normal_consistency']:.4f}"
                + ('' if colors is None else
                   f" colour_contrast {c['color_contrast']:+.3f} "
                   f"(vs {c['surround_points']} surround pts)"))

        low, high = PLAUSIBLE_EYE_COUNT
        if eyes and not (low <= len(eyes) <= high):
            self.get_logger().warn(
                f'{len(eyes)} eyes is outside the {low}-{high} a potato plausibly has. '
                f'Check curvature_min against the percentiles above before drilling '
                f'these.')

        self._publish_markers(eyes)
        self._publish_poses(eyes)

    def _publish_poses(self, eyes):
        arr = PoseArray()
        arr.header.frame_id = self.base_frame
        arr.header.stamp = self.get_clock().now().to_msg()
        for position, normal, _ in eyes:
            pose = Pose()
            pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
            q = normal_to_quat(normal)
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
            arr.poses.append(pose)
        self.pose_pub.publish(arr)

    def _publish_markers(self, eyes):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, (position, normal, diameter) in enumerate(eyes):
            distance = float(np.linalg.norm(position))  # distance from base_link origin

            sphere = Marker()
            sphere.header.frame_id = self.base_frame
            sphere.header.stamp = stamp
            sphere.ns = 'eyes'
            sphere.id = i * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = Point(x=float(position[0]), y=float(position[1]), z=float(position[2]))
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = max(diameter, 0.004)
            sphere.color.a = 1.0
            sphere.color.r = 1.0
            sphere.color.g = 0.6

            text = Marker()
            text.header.frame_id = self.base_frame
            text.header.stamp = stamp
            text.ns = 'eye_labels'
            text.id = i * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(
                x=float(position[0]), y=float(position[1]), z=float(position[2]) + 0.015)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.008
            text.color.a = 1.0
            text.color.r = text.color.g = text.color.b = 1.0
            text.text = (f'#{i} ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}) '
                         f'd={distance:.3f}m')

            arr.markers.append(sphere)
            arr.markers.append(text)
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = EyeDetector()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
