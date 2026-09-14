"""Accumulates the eye-in-hand camera's point clouds into the robot base
frame for RViz visualization, and reports point-count growth so
scan_controller can confirm a given view actually captured new surface
(as opposed to looking at empty space / an occluded angle).

Each incoming frame is statistically outlier-filtered BEFORE it is merged.
Stereo depth sensors manufacture "flying pixels" along depth
discontinuities -- points interpolated between the potato's silhouette and
the background, which lie in empty space near the rim. They matter here
twice over:

  * scan_controller judges coverage by counting merged points per surface
    direction, so flying pixels sitting in the radius band register as
    surface that was never actually seen. Cells get marked covered, the
    gap-filling phase never visits them, and the reconstruction keeps a
    hole it believes it has filled.
  * eye_detector estimates normals and concavity from local neighbourhoods,
    and these points are concentrated exactly at the rim, where the
    detected eye normals are already least reliable.

Filtering per frame rather than on the merged cloud is deliberate: the
artefact is per-frame, and once many views are merged a flying pixel has
genuine neighbours from other views to hide among.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header, Int32
import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d

from potato_scan.cloud_rgb import pack_rgb, unpack_rgb


# The camera topic this subscribes to (/camera/depth/color/points) is the
# COLOURED cloud, so rgb is already arriving; it was simply being dropped by
# asking read_points for x, y and z only. eye_detector needs it: curvature and
# shape index describe a pit, and a clod of soil in a hollow is also a pit, so
# geometry alone has no way to separate them.
RGB_FIELDS = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
]


def transform_to_matrix(t):
    q = t.transform.rotation
    trans = t.transform.translation
    rot = Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = [trans.x, trans.y, trans.z]
    return m


class PointCloudAccumulator(Node):
    def __init__(self):
        super().__init__('pointcloud_accumulator')
        self.declare_parameter('camera_topic', '/camera/depth/color/points')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('voxel_size', 0.001)  # 1mm, potato-scale detail
        # Statistical outlier removal on each incoming frame. A point is
        # dropped when its mean distance to its `outlier_neighbors` nearest
        # neighbours is more than `outlier_std_ratio` standard deviations
        # above the frame's average. Set outlier_neighbors to 0 to disable.
        self.declare_parameter('outlier_neighbors', 50)
        self.declare_parameter('outlier_std_ratio', 1.5)

        self.base_frame = self.get_parameter('base_frame').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.outlier_neighbors = self.get_parameter('outlier_neighbors').value
        self.outlier_std_ratio = self.get_parameter('outlier_std_ratio').value
        camera_topic = self.get_parameter('camera_topic').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(PointCloud2, camera_topic, self.on_cloud, qos)
        self.cloud_pub = self.create_publisher(PointCloud2, '/potato_scan/merged_cloud', 10)
        self.count_pub = self.create_publisher(Int32, '/potato_scan/point_count', 10)

        self.merged = o3d.geometry.PointCloud()
        self._warned_no_rgb = False

        self.create_timer(1.0, self.publish_status)

    def on_cloud(self, msg: PointCloud2):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, msg.header.frame_id, msg.header.stamp)
        except TransformException as ex:
            self.get_logger().warn(f'TF lookup failed: {ex}', throttle_duration_sec=2.0)
            return

        has_rgb = any(f.name == 'rgb' for f in msg.fields)
        fields = ('x', 'y', 'z', 'rgb') if has_rgb else ('x', 'y', 'z')
        # read_points returns a STRUCTURED array (one named dtype field per
        # requested field name), not a plain (N, len(fields)) float array --
        # CONFIRMED 2026-09-14 against this project's own isaac_scene.py
        # publisher: raw[:, :3] on that raised "too many indices for array:
        # array is 1-dimensional, but 2 were indexed" on every single cloud,
        # so this had never actually been run against a real PointCloud2
        # message before. Indexed by field name instead, which works
        # regardless of the array's dtype layout.
        raw = np.array(list(pc2.read_points(msg, field_names=fields, skip_nans=True)))
        if raw.size == 0:
            return
        points = np.column_stack([raw['x'], raw['y'], raw['z']])
        colors = unpack_rgb(raw['rgb']) if has_rgb else None
        if not has_rgb and not self._warned_no_rgb:
            self._warned_no_rgb = True
            self.get_logger().warn(
                'the camera topic carries no rgb field, so eye_detector gets geometry '
                'only and cannot tell a soil clod in a hollow from an eye. The coloured '
                'RealSense topic is /camera/depth/color/points.')

        mat = transform_to_matrix(tf)
        pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
        pts_base = (mat @ pts_h.T).T[:, :3]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts_base)
        if colors is not None:
            # carried through the merge and the voxel grid, which averages
            # colour over each voxel the same way it averages position
            cloud.colors = o3d.utility.Vector3dVector(colors)
        cloud = self._remove_outliers(cloud)
        self.merged += cloud
        self.merged = self.merged.voxel_down_sample(self.voxel_size)

    def _remove_outliers(self, cloud):
        """Drop this frame's flying pixels. Skipped when disabled, or when
        the frame holds too few points for the neighbourhood statistic to
        mean anything -- filtering a handful of points would just delete
        the frame."""
        if self.outlier_neighbors <= 0:
            return cloud
        if len(cloud.points) <= self.outlier_neighbors:
            return cloud

        before = len(cloud.points)
        filtered, _ = cloud.remove_statistical_outlier(
            nb_neighbors=self.outlier_neighbors, std_ratio=self.outlier_std_ratio)
        removed = before - len(filtered.points)
        if removed:
            self.get_logger().debug(
                f'outlier filter dropped {removed}/{before} points '
                f'({100.0 * removed / before:.1f}%)')
        return filtered

    def publish_status(self):
        count = len(self.merged.points)
        self.count_pub.publish(Int32(data=count))

        if count == 0:
            return
        pts = np.asarray(self.merged.points, dtype=np.float32)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.base_frame
        if self.merged.has_colors():
            packed = pack_rgb(np.asarray(self.merged.colors))
            rows = [(*point, color) for point, color in zip(pts, packed)]
            msg = pc2.create_cloud(header, RGB_FIELDS, rows)
        else:
            msg = pc2.create_cloud_xyz32(header, pts)
        self.cloud_pub.publish(msg)

    def save(self, path):
        o3d.io.write_point_cloud(path, self.merged)


def main():
    rclpy.init()
    node = PointCloudAccumulator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
