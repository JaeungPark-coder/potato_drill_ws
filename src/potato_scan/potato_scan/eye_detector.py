"""Stage 2: detect potato eyes (sprout buds) on the completed scan and
publish their 3D coordinates + surface normals for the drill task
planner (stage 3), plus RViz markers showing each eye's coordinate and
distance from the robot base.

Triggered once by /potato_scan/scan_complete; operates on a snapshot of
/potato_scan/merged_cloud taken at that moment.

Detection approach (classical, no training data needed -- swap in a
learned keypoint/segmentation model later if precision is insufficient):
  1. Estimate per-point normals + a concavity score. The score is the
     offset of each point's local-neighborhood centroid along its own
     outward normal: positive => the point sits in a pit (neighbors are
     further "out"), negative => it sits on a bump.
  2. Threshold + DBSCAN-cluster the high-concavity points -> eye
     candidates, filtered by expected eye size.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseArray, Pose, Point
from visualization_msgs.msg import Marker, MarkerArray
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d


def estimate_concavity(points, normals, k=30):
    k = min(k, len(points) - 1)
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)
    neighbor_pts = points[idx[:, 1:]]
    centroids = neighbor_pts.mean(axis=1)
    return np.einsum('ij,ij->i', centroids - points, normals)


def normal_to_quat(normal):
    """Quaternion whose local +Z axis aligns with `normal` (used as the
    drill approach/insertion axis downstream)."""
    z = np.asarray(normal, dtype=float)
    z = z / np.linalg.norm(z)
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.column_stack((x, y, z))
    return Rot.from_matrix(rot).as_quat()  # x, y, z, w


class EyeDetector(Node):
    def __init__(self):
        super().__init__('eye_detector')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('knn', 30)
        self.declare_parameter('concavity_threshold', 0.0015)  # meters, tune per scan density
        self.declare_parameter('cluster_eps', 0.003)
        self.declare_parameter('cluster_min_points', 8)
        self.declare_parameter('min_eye_diameter', 0.002)
        self.declare_parameter('max_eye_diameter', 0.015)

        self.base_frame = self.get_parameter('base_frame').value
        self.knn = self.get_parameter('knn').value
        self.concavity_threshold = self.get_parameter('concavity_threshold').value
        self.cluster_eps = self.get_parameter('cluster_eps').value
        self.cluster_min_points = self.get_parameter('cluster_min_points').value
        self.min_eye_diameter = self.get_parameter('min_eye_diameter').value
        self.max_eye_diameter = self.get_parameter('max_eye_diameter').value

        self._latest_cloud_msg = None
        self.create_subscription(PointCloud2, '/potato_scan/merged_cloud', self._on_cloud, 10)
        self.create_subscription(Bool, '/potato_scan/scan_complete', self._on_scan_complete, 10)

        self.marker_pub = self.create_publisher(MarkerArray, '/potato_scan/eye_markers', 10)
        self.pose_pub = self.create_publisher(PoseArray, '/potato_scan/eye_poses', 10)

    def _on_cloud(self, msg: PointCloud2):
        self._latest_cloud_msg = msg

    def _on_scan_complete(self, msg: Bool):
        if not msg.data or self._latest_cloud_msg is None:
            return
        self.get_logger().info('scan_complete received, running eye detection')
        self.detect_and_publish(self._latest_cloud_msg)

    def detect_and_publish(self, cloud_msg: PointCloud2):
        pts = np.array(list(pc2.read_points(
            cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)))
        if len(pts) < self.knn + 1:
            self.get_logger().warn('not enough points for eye detection')
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.knn))
        pcd.orient_normals_consistent_tangent_plane(self.knn)
        normals = np.asarray(pcd.normals)

        concavity = estimate_concavity(pts, normals, k=self.knn)
        pit_mask = concavity > self.concavity_threshold
        if not np.any(pit_mask):
            self.get_logger().info('no eye candidates found')
            return

        pit_pcd = o3d.geometry.PointCloud()
        pit_pcd.points = o3d.utility.Vector3dVector(pts[pit_mask])
        labels = np.array(pit_pcd.cluster_dbscan(
            eps=self.cluster_eps, min_points=self.cluster_min_points))

        pit_pts = pts[pit_mask]
        pit_normals = normals[pit_mask]

        eyes = []  # (position, normal, diameter)
        for label in set(labels):
            if label < 0:
                continue
            cluster_pts = pit_pts[labels == label]
            cluster_normals = pit_normals[labels == label]
            diameter = float(np.linalg.norm(cluster_pts.max(axis=0) - cluster_pts.min(axis=0)))
            if not (self.min_eye_diameter <= diameter <= self.max_eye_diameter):
                continue
            position = cluster_pts.mean(axis=0)
            normal = cluster_normals.mean(axis=0)
            normal = normal / np.linalg.norm(normal)
            eyes.append((position, normal, diameter))

        self.get_logger().info(f'detected {len(eyes)} potato eyes')
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
