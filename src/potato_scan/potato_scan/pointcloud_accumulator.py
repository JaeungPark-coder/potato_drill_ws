"""Accumulates the eye-in-hand camera's point clouds into the robot base
frame for RViz visualization, and reports point-count growth so
scan_controller can confirm a given view actually captured new surface
(as opposed to looking at empty space / an occluded angle)."""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header, Int32
import tf2_ros
from tf2_ros import TransformException
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d


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

        self.base_frame = self.get_parameter('base_frame').value
        self.voxel_size = self.get_parameter('voxel_size').value
        camera_topic = self.get_parameter('camera_topic').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST)
        self.sub = self.create_subscription(PointCloud2, camera_topic, self.on_cloud, qos)
        self.cloud_pub = self.create_publisher(PointCloud2, '/potato_scan/merged_cloud', 10)
        self.count_pub = self.create_publisher(Int32, '/potato_scan/point_count', 10)

        self.merged = o3d.geometry.PointCloud()

        self.create_timer(1.0, self.publish_status)

    def on_cloud(self, msg: PointCloud2):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, msg.header.frame_id, msg.header.stamp)
        except TransformException as ex:
            self.get_logger().warn(f'TF lookup failed: {ex}', throttle_duration_sec=2.0)
            return

        points = np.array(list(pc2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True)))
        if points.size == 0:
            return

        mat = transform_to_matrix(tf)
        pts_h = np.hstack([points, np.ones((points.shape[0], 1))])
        pts_base = (mat @ pts_h.T).T[:, :3]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pts_base)
        self.merged += cloud
        self.merged = self.merged.voxel_down_sample(self.voxel_size)

    def publish_status(self):
        count = len(self.merged.points)
        self.count_pub.publish(Int32(data=count))

        if count == 0:
            return
        pts = np.asarray(self.merged.points, dtype=np.float32)
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.base_frame
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
