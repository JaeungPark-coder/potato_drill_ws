"""Step 1 of the recommended bring-up order: eye-in-hand hand-eye
calibration. Produces the tcp_cam_translation / tcp_cam_quat_xyzw values
that pose_utils.camera_pose_to_tcp_pose() and every downstream node
(scan_controller, eye_detector, drill_controller) depend on -- run this
BEFORE trusting any scan.

Setup:
  - Print/mount a checkerboard target somewhere FIXED near the potato
    fixture (it must not move during calibration), flat and visible from
    a range of angles the camera will actually scan from.
  - Put the UR5e in freedrive (this script enables teach mode for you via
    RTDE, so you can hand-guide the arm) and jog the camera to see the
    board from a varied set of orientations -- vary tilt/rotation, not
    just straight-on distance, or the solve will be poorly conditioned.
  - At each pose, press Enter to capture; take at least 10-15 poses.

Method: classical eye-in-hand hand-eye calibration (Tsai-Lenz), i.e.
solving AX=XB for the static camera-in-gripper transform, using
cv2.calibrateHandEye. Board pose per frame comes from solvePnP against
the camera's published intrinsics; robot pose per frame comes from RTDE.

Run: ros2 run potato_scan handeye_calibration --ros-args -p output_path:=handeye_result.yaml
Then copy the printed tcp_cam_translation / tcp_cam_quat_xyzw into
config/params.yaml (scan_controller block) by hand -- this script does
not overwrite params.yaml automatically.
"""
import threading

import cv2
import numpy as np
import rclpy
import rtde_control
import rtde_receive
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation as Rot
from sensor_msgs.msg import CameraInfo, Image


class HandEyeCalibration(Node):
    def __init__(self):
        super().__init__('handeye_calibration')

        self.declare_parameter('robot_ip', '192.168.1.100')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('chessboard_cols', 9)  # inner corners, along image width
        self.declare_parameter('chessboard_rows', 6)  # inner corners, along image height
        self.declare_parameter('square_size_m', 0.025)
        self.declare_parameter('output_path', 'handeye_result.yaml')

        self.cols = self.get_parameter('chessboard_cols').value
        self.rows = self.get_parameter('chessboard_rows').value
        self.square_size = self.get_parameter('square_size_m').value
        self.output_path = self.get_parameter('output_path').value

        self.bridge = CvBridge()
        self._latest_image = None
        self._camera_matrix = None
        self._dist_coeffs = None

        self.create_subscription(
            Image, self.get_parameter('image_topic').value, self._on_image, 10)
        self.create_subscription(
            CameraInfo, self.get_parameter('camera_info_topic').value, self._on_camera_info, 10)

        robot_ip = self.get_parameter('robot_ip').value
        self.rtde_receive = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.rtde_control = rtde_control.RTDEControlInterface(robot_ip)

        # object points for solvePnP: board corners in the board's own frame, Z=0
        self._objp = np.zeros((self.rows * self.cols, 3), dtype=np.float64)
        self._objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2) * self.square_size

        self.r_gripper2base = []
        self.t_gripper2base = []
        self.r_target2cam = []
        self.t_target2cam = []

    def _on_image(self, msg: Image):
        self._latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _on_camera_info(self, msg: CameraInfo):
        self._camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._dist_coeffs = np.array(msg.d, dtype=np.float64)

    def capture_one(self):
        """Detect the board in the latest frame + read the current TCP
        pose. Returns True and stores the sample pair on success."""
        if self._latest_image is None or self._camera_matrix is None:
            self.get_logger().warn('no image / camera_info received yet')
            return False

        gray = cv2.cvtColor(self._latest_image, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (self.cols, self.rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            self.get_logger().warn('chessboard not found in current frame -- reposition and retry')
            return False

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        ok, rvec, tvec = cv2.solvePnP(self._objp, corners, self._camera_matrix, self._dist_coeffs)
        if not ok:
            self.get_logger().warn('solvePnP failed on this frame')
            return False

        r_target2cam, _ = cv2.Rodrigues(rvec)
        t_target2cam = tvec.reshape(3)

        tcp_pose = self.rtde_receive.getActualTCPPose()
        r_gripper2base = Rot.from_rotvec(tcp_pose[3:]).as_matrix()
        t_gripper2base = np.array(tcp_pose[:3])

        self.r_target2cam.append(r_target2cam)
        self.t_target2cam.append(t_target2cam)
        self.r_gripper2base.append(r_gripper2base)
        self.t_gripper2base.append(t_gripper2base)
        return True

    def solve(self):
        n = len(self.r_gripper2base)
        if n < 3:
            raise RuntimeError(f'need at least 3 captures to solve, have {n}')

        r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            self.r_gripper2base, self.t_gripper2base,
            self.r_target2cam, self.t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI)

        quat_xyzw = Rot.from_matrix(r_cam2gripper).as_quat()
        translation = t_cam2gripper.reshape(3)
        return translation, quat_xyzw

    def close(self):
        self.rtde_control.endTeachMode()
        self.rtde_control.stopScript()
        self.rtde_control.disconnect()
        self.rtde_receive.disconnect()


def _spin_in_background(node):
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    return executor


def main():
    rclpy.init()
    node = HandEyeCalibration()
    executor = _spin_in_background(node)

    print('\n=== hand-eye calibration ===')
    print('Enabling freedrive (teach mode) -- you can now hand-guide the UR5e.')
    print(f'Target: {node.cols}x{node.rows} inner-corner checkerboard, '
          f'{node.square_size * 1000:.1f} mm squares, FIXED in the scene.\n')
    node.rtde_control.teachMode()

    captured = 0
    try:
        while True:
            cmd = input(
                f'[{captured} captured] move to a new view, then press Enter to capture '
                f"('q' to finish and solve): ").strip().lower()
            if cmd == 'q':
                break
            if node.capture_one():
                captured += 1
                print(f'  captured pose {captured}')
            else:
                print('  capture failed, see warning above -- try again')

        if captured < 10:
            print(f'\nWARNING: only {captured} captures -- 10-15+ with varied '
                  'orientation is recommended for a well-conditioned solve.')

        translation, quat_xyzw = node.solve()
        print('\n=== result: camera pose in TCP frame ===')
        print(f'tcp_cam_translation: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]')
        print(f'tcp_cam_quat_xyzw:   [{quat_xyzw[0]:.6f}, {quat_xyzw[1]:.6f}, '
              f'{quat_xyzw[2]:.6f}, {quat_xyzw[3]:.6f}]')
        print('\nCopy these two lines into the scan_controller block of config/params.yaml.\n')

        with open(node.output_path, 'w') as f:
            yaml.safe_dump({
                'scan_controller': {
                    'ros__parameters': {
                        'tcp_cam_translation': [float(v) for v in translation],
                        'tcp_cam_quat_xyzw': [float(v) for v in quat_xyzw],
                    }
                }
            }, f)
        print(f'Also wrote this to {node.output_path}')

    finally:
        executor.shutdown()
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
