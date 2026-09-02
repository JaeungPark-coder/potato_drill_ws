"""Camera/TCP pose geometry helpers for potato scanning.

Frame convention:
  base  (B) - robot base_link
  tcp   (T) - UR5e tool flange / TCP frame
  cam   (C) - camera optical frame

Hand-eye calibration (eye-in-hand) gives the static transform T_T_C
(camera pose expressed in the TCP frame): p_T = R_tcp_cam @ p_C + t_tcp_cam.
Replace the placeholder values in config/params.yaml with the real result
of your calibration (e.g. cv2.calibrateHandEye / easy_handeye output).
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def look_at_rotation(cam_pos, target, up=(0.0, 0.0, 1.0)):
    """Rotation matrix (base frame) for a camera at cam_pos whose +Z
    (optical axis) points at target."""
    cam_pos = np.asarray(cam_pos, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    z_axis = target - cam_pos
    z_axis = z_axis / np.linalg.norm(z_axis)

    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        # z_axis nearly parallel to up -> pick a different reference
        up = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    return np.column_stack((x_axis, y_axis, z_axis))


def rotmat_to_rotvec(rot_mat):
    """3x3 rotation matrix -> UR axis-angle vector (rx, ry, rz)."""
    return Rot.from_matrix(rot_mat).as_rotvec()


def rotvec_to_rotmat(rotvec):
    return Rot.from_rotvec(rotvec).as_matrix()


def camera_pose_to_tcp_pose(cam_pos_base, cam_rot_base, r_tcp_cam, t_tcp_cam):
    """Given the desired camera pose in the base frame, and the static
    hand-eye extrinsic (camera pose in the TCP frame), return the TCP
    pose in the base frame that puts the camera where we want it.

    T_B_C = T_B_T . T_T_C  =>  R_B_T = R_B_C @ R_T_C^T
                                t_B_T = t_B_C - R_B_T @ t_T_C
    """
    r_tcp_base = cam_rot_base @ r_tcp_cam.T
    t_tcp_base = np.asarray(cam_pos_base, dtype=float) - r_tcp_base @ np.asarray(t_tcp_cam, dtype=float)
    return t_tcp_base, r_tcp_base
