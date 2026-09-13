"""Validates the flange-mounted (wrist_3) eye-in-hand scan geometry:
the camera roll sweep, and the hand-eye chain that turns a desired camera
pose into the TCP pose the arm is actually commanded to."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from potato_scan.pose_utils import (
    camera_pose_to_tcp_pose, look_at_rotation, rotmat_to_rotvec)
from potato_scan.scan_schedule import RasterOrbitSchedule

ROLLS = [0, 45, -45, 90, -90, 135, -135, 180]
CENTER = np.array([0.50, 0.00, 0.15])
RADIUS = 0.15

# A realistic flange mount: camera 60 mm off the flange axis, 35 mm forward,
# tilted 15 deg so it looks slightly past the drill bit.
R_TCP_CAM = Rot.from_euler('xyz', [0.0, 15.0, 0.0], degrees=True).as_matrix()
T_TCP_CAM = np.array([0.060, 0.0, 0.035])


@pytest.fixture(scope='module')
def front_view():
    """Camera pose and rotation for the straight-ahead view."""
    position = CENTER + np.array([1.0, 0.0, 0.0]) * RADIUS
    return position, look_at_rotation(position, CENTER)


@pytest.mark.parametrize('roll_deg', ROLLS)
def test_roll_leaves_the_optical_axis_on_the_potato(roll_deg, front_view):
    """Rolling the camera must not re-aim it -- that is what makes the roll
    sweep a free retry when a view is unreachable."""
    position, base = front_view
    rotation = look_at_rotation(position, CENTER, roll_deg=roll_deg)

    aim = CENTER - position
    aim = aim / np.linalg.norm(aim)
    assert np.allclose(rotation[:, 2], base[:, 2], atol=1e-12)
    assert np.allclose(rotation[:, 2], aim, atol=1e-12)

    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert abs(np.linalg.det(rotation) - 1.0) < 1e-12


def test_each_roll_is_a_genuinely_different_wrist_pose(front_view):
    """If the rolls collapsed to the same arm pose the sweep would be a
    no-op that silently costs the scan every retry it appears to make."""
    position, _ = front_view
    rotvecs = []
    for roll_deg in ROLLS:
        rotation = look_at_rotation(position, CENTER, roll_deg=roll_deg)
        _, tcp_rot = camera_pose_to_tcp_pose(position, rotation, R_TCP_CAM, T_TCP_CAM)
        rotvecs.append(rotmat_to_rotvec(tcp_rot))

    closest = min(np.linalg.norm(rotvecs[i] - rotvecs[j])
                  for i in range(len(ROLLS)) for j in range(i + 1, len(ROLLS)))
    assert closest > 1e-3, f'two rolls only {np.degrees(closest):.3f} deg apart'


def test_hand_eye_chain_round_trips_for_every_raster_view():
    """Drive the conversion forward again: the TCP pose it returns must put
    the camera exactly where the scan asked for it."""
    schedule = RasterOrbitSchedule()
    worst_position = worst_rotation = 0.0
    checked = 0

    while not schedule.done:
        _, _, direction = schedule.next_view()
        camera_position = CENTER + direction * RADIUS
        for roll_deg in ROLLS:
            camera_rotation = look_at_rotation(camera_position, CENTER, roll_deg=roll_deg)
            tcp_pos, tcp_rot = camera_pose_to_tcp_pose(
                camera_position, camera_rotation, R_TCP_CAM, T_TCP_CAM)

            landed_position = tcp_pos + tcp_rot @ T_TCP_CAM
            landed_rotation = tcp_rot @ R_TCP_CAM
            worst_position = max(worst_position,
                                 np.linalg.norm(landed_position - camera_position))
            worst_rotation = max(worst_rotation,
                                 Rot.from_matrix(landed_rotation.T @ camera_rotation).magnitude())
            checked += 1

    assert checked == 40 * len(ROLLS)
    assert worst_position < 1e-12, f'{worst_position * 1e6:.3f} um'
    assert worst_rotation < 1e-12, f'{np.degrees(worst_rotation) * 1e6:.3f} udeg'


def test_the_mount_offset_is_actually_modelled(front_view):
    """A TCP pose equal to the camera pose would mean the mount was ignored,
    which is the failure that puts the camera 60 mm from where it was aimed."""
    camera_position, camera_rotation = front_view
    tcp_position, _ = camera_pose_to_tcp_pose(
        camera_position, camera_rotation, R_TCP_CAM, T_TCP_CAM)
    assert np.linalg.norm(tcp_position - camera_position) > 0.05
