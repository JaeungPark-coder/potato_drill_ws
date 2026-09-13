"""Checks the drill actually points INTO the potato.

The bug this exists to catch shipped once: the tool rotation was built from
+normal, so the feed axis pointed away from the surface and the wrist was
driven through the potato's own volume. Both halves are checked here -- the
feed lands on the eye, and the arm behind the tool stays outside.
"""
import io
from pathlib import Path

import numpy as np
import pytest

from potato_scan.drill_task_planner import (
    approach_pose, normal_rotation)

PLANNER_SRC = (Path(__file__).resolve().parents[1]
               / 'potato_scan' / 'drill_task_planner.py')

POTATO_CENTER = np.array([0.50, 0.00, 0.15])
POTATO_RADIUS = 0.035
STANDOFF = 0.03

NORMALS = [np.array([1.0, 0.0, 0.0]),
           np.array([0.0, 0.0, 1.0]),
           np.array([0.3, -0.7, 0.65]),
           np.array([-0.5, 0.5, 0.707])]


def unit(vector):
    return np.asarray(vector, dtype=float) / np.linalg.norm(vector)


def distance_to_segment(point, start, end):
    """Closest approach of `point` to the segment, not to its endpoints.

    Checking only the far endpoint is what makes the wrist test weak: a shaft
    can pass clean through the potato and come out the other side with both
    ends comfortably outside it.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    along = end - start
    t = np.clip(np.dot(point - start, along) / np.dot(along, along), 0.0, 1.0)
    return float(np.linalg.norm(point - (start + t * along)))


def test_approach_candidates_still_build_the_rotation_from_minus_normal():
    """A source check, because the sign is the whole bug and it is one
    character wide. If this fails, read the sign before the geometry tests."""
    source = io.open(PLANNER_SRC, encoding='utf-8').read()
    assert 'normal_rotation(-np.asarray(normal, dtype=float))' in source


@pytest.mark.parametrize('normal', NORMALS, ids=lambda n: str(np.round(n, 2)))
def test_feeding_along_plus_z_reaches_the_eye(normal):
    normal = unit(normal)
    eye = POTATO_CENTER + normal * POTATO_RADIUS
    approach = approach_pose(eye, normal, STANDOFF)
    feed_axis = normal_rotation(-normal)[:, 2]      # force_drill feeds along +Z

    # the approach point sits outside the potato, further out than the eye
    assert np.linalg.norm(approach - POTATO_CENTER) > np.linalg.norm(eye - POTATO_CENTER)
    # and feeding the standoff distance from it lands on the eye
    landed = approach + feed_axis * STANDOFF
    assert np.linalg.norm(landed - eye) < 1e-9


@pytest.mark.parametrize('normal', NORMALS, ids=lambda n: str(np.round(n, 2)))
def test_the_whole_tool_shaft_stays_outside_the_potato(normal):
    """100 mm back along the tool axis is roughly where the wrist sits, and
    none of the shaft between there and the tip may be inside the potato."""
    normal = unit(normal)
    eye = POTATO_CENTER + normal * POTATO_RADIUS
    approach = approach_pose(eye, normal, STANDOFF)
    feed_axis = normal_rotation(-normal)[:, 2]

    wrist = approach - feed_axis * 0.10
    clearance = distance_to_segment(POTATO_CENTER, approach, wrist)
    assert clearance >= POTATO_RADIUS, f'shaft passes {clearance * 1000:.1f} mm from centre' 


def test_the_old_convention_really_was_backwards():
    """Guards the guard: if +normal ever stopped being wrong, the tests above
    would be checking nothing, and this says so loudly."""
    normal = np.array([0.0, 0.0, 1.0])
    eye = POTATO_CENTER + normal * POTATO_RADIUS
    approach = approach_pose(eye, normal, STANDOFF)

    old_feed_axis = normal_rotation(normal)[:, 2]
    old_wrist = approach - old_feed_axis * 0.10
    clearance = distance_to_segment(POTATO_CENTER, approach, old_wrist)
    assert clearance < POTATO_RADIUS, \
        'the old convention no longer drives the tool through the potato'


def test_the_perception_and_tool_conventions_are_exact_opposites():
    """normal_rotation is now the single definition of this convention --
    eye_detector's normal_to_quat and the drill's approach search both build
    on it rather than on copies. What they must agree on is the sign: the
    perception frame's +Z is the outward normal, the tool's +Z points the
    other way, into the surface.
    """
    for normal in NORMALS:
        normal = unit(normal)
        perception_z = normal_rotation(normal)[:, 2]
        tool_z = normal_rotation(-normal)[:, 2]

        assert np.allclose(perception_z, normal, atol=1e-12)
        assert np.allclose(tool_z, -perception_z, atol=1e-12)
