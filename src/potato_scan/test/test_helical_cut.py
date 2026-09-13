"""Checks the widening pass geometry: that it degenerates to the bore it
replaces, that it opens into a cone, and that it starts exactly where the
plunge left the tool."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from potato_scan.drill_task_planner import helical_cut_path

CONTACT = np.array([0.500, 0.010, 0.170])
DEPTH = 0.008
LATERAL_RADIUS = 0.004
STRAIGHT_DOWN = np.array([0.0, 0.0, -1.0])


@pytest.fixture
def rng():
    return np.random.default_rng(4)


def axial_and_radial(path, contact, axis):
    """Each waypoint's depth below the surface and distance off the axis."""
    relative = path - contact
    axial = relative @ axis
    radial = np.linalg.norm(relative - np.outer(axial, axis), axis=1)
    return axial, radial


# --- 1. it degenerates to the bore the plunge already made ---------------

def test_zero_lateral_radius_never_leaves_the_axis():
    path = helical_cut_path(CONTACT, STRAIGHT_DOWN, DEPTH, lateral_radius=0.0)
    _, radial = axial_and_radial(path, CONTACT, STRAIGHT_DOWN)
    assert radial.max() < 1e-12


def test_it_runs_from_the_hole_bottom_out_to_the_surface():
    path = helical_cut_path(CONTACT, STRAIGHT_DOWN, DEPTH, lateral_radius=0.0)
    axial, _ = axial_and_radial(path, CONTACT, STRAIGHT_DOWN)
    assert np.allclose(path[0], CONTACT + STRAIGHT_DOWN * DEPTH)
    assert np.allclose(path[-1], CONTACT)
    assert np.all(np.diff(axial) < 0), 'it must rise monotonically out of the hole'


# --- 2. a positive radius opens a cone -----------------------------------

def test_the_radius_grows_linearly_with_height():
    """A cone, not a cylinder: at the bottom the cut is the bore it already
    made, and it only widens on the way out."""
    path = helical_cut_path(CONTACT, STRAIGHT_DOWN, DEPTH,
                            lateral_radius=LATERAL_RADIUS, turns=2.0)
    axial, radial = axial_and_radial(path, CONTACT, STRAIGHT_DOWN)
    height = 1.0 - axial / DEPTH            # 0 at the bottom, 1 at the surface

    assert abs(radial[0]) < 1e-12
    assert radial[-1] == pytest.approx(LATERAL_RADIUS, abs=1e-12)
    assert np.allclose(radial, LATERAL_RADIUS * height, atol=1e-12)


def test_it_actually_spirals_the_requested_number_of_turns():
    path = helical_cut_path(CONTACT, STRAIGHT_DOWN, DEPTH,
                            lateral_radius=LATERAL_RADIUS, turns=2.0)
    relative = path - CONTACT
    reference = (np.array([0.0, 0.0, 1.0]) if abs(STRAIGHT_DOWN[2]) < 0.9
                 else np.array([1.0, 0.0, 0.0]))
    u = np.cross(reference, STRAIGHT_DOWN)
    u /= np.linalg.norm(u)
    v = np.cross(STRAIGHT_DOWN, u)

    angle = np.unwrap(np.arctan2(relative @ v, relative @ u))
    assert (angle[-1] - angle[0]) / (2 * np.pi) == pytest.approx(2.0, abs=1e-9)


@pytest.mark.parametrize('turns', [0.5, 1.0, 3.0])
def test_waypoint_count_follows_turns_times_points_per_turn(turns):
    path = helical_cut_path(CONTACT, STRAIGHT_DOWN, DEPTH, LATERAL_RADIUS,
                            turns=turns, points_per_turn=16)
    assert len(path) == int(round(turns * 16)) + 1


# --- 3. containment, for any insertion axis ------------------------------

def test_every_waypoint_stays_inside_the_cone_it_is_cutting(rng):
    """Random axes because the eye normal is arbitrary: a path that escapes
    its own cone is cutting material the plan never accounted for."""
    worst = 0.0
    for _ in range(200):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        depth = rng.uniform(0.003, 0.012)
        lateral_radius = rng.uniform(0.0, 0.008)

        path = helical_cut_path(CONTACT, axis, depth, lateral_radius,
                                turns=rng.uniform(0.5, 4.0))
        axial, radial = axial_and_radial(path, CONTACT, axis)

        assert np.all(axial >= -1e-12)
        assert np.all(axial <= depth + 1e-12)
        worst = max(worst, float(np.max(radial - lateral_radius * (1.0 - axial / depth))))

    assert worst < 1e-12, f'worst excursion past the cone {worst * 1e9:.3f} nm'


# --- 4. it starts where the plunge actually left the tool ----------------

def test_the_first_waypoint_is_the_tools_current_position(rng):
    """drill_controller reconstructs the contact point as tcp - axis * depth
    and asks for a path from it. If the first waypoint is not the tool's own
    position, the pass jumps before it cuts."""
    for _ in range(50):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        reached = rng.uniform(0.002, 0.010)

        tcp_now = CONTACT + axis * reached          # where force_drill stopped
        contact = tcp_now - axis * reached          # what the controller rebuilds
        path = helical_cut_path(contact, axis, reached, lateral_radius=LATERAL_RADIUS)

        assert np.linalg.norm(path[0] - tcp_now) < 1e-12


# --- 5. feed axis and cut axis are the same column -----------------------

def test_the_controllers_rotvec_really_is_the_insertion_axis(rng):
    """_widening_pass takes tool_z from the same column of the same rotation
    that force_drill feeds along, so the two agree by construction -- this
    checks that construction still holds."""
    for _ in range(50):
        rotvec = rng.normal(size=3)
        tool_z = Rot.from_rotvec(rotvec).as_matrix()[:, 2]
        path = helical_cut_path(CONTACT, tool_z, DEPTH, LATERAL_RADIUS)
        axial, _ = axial_and_radial(path, CONTACT, tool_z)
        assert axial[0] == pytest.approx(DEPTH, abs=1e-12)
        assert axial[-1] == pytest.approx(0.0, abs=1e-12)
