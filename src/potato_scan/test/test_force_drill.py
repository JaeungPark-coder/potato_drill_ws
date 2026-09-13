"""Drives the real UR5eInterface.force_drill against a stubbed RTDE.

The bug this exists to catch shipped once: depth was measured from the
approach pose rather than from contact, so the drill reported reaching depth
while still 15 mm clear of the potato. force_drill is now two-phase -- feed
until the force says contact, only then count depth -- and this simulates a
TCP feeding toward a surface to check all three of its outcomes.
"""
import types

import pytest

# RTDE is not installed on a development machine and is not needed: the stub
# has to be in place before robot_interface is imported at all.
for _name in ('rtde_control', 'rtde_receive', 'rtde_io'):
    _module = types.ModuleType(_name)
    _module.RTDEControlInterface = lambda *a, **k: None
    _module.RTDEReceiveInterface = lambda *a, **k: None
    _module.RTDEIOInterface = lambda *a, **k: None
    import sys
    sys.modules.setdefault(_name, _module)

from potato_scan.robot_interface import UR5eInterface   # noqa: E402


class FeedingRobot(UR5eInterface):
    """A TCP feeding along +Z into a surface at `surface_distance_m`.

    Contact force ramps with penetration once past the surface, so the stub
    reproduces the shape of a real plunge: nothing, then a rising load.
    """

    def __init__(self, surface_distance_m, stiffness_n_per_m=4000.0, feed_speed=0.02):
        self.surface_distance = surface_distance_m
        self.stiffness = stiffness_n_per_m
        self.feed_speed = feed_speed
        self.t = 0.0
        self.control = types.SimpleNamespace(
            forceMode=lambda *a: None, forceModeStop=lambda: None)
        self.receive = types.SimpleNamespace(
            getActualTCPPose=self._pose, getActualTCPForce=self._force)

    def _travel(self):
        return self.t * self.feed_speed

    def _pose(self):
        self.t += 0.05                      # one poll_dt of motion per read
        return [0.0, 0.0, self._travel(), 0.0, 0.0, 0.0]

    def _force(self):
        penetration = max(0.0, self._travel() - self.surface_distance)
        return [0.0, 0.0, penetration * self.stiffness, 0.0, 0.0, 0.0]


def drill(surface_distance, max_depth=0.008, max_force=40.0,
          contact_force=5.0, max_approach_travel=0.05):
    robot = FeedingRobot(surface_distance)
    return robot.force_drill([0, 0, 0, 0, 0, 0],
                             max_depth=max_depth, max_force=max_force,
                             contact_force=contact_force,
                             max_approach_travel=max_approach_travel,
                             timeout_s=30.0, poll_dt=0.0)


def test_reaches_depth_measured_from_contact():
    """30 mm of standoff, then 8 mm of cut -- the real approach geometry."""
    outcome = drill(0.030)
    assert outcome.status == 'reached'
    assert outcome.contacted
    assert outcome.depth_m == pytest.approx(0.008, abs=0.002)
    assert bool(outcome) is True


def test_never_touching_the_potato_is_not_a_success():
    """The surface sits beyond the allowed approach travel, which is what a
    wrong eye pose looks like. Reporting `reached` here was the old bug."""
    outcome = drill(0.060)
    assert outcome.status == 'no_contact'
    assert not outcome.contacted
    assert outcome.depth_m == 0.0
    assert bool(outcome) is False


def test_binding_trips_the_force_limit_before_depth():
    outcome = drill(0.030, max_force=12.0)
    assert outcome.status == 'force_limit'
    assert outcome.contacted
    assert outcome.depth_m < 0.008


def test_depth_is_referenced_to_contact_not_to_the_start():
    """The distinguishing property, stated directly: move the surface and the
    cut depth must not move with it."""
    near = drill(0.020)
    far = drill(0.035)
    assert near.status == far.status == 'reached'
    assert near.depth_m == pytest.approx(far.depth_m, abs=0.002)
