"""Thin wrapper around ur_rtde for the scan controller.

Motion during scanning is simple free-space movement around a known
fixed point at a safe standoff radius, so direct Cartesian control via
RTDE (moveL) is used instead of MoveIt2 -- no collision-aware planning
is needed here. Stage 3 (drilling) is where MoveIt2 / force_mode should
take over for the approach + compliant insertion.
"""
import time
import numpy as np
import rtde_control
import rtde_receive
import rtde_io

from potato_scan.drill_task_planner import DrillOutcome


class UR5eInterface:
    def __init__(self, robot_ip, speed=0.25, acceleration=0.5, drill_output_pin=0):
        self.robot_ip = robot_ip
        self.speed = speed
        self.acceleration = acceleration
        self.drill_output_pin = drill_output_pin
        self.control = rtde_control.RTDEControlInterface(robot_ip)
        self.receive = rtde_receive.RTDEReceiveInterface(robot_ip)
        self.io = rtde_io.RTDEIOInterface(robot_ip)

    def move_to_pose(self, position, rotvec, speed=None, acceleration=None):
        """Returns True on success. False (rather than raising) on a
        target the controller rejects as unreachable -- e.g. past a joint
        limit or through a wrist singularity -- so callers with a free
        redundant DOF (drill_controller's roll search) can try an
        alternative instead of crashing the node."""
        pose = list(np.asarray(position, dtype=float)) + list(np.asarray(rotvec, dtype=float))
        try:
            result = self.control.moveL(pose, speed or self.speed, acceleration or self.acceleration)
        except RuntimeError:
            return False
        return result is not False

    def get_tcp_pose(self):
        pose = self.receive.getActualTCPPose()
        return np.array(pose[:3]), np.array(pose[3:])

    def drill_on(self):
        """Enable the drill spindle via tool digital output. Wire the drill
        motor relay to this pin (default 0) -- adjust `drill_output_pin` to
        match your actual wiring."""
        self.io.setToolDigitalOut(self.drill_output_pin, True)

    def drill_off(self):
        self.io.setToolDigitalOut(self.drill_output_pin, False)

    def force_drill(self, task_frame, axis_index=2, feed_force=15.0, max_force=40.0,
                     max_depth=0.008, timeout_s=8.0, poll_dt=0.05,
                     contact_force=5.0, max_approach_travel=0.05,
                     free_axis_speed_limit=0.05, held_axis_deviation_limit=0.005):
        """Feed along the POSITIVE direction of `axis_index` of `task_frame`
        with `feed_force` newtons and return a DrillOutcome.

        Positive, because the tool frame's +Z now points INTO the surface
        (drill_controller.normal_rotation builds it from -normal, matching
        rl.drill_policy_spec.compose_approach_pose) -- the drill bit
        extends along the tool's +Z, so that is the direction it cuts.

        Depth is measured from the CONTACT POINT, not from the approach
        pose this starts at. The approach pose sits `standoff` metres off
        the surface, so measuring travel from it -- what this used to do --
        meant any max_depth below standoff reported "reached" while the bit
        was still that far short of the potato, having touched nothing.
        Insertion therefore runs in two phases:

          1. approach: feed until the measured force first rises past
             `contact_force`, which defines depth zero. If the whole
             `max_approach_travel` is consumed without that happening,
             nothing is there -- stop and report 'no_contact' instead of
             pushing on into empty space.
          2. penetration: feed until `max_depth` past that contact point
             ('reached'), or `max_force` ('force_limit').

        The other 5 axes stay position-held to within
        `held_axis_deviation_limit` m/rad throughout.
        """
        selection_vector = [0, 0, 0, 0, 0, 0]
        selection_vector[axis_index] = 1
        wrench = [0.0] * 6
        wrench[axis_index] = feed_force  # +axis: into the surface

        limits = [held_axis_deviation_limit] * 6
        limits[axis_index] = free_axis_speed_limit

        start_pos, _ = self.get_tcp_pose()
        self.control.forceMode(task_frame, selection_vector, wrench, 2, limits)

        contact_pos = None
        peak_force = 0.0
        depth = 0.0
        status = 'timeout'
        t0 = time.time()
        try:
            while time.time() - t0 < timeout_s:
                pos, _ = self.get_tcp_pose()
                force_mag = float(np.linalg.norm(self.receive.getActualTCPForce()[:3]))
                peak_force = max(peak_force, force_mag)

                if contact_pos is None:
                    if force_mag >= contact_force:
                        contact_pos = pos
                    elif float(np.linalg.norm(pos - start_pos)) >= max_approach_travel:
                        status = 'no_contact'
                        break

                if contact_pos is not None:
                    depth = float(np.linalg.norm(pos - contact_pos))
                    if depth >= max_depth:
                        status = 'reached'
                        break
                    if force_mag >= max_force:
                        status = 'force_limit'
                        break

                time.sleep(poll_dt)
        finally:
            self.control.forceModeStop()

        return DrillOutcome(status=status, depth_m=depth, peak_force_n=peak_force,
                            contacted=contact_pos is not None)

    def stop(self):
        self.control.stopL()

    def close(self):
        self.drill_off()
        self.control.stopScript()
        self.control.disconnect()
        self.receive.disconnect()
        self.io.disconnect()
