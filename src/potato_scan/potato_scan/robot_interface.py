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
                     max_depth=0.015, timeout_s=8.0, poll_dt=0.05,
                     free_axis_speed_limit=0.05, held_axis_deviation_limit=0.005):
        """Feed along the NEGATIVE direction of `axis_index` of `task_frame`
        (i.e. into the surface, opposite the outward normal used to build
        task_frame's orientation) with `feed_force` newtons, until either
        `max_depth` (measured TCP travel) or `max_force` (measured TCP
        force) is reached, then stops force mode. The other 5 axes stay
        position-held to within `held_axis_deviation_limit` m/rad.

        Returns True if max_depth was reached, False if aborted on the
        force limit (breaking through the far side is the expected safe
        outcome -- tune max_force/max_depth per drill bit and potato size
        on the real setup before trusting the defaults).
        """
        selection_vector = [0, 0, 0, 0, 0, 0]
        selection_vector[axis_index] = 1
        wrench = [0.0] * 6
        wrench[axis_index] = -feed_force

        limits = [held_axis_deviation_limit] * 6
        limits[axis_index] = free_axis_speed_limit

        start_pos, _ = self.get_tcp_pose()
        self.control.forceMode(task_frame, selection_vector, wrench, 2, limits)

        reached = False
        t0 = time.time()
        try:
            while time.time() - t0 < timeout_s:
                pos, _ = self.get_tcp_pose()
                depth = float(np.linalg.norm(pos - start_pos))
                force = self.receive.getActualTCPForce()
                force_mag = float(np.linalg.norm(force[:3]))
                if depth >= max_depth:
                    reached = True
                    break
                if force_mag >= max_force:
                    break
                time.sleep(poll_dt)
        finally:
            self.control.forceModeStop()

        return reached

    def stop(self):
        self.control.stopL()

    def close(self):
        self.drill_off()
        self.control.stopScript()
        self.control.disconnect()
        self.receive.disconnect()
        self.io.disconnect()
