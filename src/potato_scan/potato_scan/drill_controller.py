"""Stage 3: visits each detected potato eye in an efficient order and
removes it with the UR5e + drill end-effector, using force_mode to feed
into the surface along its normal so the (initially unknown) exact
insertion depth is force-limited rather than blindly position-commanded.
The feed is two-phase -- close the standoff gap until contact is
detected, then penetrate max_depth past THAT point -- so max_depth means
penetration into the potato rather than travel from the approach pose;
see robot_interface.force_drill.

Triggered by publishing True on /potato_scan/start_drilling once the
eye positions in RViz look correct.

SAFETY: max_force / feed_force / contact_force below still need tuning
on the real setup (drill bit, potato size, UR5e safety limits) before
running unattended -- start with a low feed_force and a conservative
max_depth, and be ready to hit the pendant e-stop. max_depth's 8mm
default is at least grounded in the literature (Divyanth et al. 2025
sample potato eyes at an intended 7.00mm); the force values are not.

Different eyes on different potatoes end up needing very different
approach angles (whatever the local surface normal happens to be), and
not every one of those is guaranteed reachable -- some may sit past a
joint limit or through a wrist singularity for the default orientation.
Rather than just failing on those, this exploits the fact that a drill
bit is rotationally symmetric about its own insertion axis: rotation
about that axis (roll) doesn't change the actual drilling geometry at
all, so it's a free parameter that gets swept (ROLL_SEARCH_DEG) to find
an orientation the arm can actually reach before giving up on an eye.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseArray, PointStamped
from scipy.spatial.transform import Rotation as Rot

from potato_scan.isaac_robot_interface import IsaacSimRobotInterface
from potato_scan.run_metrics import RunMetrics
from potato_scan.drill_task_planner import (
    plan_visit_order, approach_blocked_by_fixture, approach_candidates,
    helical_cut_path, tilt_search_sequence, ROLL_SEARCH_DEG,
    EyeAttempt, format_attempt_table)

class DrillController(Node):
    def __init__(self):
        super().__init__('drill_controller')

        # See main()'s MultiThreadedExecutor for why this exists:
        # robot_backend=isaac_sim's move_to_pose (TF) and force_drill (the
        # wrench subscription) both block in polling loops from inside
        # run_drilling, itself called from the start_drilling subscription
        # callback -- those need to run concurrently, not queued behind it.
        self._cb_group = ReentrantCallbackGroup()

        self.declare_parameter('robot_ip', '192.168.1.100')
        self.declare_parameter('robot_backend', 'rtde')  # 'rtde' or 'isaac_sim' -- see scan_controller.py
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('tcp_frame', 'tool0')  # only used by isaac_sim backend
        self.declare_parameter('drill_output_pin', 0)
        self.declare_parameter('standoff', 0.03)
        self.declare_parameter('max_depth', 0.008)
        self.declare_parameter('feed_force', 15.0)
        self.declare_parameter('max_force', 40.0)
        # Force at which the bit is considered to have touched the potato.
        # This defines depth zero for max_depth -- see
        # robot_interface.force_drill. Must sit above the free-space noise
        # floor and below max_force.
        self.declare_parameter('contact_force', 5.0)
        # How far the real surface may sit from where the eye detection put
        # it. force_drill gives up and reports 'no_contact' after feeding
        # standoff + this much without touching anything, instead of
        # pushing on into empty space.
        self.declare_parameter('surface_position_tolerance', 0.02)
        self.declare_parameter('approach_speed', 0.2)
        self.declare_parameter('approach_acceleration', 0.5)
        self.declare_parameter('drill_timeout_s', 8.0)
        # 'heuristic' (default) = sweep ROLL_SEARCH_DEG until one orientation
        # is reachable. 'rl' = a trained policy (see potato_scan/rl/ and
        # isaac/train_drill_policy.py) picks the approach roll + a small
        # lateral offset instead -- requires rl_model_path.
        # Task tolerance on the insertion axis. The drill does not have to
        # go in exactly along the surface normal: a few degrees off still
        # cuts the same eye, and allowing that turns a 1-D search (roll
        # only) into a 3-D one, which is what stops eyes being skipped as
        # "unreachable". Candidates are tried smallest-deviation-first, so
        # a tilt is only ever used when drilling straight down the normal
        # could not be reached at any roll.
        # Fixture keep-out. The potato is impaled on a pin, so a cone
        # opening from the potato's centre toward the fixture is solid
        # hardware. The reachability search cannot see it -- driving the
        # arm into a pin is perfectly reachable -- so approaches starting
        # inside this cone are rejected geometrically instead.
        # potato_center here is only a fallback: scan_controller publishes
        # the centre it actually settled on (/potato_scan/potato_center).
        self.declare_parameter('potato_center', [0.5, 0.0, 0.15])
        self.declare_parameter('fixture_axis', [0.0, 0.0, -1.0])
        self.declare_parameter('fixture_keepout_half_angle_deg', 35.0)
        # Shape of the cut. The plunge always bores to max_depth; this is
        # the pass that follows, spiralling outward as it rises so the hole
        # opens into a cone -- and since it ends at the surface, it doubles
        # as the retraction.
        #
        # 0.0 means a straight pull-out, i.e. exactly the bore the plunge
        # made, which is the behaviour this had before and the only one
        # anything has been validated against. Raising it trades a wider
        # removal for more material taken and more force, and it is the
        # natural knob for a learned policy to set per eye: eyes sit at
        # different depths in differently shaped potatoes, and the
        # literature's own failures cluster on shallow ones at the edges,
        # where a wider, shallower cut is what is wanted.
        self.declare_parameter('cut_lateral_radius', 0.0)
        self.declare_parameter('cut_turns', 2.0)
        self.declare_parameter('max_approach_tilt_deg', 15.0)
        self.declare_parameter('approach_tilt_step_deg', 7.5)
        self.declare_parameter('approach_policy', 'heuristic')
        self.declare_parameter('rl_model_path', '')

        self.standoff = self.get_parameter('standoff').value
        self.max_depth = self.get_parameter('max_depth').value
        self.feed_force = self.get_parameter('feed_force').value
        self.max_force = self.get_parameter('max_force').value
        self.contact_force = self.get_parameter('contact_force').value
        self.cut_lateral_radius = self.get_parameter('cut_lateral_radius').value
        self.cut_turns = self.get_parameter('cut_turns').value
        self.potato_center = np.array(self.get_parameter('potato_center').value, dtype=float)
        self.fixture_axis = np.array(self.get_parameter('fixture_axis').value, dtype=float)
        self.fixture_keepout_half_angle_deg = self.get_parameter(
            'fixture_keepout_half_angle_deg').value
        self.tilt_search_deg = tilt_search_sequence(
            self.get_parameter('max_approach_tilt_deg').value,
            self.get_parameter('approach_tilt_step_deg').value)
        self.max_approach_travel = (
            self.standoff + self.get_parameter('surface_position_tolerance').value)
        self.drill_timeout_s = self.get_parameter('drill_timeout_s').value

        approach_policy_mode = self.get_parameter('approach_policy').value
        if approach_policy_mode == 'rl':
            from potato_scan.rl.drill_policy_backend import RLApproachPolicy
            self.approach_policy = RLApproachPolicy(self.get_parameter('rl_model_path').value)
            self.get_logger().info(
                f"approach_policy=rl, loaded {self.get_parameter('rl_model_path').value}")
        else:
            self.approach_policy = None

        backend = self.get_parameter('robot_backend').value
        if backend == 'isaac_sim':
            self.robot = IsaacSimRobotInterface(
                self, base_frame=self.get_parameter('base_frame').value,
                tcp_frame=self.get_parameter('tcp_frame').value,
                callback_group=self._cb_group)
        else:
            # Lazy import -- see scan_controller.py's matching comment:
            # robot_interface.py imports rtde_control at module level, which
            # is not installed for robot_backend:=isaac_sim.
            from potato_scan.robot_interface import UR5eInterface
            self.robot = UR5eInterface(
                self.get_parameter('robot_ip').value,
                speed=self.get_parameter('approach_speed').value,
                acceleration=self.get_parameter('approach_acceleration').value,
                drill_output_pin=self.get_parameter('drill_output_pin').value)

        self._eyes = None  # list of (position, normal)
        # CONFIRMED risk 2026-09-16, not yet seen in the field: _on_start is
        # on the reentrant _cb_group deliberately (see that group's own
        # comment -- run_drilling's move_to_pose/force_drill need to keep
        # polling TF/the wrench topic while blocked), which also means a
        # SECOND start_drilling message arriving while run_drilling is still
        # going would enter _on_start again and run a second run_drilling
        # concurrently with the first -- two overlapping visits to the same
        # eyes, driving the arm from two places at once. This flag is
        # narrower than switching the group (which would break the
        # concurrency run_drilling actually needs): it only refuses a
        # second start while one is already in progress.
        self._drilling_in_progress = False
        self.create_subscription(
            PoseArray, '/potato_scan/eye_poses', self._on_eye_poses, 10, callback_group=self._cb_group)
        self.create_subscription(
            Bool, '/potato_scan/start_drilling', self._on_start, 10, callback_group=self._cb_group)
        self.create_subscription(
            PointStamped, '/potato_scan/potato_center', self._on_potato_center, 10,
            callback_group=self._cb_group)
        self.status_pub = self.create_publisher(Bool, '/potato_scan/drilling_complete', 10)

    def _on_eye_poses(self, msg: PoseArray):
        eyes = []
        for pose in msg.poses:
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            normal = Rot.from_quat(quat).as_matrix()[:, 2]  # +Z axis = outward normal
            eyes.append((position, normal))
        self._eyes = eyes
        self.get_logger().info(f'received {len(eyes)} eye poses')

    def _on_potato_center(self, msg: PointStamped):
        self.potato_center = np.array(
            [msg.point.x, msg.point.y, msg.point.z], dtype=float)
        self.get_logger().info(
            f'potato_center from scan: {np.round(self.potato_center, 4)} '
            '(fixture keep-out cone is measured from here)')

    def _on_start(self, msg: Bool):
        if not msg.data:
            return
        if not self._eyes:
            self.get_logger().warn('start_drilling received but no eye poses available')
            return
        if self._drilling_in_progress:
            self.get_logger().warn(
                'start_drilling received while a drilling pass is already in progress -- '
                'ignoring it rather than running a second pass concurrently with the first')
            return
        self._drilling_in_progress = True
        try:
            self.run_drilling()
        finally:
            self._drilling_in_progress = False

    def _find_reachable_approach(self, position, normal):
        """Try the default approach orientation, then sweep roll about the
        (rotationally symmetric) drill axis until one is actually
        reachable. Returns (approach_position, rotvec) of the pose the
        robot successfully moved to, or (None, None) if every roll offset
        was rejected. Returns (approach, rotvec, tilt_deg, roll_deg,
        status), where status is None on success and otherwise names why
        no insertion was attempted -- 'unreachable' (the arm refused every
        pose) or 'fixture_blocked' (every approach would have started
        inside the mounting pin). Those two call for opposite fixes, so
        they are reported apart. When approach_policy == 'rl', delegates
        to the trained policy instead (see
        potato_scan/rl/drill_policy_backend.py)."""
        if self.approach_policy is not None:
            approach, rotvec = self.approach_policy.find_approach(
                self.robot, position, normal, self.standoff)
            if approach is None:
                return None, None, 0.0, 0.0, 'unreachable'
            if self._fixture_blocks(approach):
                self.get_logger().warn(
                    'the policy-chosen approach starts inside the fixture keep-out cone')
                return None, None, 0.0, 0.0, 'fixture_blocked'
            return approach, rotvec, 0.0, 0.0, None

        tried = 0
        blocked = 0
        for approach, rotvec, tilt_deg, roll_deg in self._approach_candidates(position, normal):
            if self._fixture_blocks(approach):
                blocked += 1
                continue
            tried += 1
            if self.robot.move_to_pose(approach, rotvec):
                if tilt_deg or roll_deg:
                    self.get_logger().info(
                        f'reached on candidate {tried}: tilt={tilt_deg:.1f}deg '
                        f'roll={roll_deg}deg off the straight-down-the-normal approach')
                return approach, rotvec, tilt_deg, roll_deg, None

        if tried == 0:
            self.get_logger().warn(
                f'every one of the {blocked} approaches for this eye starts inside the '
                f'fixture keep-out cone -- the eye faces the pin. Re-seat the potato to '
                f'bring it into the open.')
            return None, None, 0.0, 0.0, 'fixture_blocked'

        self.get_logger().warn(
            f'all {tried} approach candidates rejected by the arm (rolls x tilts up to '
            f'{self.tilt_search_deg[-1]:.1f}deg); {blocked} more were inside the fixture cone)')
        return None, None, 0.0, 0.0, 'unreachable'

    def _widening_pass(self, rotvec, reached_depth_m):
        """Spiral out of the hole just bored, opening it into a cone.

        Position-controlled, unlike the plunge: force_mode holds a force
        along ONE axis, and this moves in all three. So it watches the force
        itself and stops early if the cut binds, leaving a bored hole rather
        than forcing a wider one.

        The path starts where the tool already is -- the bottom of the hole
        -- so the contact point is recovered from the current pose and the
        depth force_drill reported, rather than being tracked separately.

        Returns (waypoints_followed, stopped_on_force).
        """
        rotation = Rot.from_rotvec(rotvec).as_matrix()
        tool_z = rotation[:, 2]
        bottom, _ = self.robot.get_tcp_pose()
        contact = np.asarray(bottom, dtype=float) - tool_z * reached_depth_m

        path = helical_cut_path(contact, tool_z, reached_depth_m,
                                self.cut_lateral_radius, turns=self.cut_turns)

        for i, waypoint in enumerate(path):
            if not self.robot.move_to_pose(waypoint, rotvec):
                self.get_logger().warn(
                    f'widening pass: waypoint {i}/{len(path)} unreachable, stopping there')
                return i, False
            force = self.robot.tcp_force_magnitude()
            if force is not None and force >= self.max_force:
                self.get_logger().warn(
                    f'widening pass: {force:.1f}N at waypoint {i}/{len(path)} reached '
                    f'max_force -- stopping with the bore cut but not widened')
                return i, True
        return len(path), False

    def _fixture_blocks(self, approach_position):
        return approach_blocked_by_fixture(
            approach_position, self.potato_center, self.fixture_axis,
            self.fixture_keepout_half_angle_deg)

    def _approach_candidates(self, position, normal):
        """Delegates to drill_task_planner.approach_candidates so the pose
        convention has exactly one definition -- the calliper check aims at
        eyes through the same function."""
        return approach_candidates(position, normal, self.standoff,
                                   tilt_search_deg=self.tilt_search_deg,
                                   roll_search_deg=ROLL_SEARCH_DEG)

    @staticmethod
    def _outcome_message(idx, outcome):
        detail = (f'depth={outcome.depth_m * 1000:.1f}mm '
                  f'peak_force={outcome.peak_force_n:.1f}N')
        if outcome.status == 'reached':
            return 'info', f'eye {idx}: drilled to target depth -- {detail}'
        if outcome.status == 'force_limit':
            return 'warn', (f'eye {idx}: force limit hit before target depth -- {detail} '
                            '(lower feed_force/max_depth, or the bit is binding)')
        if outcome.status == 'no_contact':
            return 'error', (f'eye {idx}: fed the full approach travel without touching anything '
                             f'-- {detail}. The eye position/normal is wrong, or the potato moved; '
                             'this is NOT a force-tuning problem')
        return 'warn', f'eye {idx}: insertion timed out before either limit -- {detail}'

    def _report_outcome(self, idx, outcome):
        level, message = self._outcome_message(idx, outcome)
        getattr(self.get_logger(), level)(message)

    def run_drilling(self):
        positions = np.array([p for p, _ in self._eyes])
        normals = np.array([n for _, n in self._eyes])
        start_pos, _ = self.robot.get_tcp_pose()
        # normals matter to the ordering: two eyes millimetres apart can
        # face tens of degrees apart, and re-aiming between them costs far
        # more wrist motion than the distance suggests.
        order = plan_visit_order(positions, start_position=start_pos, normals=normals)
        self.get_logger().info(f'visiting {len(order)} eyes in order {order}')

        metrics = RunMetrics()
        metrics.eyes_detected = len(self._eyes)
        attempts = []
        for count, idx in enumerate(order):
            position, normal = self._eyes[idx]

            # A convex potato's surface normal should roughly agree with the
            # outward radial direction from potato_center -- cheap sanity
            # check on the normal itself, independent of whether the arm can
            # reach the pose built from it. CONFIRMED 2026-09-14 this catches
            # a real failure mode: an eye clustered from only a handful of
            # points had a normal 83.6deg off (see README's Known gaps), and
            # every roll/tilt built from it was then, correctly, unreachable
            # -- the arm and approach_candidates were not the bug.
            radial = position - self.potato_center
            radial_norm = np.linalg.norm(radial)
            radial_dir = radial / radial_norm if radial_norm > 1e-9 else radial
            normal_vs_radial_deg = np.degrees(
                np.arccos(np.clip(np.dot(normal, radial_dir), -1.0, 1.0)))
            self.get_logger().info(
                f'[{count + 1}/{len(order)}] approaching eye {idx} at {position} '
                f'normal={np.round(normal, 3)} normal_vs_radial_deg={normal_vs_radial_deg:.1f}')
            if normal_vs_radial_deg > 45.0:
                self.get_logger().warn(
                    f'eye {idx}: normal is {normal_vs_radial_deg:.1f}deg off the outward-radial '
                    f'direction from potato_center -- likely a noisy normal (few contributing '
                    f'points), not a reachability problem; an approach built from this may fail '
                    f'every roll/tilt for a reason that has nothing to do with the arm')
            with metrics.timer('approach'):
                approach, rotvec, tilt_deg, roll_deg, status = self._find_reachable_approach(
                    position, normal)
            if approach is None:
                self.get_logger().error(f'eye {idx}: {status} -- skipping')
                attempts.append(EyeAttempt(index=idx, status=status))
                continue

            task_frame = list(approach) + list(rotvec)
            self.robot.drill_on()
            try:
                with metrics.timer('drill'):
                    outcome = self.robot.force_drill(
                        task_frame, axis_index=2,
                        feed_force=self.feed_force, max_force=self.max_force,
                        max_depth=self.max_depth, timeout_s=self.drill_timeout_s,
                        contact_force=self.contact_force,
                        max_approach_travel=self.max_approach_travel)
                self._report_outcome(idx, outcome)
                attempts.append(EyeAttempt(
                    index=idx, status=outcome.status, depth_m=outcome.depth_m,
                    peak_force_n=outcome.peak_force_n, tilt_deg=tilt_deg, roll_deg=roll_deg))

                if self.cut_lateral_radius > 0.0 and outcome.contacted:
                    with metrics.timer('widen'):
                        followed, on_force = self._widening_pass(rotvec, outcome.depth_m)
                    self.get_logger().info(
                        f'eye {idx}: widening pass followed {followed} waypoints'
                        + (' (stopped on force)' if on_force else ''))

                with metrics.timer('retract'):
                    self.robot.move_to_pose(approach, rotvec)  # clear the surface
            finally:
                # CONFIRMED risk 2026-09-16, not yet seen in the field: this
                # used to be a plain call after retract, so an exception
                # anywhere from drill_on() to here (force_drill, the
                # widening pass, or the retract move) skipped it entirely,
                # leaving the drill motor energized with no code path left
                # to turn it off.
                self.robot.drill_off()

        self.get_logger().info('drilling pass complete')
        self.get_logger().info(format_attempt_table(attempts))
        metrics.attempts = attempts
        self.get_logger().info(metrics.format_report())
        self.get_logger().info(
            '  localization is reported against a hand count of the eyes actually on '
            'this potato; nothing in the pipeline can supply that, since a detector '
            'cannot report what it failed to detect.')
        self.status_pub.publish(Bool(data=True))


def main():
    rclpy.init()
    node = DrillController()
    # MultiThreadedExecutor -- see scan_controller.py's main() for why:
    # robot_backend=isaac_sim's move_to_pose (TF) and force_drill (the
    # /isaac_sim/drill_tip/wrench subscription) both block in polling loops
    # from inside run_drilling, itself called from the start_drilling
    # subscription callback -- all on this same node, so those OTHER
    # subscriptions need to run concurrently, not queued behind it.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.robot.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
