"""Isaac Sim scene-building helpers shared between the deployment script
(isaac_scene.py) and the two RL training envs (rl_scan_train_env.py,
rl_drill_train_env.py) -- factored out here so the simulated potato/robot/
sensor model is defined exactly once, and training never drifts from what
isaac_scene.py actually runs at deployment.

Written and reasoned about against Isaac Sim 5.1 / ROS2 Humble, WITHOUT the
ability to actually run Isaac Sim in the environment this was authored in --
same caveat as isaac_scene.py: treat this as a solid first draft, not a
verified-working module. "ADJUST:" marks the spots most likely to need
tweaking on your install.

Only import this from a script already running inside Isaac Sim's own Kit
runtime (i.e. after `from isaacsim import SimulationApp; SimulationApp(...)`
has run) -- omni/pxr/isaacsim.* aren't importable otherwise.
"""
import os
import sys

import numpy as np
import carb
from pxr import UsdGeom, UsdPhysics, Gf

# ADJUST: relative path under the Isaac asset root for the UR5e USD. If this
# raises, open the Isaac Sim Asset Browser (Window -> Browsers -> Assets) and
# search "ur5e" to find the correct path for your install, then paste it here.
UR5E_ASSET_RELATIVE_PATH = "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"

# CONFIRMED (2026-09-07, real Isaac Sim run against this same UR5e asset in
# the sibling vla_ur5e_ws project): there is no "tool0" prim on this asset
# (that's a ROS URDF-ism, not what this USD uses) -- the real tool flange
# child Xform is wrist_3_link/flange.
ROBOT_PRIM_PATH = "/World/ur5e"
TOOL_LINK_PRIM_PATH = "/World/ur5e/wrist_3_link/flange"


# The potato's geometry -- vertex grid, eye pits, bumps, and the seed that
# reproduces them -- lives in potato_scan.procedural_potato, as plain numpy,
# so the same potato Kit builds here can be generated and analysed on a
# machine with no Isaac Sim at all (see sim_detection_check). This module
# only wraps it in a USD mesh. The path insert is the same one isaac_scene.py
# and the RL envs already carry, repeated here so that this file resolves
# potato_scan even if imported first, and it is a no-op when they did.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from potato_scan.procedural_potato import (  # noqa: E402
    EYE_DEPTH_M, EYE_SIGMA_RAD, generate as generate_potato_geometry)


def make_potato_mesh(stage, prim_path, center, base_radius=0.035, bumpiness=0.35,
                      n_lat=60, n_lon=90, seed=None, geometry=None):
    """Procedural bumpy-blob mesh standing in for a real potato: a
    perturbed sphere with a handful of random low-frequency bumps
    (irregular overall shape -- exercises the shape-agnostic coverage
    grid) plus a few sharper inward pits (stand-in eyes -- exercises
    eye_detector's concavity clustering, and give the drill-approach RL
    env ground-truth targets without needing to run eye_detector during
    training). Re-seed for a different "potato" each run/episode.

    The geometry itself (what the bumps and pits are, why 60x90 vertices,
    the eye depth/width reused from the test suite) is defined once in
    potato_scan.procedural_potato.generate -- this only turns its output
    into a USD mesh. Pass `geometry` (a PotatoGeometry) to build from one
    already generated, which is how isaac_scene.py logs the seed and the
    bumps of the potato it is about to publish ground truth for; otherwise
    one is generated here from the remaining arguments.

    Returns (mesh, eye_points, eye_normals): eye_points/eye_normals are
    world-space (N, 3) arrays -- the position where each procedurally-placed
    eye pit meets the actual surface (bumps included) and the pit's outward
    axis (+Z-is-outward convention, matching eye_detector.normal_to_quat).
    """
    if geometry is None:
        geometry = generate_potato_geometry(
            center, base_radius=base_radius, bumpiness=bumpiness,
            n_lat=n_lat, n_lon=n_lon, seed=seed)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in geometry.points])
    mesh.CreateFaceVertexCountsAttr([4] * len(geometry.faces))
    mesh.CreateFaceVertexIndicesAttr([int(idx) for f in geometry.faces for idx in f])
    mesh.CreateSubdivisionSchemeAttr("none")

    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)  # static collider so the drill tip can contact it

    # Without this, PhysX's default convex-hull approximation would fill in
    # the concave eye pits (eye_dirs above), so the drill tip would never
    # actually be able to contact the "inside" of an eye even though the
    # mesh looks visually correct. The potato is a static (non-articulated,
    # non-rigid-body) collider, so a full non-convex triangle mesh
    # approximation is allowed and keeps the pits geometrically real for
    # contact purposes.
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision_api.CreateApproximationAttr("none")

    return mesh, geometry.eye_points, geometry.eye_normals


def add_drill_tip(stage, parent_path, prim_path="drill_tip", length=0.02, radius=0.0015,
                   r_parent_to_tool=None):
    """A small collider rigidly attached to the tool link, standing in
    for the physical drill bit -- gives PhysX something to report
    contact force on during force_drill's insertion.

    r_parent_to_tool: scipy Rotation taking a vector expressed in the frame
    the motion controller drives (RMPflow's "tool0") into `parent_path`'s
    own frame. MEASURED on this UR5e asset (2026-09-07): those two frames
    share a position but their orientations differ by about
    (-90, -90, 0) degrees, so "extends along tool +Z" below is only true
    once that rotation is applied -- without it the bit sticks out roughly
    sideways from the direction force_drill believes it is inserting.
    Defaults to identity for callers that genuinely mount on the tool frame.

    Deliberately does NOT apply RigidBodyAPI: drill_tip is a child prim of
    tool0 (an articulation link), so giving it its own RigidBodyAPI would
    make PhysX treat it as an independent free body rather than a fixed
    attachment -- with no fixed joint actually welding it to the arm.
    Collision alone is sufficient here: as a child prim under the tool0
    link with no rigid-body API of its own, it's treated as rigidly fixed
    to tool0 and still registers PhysX contacts for the ContactSensor to
    read.
    """
    full_path = f"{parent_path}/{prim_path}"
    cyl = UsdGeom.Cylinder.Define(stage, full_path)
    cyl.CreateHeightAttr(length)
    cyl.CreateRadiusAttr(radius)

    offset = np.array([0.0, 0.0, length / 2.0])  # extends out along tool +Z
    if r_parent_to_tool is not None:
        offset = r_parent_to_tool.apply(offset)
        q = r_parent_to_tool.as_quat()  # xyzw
        cyl.AddTranslateOp().Set(Gf.Vec3d(*offset))
        # a UsdGeom.Cylinder is Z-aligned, so the same rotation aims the bit
        cyl.AddOrientOp().Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
    else:
        cyl.AddTranslateOp().Set(Gf.Vec3d(*offset))
    prim = cyl.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    return full_path


class ContactForceReader:
    """Reads contact force at the drill tip via Isaac Sim's higher-level
    ContactSensor wrapper (a scalar normal-force reading, like a physical
    force/bump sensor -- sufficient here since IsaacSimRobotInterface only
    needs a force MAGNITUDE for its max_force check, same as the real
    UR5e's getActualTCPForce() magnitude).

    ADJUST: if isaacsim.sensors.physics.ContactSensor doesn't construct on
    your install (namespace moved again, or the sensor prim schema needs a
    slightly different setup), this prints one warning and falls back to
    always reporting zero force. That's still enough to validate the
    scan/detect/visit-order/roll-search parts of the pipeline -- for RL
    training specifically it means the drill-approach env would never see
    a real force signal, so treat a suspiciously-easy-to-train drill
    policy as a sign this fell back rather than a sign of success.
    """

    def __init__(self, prim_path, radius=0.002, suspicious_zero_streak=200):
        self._ok = False
        self._prim_path = prim_path
        # If read() keeps coming back exactly zero for this many consecutive
        # calls, warn once -- at force_drill's typical ~50Hz poll_dt=0.02s,
        # 200 reads is ~4s, comfortably longer than a real insertion should
        # go without ANY force reading if the tip is actually advancing into
        # the potato. Construction succeeding (self._ok=True below) only
        # means the ContactSensor object was created; it does NOT confirm
        # it's actually registering real PhysX contacts, which this catches.
        self._suspicious_zero_streak = suspicious_zero_streak
        self._zero_read_streak = 0
        self._warned_suspicious = False
        try:
            from isaacsim.sensors.physics import ContactSensor
            self._sensor = ContactSensor(
                prim_path=f"{prim_path}/contact_sensor",
                name="drill_tip_contact",
                # dt=-1, not frequency=60: ContactSensor.__init__ converts
                # frequency with `dt = int(1 / frequency)`, truncating to 0
                # for any frequency > 1 -- this is passed straight through as
                # the created prim's sensor_period. Isaac Sim's own shipped
                # example (isaacsim.sensors.physics.examples/contact_sensor.py)
                # uses sensor_period=-1, not 0 or a small positive dt; -1
                # reads every physics step regardless of stage units/rate.
                dt=-1,
                translation=np.array([0.0, 0.0, 0.0]),
                min_threshold=0.0,
                max_threshold=1000.0,
                radius=radius,
            )
            # Not the fix (the dt= above was) but still correct per
            # BaseSensor.initialize()'s own docstring: a prim only
            # auto-initializes on world.reset() if it was added via
            # world.scene.add(...), which this never is.
            self._sensor.initialize()
            self._ok = True
        except Exception as exc:  # noqa: BLE001 -- best-effort sim sensor setup
            carb.log_warn(
                f"ContactSensor setup failed ({exc}); drill_tip wrench will report zero force -- "
                f"force_drill will never detect contact, so every insertion aborts as 'no_contact' "
                f"once max_approach_travel is used up. Fix the sensor before reading anything into "
                f"those failures.")

    def read(self):
        if not self._ok:
            return np.zeros(3)
        frame = self._sensor.get_current_frame()
        force_mag = float(frame.get("force", 0.0) or 0.0)

        if force_mag == 0.0:
            self._zero_read_streak += 1
        else:
            self._zero_read_streak = 0
        if not self._warned_suspicious and self._zero_read_streak >= self._suspicious_zero_streak:
            self._warned_suspicious = True
            carb.log_warn(
                f"ContactSensor at {self._prim_path} has read exactly zero force "
                f"{self._zero_read_streak} times in a row. If the drill tip should be touching the "
                "potato by now, this sensor likely isn't registering real contacts (wrong prim path, "
                "sensor radius too small, or a physics substep/collision setup issue) -- NOT that the "
                "drill is genuinely floating in free space. With contact-referenced depth, force_drill "
                "reports 'no_contact' in this case rather than a clean insertion -- so treat a run of "
                "'no_contact' outcomes as a red flag to check this sensor first, before suspecting the "
                "eye poses it would otherwise be blaming.")
        return np.array([force_mag, 0.0, 0.0])


def setup_rmpflow(robot_articulation):
    """ADJUST: the exact robot-name string load_supported_motion_policy_config
    expects for the UR5e may differ from "UR5e" on your install -- if this
    raises/returns None, call get_supported_robot_policy_pairs() (same module)
    to list the exact registered names and swap it in below."""
    from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy
    from isaacsim.robot_motion.motion_generation.interface_config_loader import (
        load_supported_motion_policy_config)

    rmp_config = load_supported_motion_policy_config("UR5e", "RMPflow")
    rmpflow = RmpFlow(**rmp_config)
    physics_dt = 1.0 / 60.0
    return rmpflow, ArticulationMotionPolicy(robot_articulation, rmpflow, physics_dt)


def prim_world_pose(prim):
    from pxr import Usd, UsdGeom as _UsdGeom
    xform = _UsdGeom.Xformable(prim)
    mat = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = mat.ExtractTranslation()
    quat = mat.ExtractRotationQuat()
    quat_xyzw = [quat.imaginary[0], quat.imaginary[1], quat.imaginary[2], quat.real]
    return np.array([translation[0], translation[1], translation[2]]), np.array(quat_xyzw)
