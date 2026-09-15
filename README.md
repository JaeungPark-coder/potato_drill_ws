# potato_drill_ws

Scan a potato with an eye-in-hand depth camera on a UR5e, find its sprout
eyes, and drill them out.

```
scan (raster orbit + gap filling)  ->  detect (curvature + shape + colour)  ->  drill (force-fed, per eye)
```

## Environment

Nothing in this repository has been run against a robot or a camera. Every
number in it comes from synthetic data, analytic surfaces or stubbed
backends. This section is what to install and what to check so that the first
run on real hardware fails for real reasons rather than for setup ones.

### What you need

| | why |
|---|---|
| ROS 2 (Humble or newer) | the four pipeline nodes |
| **Python 3.12 or older** | Open3D publishes no wheel for 3.13+, and `pointcloud_accumulator` needs it |
| RealSense (or equivalent) ROS driver | publishes `/camera/depth/color/points` |
| `ur_rtde` | only for `robot_backend:=rtde`; the Isaac backend does not need it |

The Python version is the trap. Everything else in the package runs on any
recent Python — only the accumulator imports Open3D — so a 3.13+ environment
gets you a package that builds, imports most of the way, and dies on one
node.

### Install and build

```bash
# ROS-resolvable dependencies (numpy, scipy, opencv, yaml, the message packages)
rosdep install --from-paths src --ignore-src -y

# the two with no rosdep key
pip install open3d          # pointcloud_accumulator; needs Python <= 3.12
pip install ur-rtde         # only for robot_backend:=rtde

# optional, only if you intend to train the view policy
pip install gymnasium stable-baselines3

colcon build --symlink-install && source install/setup.bash
```

### Check it before you plug anything in

Most of this package is plain numpy and scipy and can be exercised with no
ROS, no robot and no camera — which is worth doing first, because it
separates "my install is wrong" from "my hardware is wrong":

```bash
cd src/potato_scan

# 166 checks over the geometry, planning and detection maths. ~45 s.
python -m pytest test/ -q

# the fast subset, if you just want to know the install is sound. ~4 s.
python -m pytest test/ -q -m "not slow"

# the view-budget sweep: runs the real coverage grid against simulated potatoes
python -m potato_scan.scan_budget --potatoes 5

# the eye detector against the Isaac Sim potato -- the same geometry
# isaac_scene.py builds, sampled at the accumulator's voxel size -- with no
# Isaac Sim. --seed N rebuilds the exact potato a scene run printed at startup.
python -m potato_scan.sim_detection_check --seeds 30

# is the camera even usable at the configured distance?
python -c "
from potato_scan.camera_ranges import check
print(check('d435', 0.15, 0.07)[1])"
```

If those run, the geometry half of the package is working and anything that
breaks later is ROS, hardware, or calibration.

**If `python -m pytest` instead dies on `ModuleNotFoundError: No module named
'lark'`, inside `launch_testing`/`launch`, not inside this package** -- that
is pytest auto-loading every installed `pytest11` plugin, ROS 2's own
`launch_testing`/`launch_ros`/`ament_lint` included, because this is normally
run in a shell with ROS already sourced (this workspace needs it for
`colcon build`). One of those imports `lark`, which ROS's apt packages don't
pull in. `setup.cfg`'s `addopts` now disables those plugins by name, so the
command above is safe to run in the same shell you built the workspace in
-- confirmed 2026-09-15 with Humble sourced and an Isaac conda env's Python
ahead on `PATH`, the actual shell this machine runs both in.

The suite is worth a second look rather than just a green tick: each test is
named for the property it holds, and several exist because the property
failed once. `test_orientation.py` holds the drill pointing into the potato
rather than away from it; `test_force_drill.py` holds depth measured from
contact rather than from the approach pose; `test_cpu_improvements.py` holds
a tilted approach still landing on the eye. If one of those goes red after a
change, the name tells you what broke before you read any code.

`test/` needs `pytest` (declared in `package.xml`, so `rosdep install`
brings it) and nothing else — no ROS, no Open3D, no robot. It is the same
suite `colcon test` runs.


## The next Isaac Sim session, in order

**2026-09-15 update: step 0 and a first pass at step 1 are done.** The
eye-pit size fix (see "Known gaps" below) took detection from 0/7 real eyes
found -- at 82% scan coverage, with the geometry actually wrong -- to 2/7
found with 0 spurious at just 43-49% coverage, and 3/7 at 67% (position
error 1.85mm mean/2.94mm worst, normal error 4.3/9.4deg -- both inside the
bars this file's own bring-up table checks against). The scan that produced
that 3/7 number was killed (exit code -9, most likely a resource squeeze
from three Isaac Sim instances running at once across two projects, not a
bug in this one) at 69-72% coverage during gap-filling, before reaching the
95% `coverage_threshold` or exhausting the 20-view gap-fill budget.
Steps 2 and 3 below have not been run at all yet.

**2026-09-15, later, off-line:** whether the remaining 4 eyes were a
coverage problem is no longer open -- see "Known gaps" (the
`sim_detection_check` entry) for the measurements. Three things came out
of running the detector against the simulated potato on a laptop, all of
which change how the next live run's report should be read:

1. **The published ground truth was inside the potato.** Eye positions
   ignored the mesh's bumps (up to 12mm of radius), so the truth sat a
   median 6mm under the surface and beyond the 8mm matching tolerance for
   37% of eyes -- each of those scored as one miss plus one spurious
   detection regardless of what the detector did. Fixed; part of the "4
   missed, 8 spurious" was very likely this, and the next run's numbers
   are not comparable to the ones above.
2. **The bumps do not make spurious eyes; the noise floor does.** The
   `bump_dirs` hypothesis below was checked over 30 potatoes and refuted.
   What produces spurious detections is kappa's 90th percentile rising
   toward `curvature_min` -- `eye_detector` logs it every run. Read it at
   43% and at 67% before touching any threshold.
3. **Recall at this threshold is ~30%, by design of the threshold, not
   the scan.** A nominal pit measures kappa 0.010-0.016 on a 1mm cloud,
   straddling `curvature_min=0.015`; which eyes clear it is decided by the
   per-potato depth/width jitter and the noise level. 3/7 is what this
   setting does on full coverage too.

### 0. Confirm the scene still starts

`isaac_scene.py` imports `potato_scan.drill_task_planner` for the one
definition of the outward-normal convention, and that import resolves only
because the file inserts its own parent on `sys.path`. It has to: this script
runs under Kit's interpreter with ROS 2 deliberately **not** sourced (see
below), so neither the workspace nor the installed package is on the path by
default. That path insert has not itself been run yet.

Launch the scene exactly as in the next section. You are looking for two
lines, in this order:

```
potato seed 1234567: 7 eyes (depth 3.50mm, sigma 5.2deg), 5 bumps
published N ground-truth eyes on /potato_scan/ground_truth_eyes
```

Write the seed down. `python -m potato_scan.sim_detection_check --seed
1234567` rebuilds that exact potato with no Isaac Sim and tags every
candidate against its true eyes and its bumps, which is how a detection the
live report cannot explain gets explained.

| what you see | what it means |
|---|---|
| `ModuleNotFoundError: No module named 'potato_scan'` | the path insert is wrong for your layout. Nothing downstream can work; fix this first |
| no `potato seed` line | the scene predates the seed logging; the potato it built cannot be reproduced off-line, so a spurious detection on it cannot be checked. Pull first |
| the scene starts but no `published N ground-truth eyes` line | the scene predates the ground-truth wiring, or `make_potato_mesh` carved no pits |
| both lines, then the usual idle | good. Leave it running and go to step 1 |

### 1. Get a real number for detection accuracy -- and this time let it finish

```bash
source install/setup.bash
ros2 run potato_scan detection_accuracy_check &
ros2 launch potato_scan potato_drill.launch.py robot_backend:=isaac_sim
```

Use `potato_drill.launch.py`, not `scan.launch.py` — only the former starts
`eye_detector` and `drill_controller`. Confirm with `ros2 node list` before
assuming a missing trigger is a bug.

A full 40+20-view raster takes tens of minutes now that views actually
settle -- confirmed 2026-09-15 to run past an hour before gap-filling either
finishes or exhausts its budget. **Run only this one Isaac Sim instance at a
time** (the 2026-09-15 kill happened with three running at once, two of
them in the sibling `vla_ur5e_ws` project); if you want an answer sooner
than a full run, publish `scan_complete` partway through as before, but a
full run is now the thing actually worth doing at least once, to see
whether the remaining eyes come in with real coverage or the geometry fix
above has its own ceiling.

```bash
ros2 topic pub --once /potato_scan/scan_complete std_msgs/msg/Bool "{data: true}"
```

`detection_accuracy_check` prints on every detection:

```
  position     : mean 2.10mm worst 4.00mm (target 2mm -- OVER TARGET)
  normal       : mean 29.0deg worst 84.2deg (warn 45deg -- OVER WARN)
```

Read the two lines separately. They are separate failures with separate
fixes, and telling them apart by hand took its own investigation on
2026-09-14:

| position | normal | what it is |
|---|---|---|
| ok | ok | detection is sound; anything left is the arm or the fixture |
| ok | OVER WARN | the 84-degree case. The eye is found, the approach direction is not. Look at `normal_consistency` in the `eye_detector` log for the same eye |
| OVER TARGET | either | the cluster centre is off. `curvature_min` / `shape_index_max` (step 4 below) |
| missed / spurious | — | a threshold problem, not a precision one. Same two parameters |

### 2. Pair the normal errors with the confidence number

`eye_detector` logs `normal_consistency` per candidate and
`drill_controller` logs `normal_vs_radial_deg` per eye. They join by eye
index. One run gives enough pairs to decide whether
`min_normal_consistency` (shipped off, in `config/params.yaml`) is worth
setting and to what — which cannot honestly be decided from synthetic data,
where the number does not predict the error at all. Not yet run: every
2026-09-15 run stopped at scan/detect, before `start_drilling` was ever
published, so there is no `normal_vs_radial_deg` to pair against yet.

### 3. Only then, the drill

```bash
ros2 topic pub --once /potato_scan/start_drilling std_msgs/msg/Bool "{data: true}"
```

Drilling has never completed in simulation: on 2026-09-14 the first two eyes
visited were both rejected as `unreachable`, and step 1 above is what says
whether that is fixed. Each eye takes about 6 s per candidate across 24
roll x tilt candidates, so a rejected eye is roughly 2.5 minutes of genuine
search, not a hang.

A `reached` outcome on one eye is the first real evidence the whole chain
works. Everything before this point has been verified; this has not.


## Running against Isaac Sim instead of real hardware

`isaac/isaac_scene.py` stands in for the robot and the camera --
`robot_backend:=isaac_sim` talks to it over the same ROS2 topics/TF a real
cell would use. It needs Isaac Sim's own Kit Python runtime, so run it with
that environment's interpreter, not the ROS2 workspace's:

```bash
ISAAC_ENV=/home/icrs/bigdisk/conda_envs/env_isaaclab
ROS2_BRIDGE_HUMBLE=$ISAAC_ENV/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble
env -i HOME="$HOME" PATH="/usr/bin:/bin" \
  ROS_DOMAIN_ID=$ROS_DOMAIN_ID \
  LD_PRELOAD=$ISAAC_ENV/lib/libstdc++.so.6 \
  LD_LIBRARY_PATH=$ROS2_BRIDGE_HUMBLE/lib \
  $ISAAC_ENV/bin/python isaac/isaac_scene.py
```

Three things about this that are not obvious, all CONFIRMED 2026-09-14 (this
was the first time this script had actually been run since it was written):

**`ROS_DOMAIN_ID` has to be passed through explicitly.** The `env -i` below
is deliberate (see the `rclpy` note next) and wipes the whole environment,
including `ROS_DOMAIN_ID` if your shell sets a non-default one (this
machine's does: 77) -- without it, this process ends up on domain 0 while
`ros2 launch` (run from a normal shell) is on whatever your shell has,
and the two halves never discover each other at all: no shared error, just
an empty `ros2 topic list`/`ros2 node list` from the other side and every
downstream TF lookup and point cloud subscription silently starved. Confirm
they match with `ros2 node list` (should show `/isaac_scene_bridge`) before
debugging anything further up the pipeline.

**`scan.launch.py` does not include `eye_detector`/`drill_controller` at
all** -- only `potato_drill.launch.py` does. If a `/potato_scan/scan_complete`
or `/potato_scan/start_drilling` you published seems to vanish, `ros2 node
list` first: it should show whichever of `/scan_controller`,
`/pointcloud_accumulator`, `/eye_detector`, `/drill_controller` your launch
file actually started, alongside `/isaac_scene_bridge`.

**Do not `source /opt/ros/humble/setup.bash` first.** Isaac Sim's Kit
runtime is built against Python 3.11; system ROS2 Humble's `rclpy` is built
against Python 3.10, and its compiled extension
(`_rclpy_pybind11.cpython-310-*.so`) does not exist for 3.11. Sourcing the
system install puts that incompatible `rclpy` on `sys.path` ahead of Isaac
Sim's own bundled one, so `import rclpy` fails with `No module named
'rclpy._rclpy_pybind11'` even though Isaac Sim ships a working Python-3.11
build of `rclpy` at `$ROS2_BRIDGE_HUMBLE/rclpy` for exactly this situation --
`enable_extension("isaacsim.ros2.bridge")` finds it on its own as long as
nothing else got there first. The `env -i` above exists specifically to
guarantee that: a shell that has ever sourced ROS2 in this session still has
it on `PYTHONPATH` otherwise.

**The first run after installing/updating Isaac Sim renders slowly, not
brokenly.** RTX shader compilation (`RtPso async group`) for this scene's
particular material/camera combination took over a minute the first time and
looked identical to a hang; it was fast (~15s to the running message) on
every run after, once cached. Give it a few minutes before assuming it is
stuck, and check the Kit log named on startup (`Logging to file: ...`) for
`Waiting for RtPso async group async compilation: N seconds so far` lines,
which do not appear on the redirected stdout/stderr this script's own prints
go to.

Stop it with Ctrl+C, or let a supervisor send SIGTERM -- either is safe as
of the shutdown fix below.

### Scoring the detector against eyes you actually know

`isaac_sim_common.make_potato_mesh` has always returned the world position
and outward normal of every pit it carves, and `isaac_scene.py` used to
discard them. It now publishes them, latched, on
`/potato_scan/ground_truth_eyes`, and:

```bash
ros2 run potato_scan detection_accuracy_check
```

scores `eye_detector`'s output against them each time it publishes:

```
detection accuracy vs ground truth
  found        : 3/3  (recall 100%)
  position     : mean 2.10mm worst 4.00mm (target 2mm -- OVER TARGET)
  normal       : mean 29.0deg worst 84.2deg (warn 45deg -- OVER WARN)

    eye 0 <- detection 0:  0.80mm  normal   0.0deg
    eye 2 <- detection 2:  4.00mm  normal  84.2deg
```

Position and normal are reported separately on purpose. The 2026-09-14 run
found six plausible-looking candidates and could not drill two of them, and
establishing that the NORMALS were wrong rather than the positions took a
separate investigation; this answers it in one run.

**This does not replace step 5.** The simulator renders depth with no sensor
noise, no specularity, and no dropout in the dark pocket an eye actually is,
so clearing the 2 mm target here is necessary and nowhere near sufficient.
What it changes is that the question no longer needs hardware to *ask*, so
thresholds (step 4) can be tuned against a number instead of against a
plausible-looking count.

## Bring-up order

The order matters. Each step's output is the next step's input, and a wrong
answer early is invisible later — the cell keeps running and produces
plausible-looking nonsense, which is how this project has lost time before.

### 1. Confirm the camera can focus this close — before anything else

`scan_radius` is measured from the potato's **centre**, and the camera images
its **surface**, a potato-radius nearer. At the configured 0.15 m around the
largest potato the fixture admits, the nearest surface sits **8 cm** from the
lens. Set `camera_model` in `config/params.yaml` and `scan_controller` checks
it at startup:

| camera | minimum | at 8 cm |
|---|---|---|
| D405 | 0.07 m | fine — built for this |
| D435 / D435i | 0.168 m (at 848×480, scales with resolution) | **fails** |
| D415 | ~0.45 m | **fails** |
| D455 | 0.40 m | **fails** |
| Gemini 335 | 0.10 m | marginal, below its optimal band |

Below the minimum, depth does not degrade — it **disappears**. An empty cloud
reads as occlusion or a bad viewpoint, not as a camera out of range, so this
failure hides. Numbers are from the manufacturers' documentation and are
resolution-dependent; bench-check yours at the working resolution.

### 2. Hand-eye calibration

```bash
ros2 run potato_scan handeye_calibration
```

Freedrive, capture 15+ views of a fixed board, paste the printed
`tcp_cam_translation` / `tcp_cam_quat_xyzw` into `config/params.yaml`.
Everything downstream is measured through this transform.

Use a **ChArUco** board rather than a plain checkerboard or plain ArUco:
chessboard-corner precision with marker robustness under partial occlusion.
Size it to fill the field of view at the working distance, and vary the
**rotation** widely between poses — translation is poorly constrained without
large rotations, which is the usual reason a calibration comes out plausible
and wrong.

### 3. Scan and detect

```bash
ros2 launch potato_scan potato_drill.launch.py
```

Watch two things in the log:

- **`curvature_min` against the logged κ percentiles.** κ is dimensionless but
  *not* scale-invariant: a denser scan gives a smaller neighbourhood, which
  reads flatter. The shipped 0.015 is validated on a synthetic at ~0.9 mm
  spacing. `eye_detector` prints the percentiles every run and warns when the
  threshold sits below the 90th.
- **`colour_contrast` per candidate.** Curvature and shape index describe a
  pit, and a clod of soil in a hollow is also a pit — colour is the only axis
  that separates them. It is measured and reported but not enforced
  (`min_color_contrast: -1.0`) until the number has been seen on real
  potatoes. On a synthetic where clods share the eyes' geometry exactly, eyes
  came out near +0.55 and clods near +0.02.

### 4. Measure where the bit actually lands — with the drill off

```bash
ros2 run potato_scan calliper_check --ros-args --params-file config/params.yaml
```

Drives the tip to each eye and asks for a calliper reading. **Nothing is
cut**, so the same potato can be measured again after every change — this is
a tuning loop, not a one-shot test. The bar is **1.84 mm** mean, the figure
the closest published system reports; eyes run 2–15 mm across, so under 2 mm
is usable and under 1 mm is comfortable.

Do this before switching the drill on. It is end-to-end — intrinsics,
hand-eye, potato centre, detection, tracking — so a bad number means
something upstream is wrong, and bisecting is cheap while nothing is being
destroyed.

### 5. Tune the force limits on one point

```bash
ros2 run potato_scan force_drill_tuner --ros-args --params-file config/params.yaml
```

Start low. The defaults are grounded in published penetration tests rather
than guessed — peak force to puncture raw flesh through the skin is 41–47 N
with a 2 mm probe on a soft cultivar, and eyes sit in the perimedullary zone,
the least resistant tissue in the tuber — but they are still a starting
envelope, not set-points. `max_force: 40.0` sits at the soft-cultivar
puncture peak and will trip early on dense ones (a needle probe on
cv. Kufri Badshah peaked at 79 N), so treat it as a per-cultivar setting.

### 6. Fit the depth collar

`drill_task_planner.depth_collar_spec()` sizes it: for a 3.25 mm bit at 40 N,
a face **8.0 mm back from the tip** and an outer diameter of at least
**5.1 mm**, which sits at a quarter of the flesh's puncture pressure.

This is a hardware ceiling on penetration that holds even when vision and
force control both err — the pattern taken from a tissue-sampling robot whose
vision computed too deep a target in 17 of 81 trials and whose punch hub,
simply wider than the blade, stopped every one at the intended depth. It
composes with force control specifically: the collar meeting skin is a force
step, so the insertion stops on `max_force`.

### 7. Drill, and report it in the field's own terms

Publish `True` on `/potato_scan/start_drilling` once the markers look right.

`run_metrics` reports success **by stage** and cycle time **by phase**,
because that is how this literature reports it and single totals are not
comparable to anything:

- **localization** — scored against a hand count of the eyes actually on the
  potato. Nothing in the pipeline can supply this; a detector cannot report
  what it failed to detect. Call `set_ground_truth()` or it stays *unknown*
  rather than being guessed.
- **removal** — of the eyes found, how many were drilled to depth.
- **overall** — both, out of the eyes present. Always the lowest of the three.
- **damage**, and **cycle time split by phase** (raster / gap-filling /
  approach / drill / widen / retract).

For a run worth publishing: a fixed *n* ≥ 30 potatoes, the success criterion
stated up front, phase timings recorded automatically, success and damage
recorded by hand.

## What is still unverified, and what verifies it

Ordered so that each step's failure is cheap and interpretable. Do not skip
ahead: a wrong answer early does not stop the pipeline, it makes the next
step produce plausible nonsense.

| # | what | needs | how you know it worked |
|---|---|---|---|
| 1 | Camera minimum range | the model number | `camera range:` line at scan startup is `info`, not `error` |
| 2 | Hand-eye calibration | board + robot | reprojection residual printed by the tool; then step 4's number |
| 3 | Scan geometry | robot + camera | coverage reaches threshold without a pile of `unreachable` warnings |
| 4 | Detection thresholds | real scans | eye count lands in 2–15, κ threshold above the 90th percentile |
| 5 | **Position accuracy** | robot + callipers | `calliper_check` mean under ~2 mm |
| 6 | Force limits | robot + potatoes | `force_drill_tuner` reaches depth without tripping `max_force` |
| 7 | Depth collar | 3D printer | insertion stops on `max_force` when the collar meets skin |
| 8 | End-to-end rates | ≥30 potatoes | `run_metrics` report with ground truth supplied |

Steps 1 and 8 bracket everything: 1 costs nothing and invalidates the rest if
wrong, and 8 is the only thing that produces numbers comparable to published
systems.

**Step 5 is the one to get to quickly.** It runs with the drill off, so
nothing is destroyed and the same potato can be measured again after every
change — which turns steps 2–4 from guesswork into a loop with a number at
the end of it.


## When something goes wrong

Messages below are quoted as the nodes actually print them.

### Build and startup

| you see | it means | do |
|---|---|---|
| `ModuleNotFoundError: open3d` | Python 3.13+, or Open3D not installed | `python -V`; if 3.13+, build the workspace against 3.12 or older |
| `ModuleNotFoundError: rtde_control` | `ur_rtde` missing | `pip install ur-rtde`, or use `robot_backend:=isaac_sim` |
| node dies immediately on `import numpy`/`scipy` | `rosdep install` not run | run it; `package.xml` declares them |

### Scanning

| you see | it means | do |
|---|---|---|
| `camera range: ... cannot focus this close` | the camera's minimum exceeds the distance to the potato's **surface** | change `camera_model` if it was wrong, else raise `scan_radius` to the value the message names, or fit a close-range camera |
| `camera range: ... is not in the table` | unknown model — **not** a pass | look up its minimum at your working resolution and compare against the distance in the message |
| `the camera topic carries no rgb field` | subscribed to a depth-only topic | point `camera_topic` at `/camera/depth/color/points` |
| many `view direction=... unreachable at every camera roll` | the orbit is outside the arm's envelope | check `potato_center` against where the potato actually is, and `scan_radius` against the workspace |
| `cell (e,a) still empty after N recovery attempts` | persistent occlusion at that direction | expected for a few cells near the fixture; a lot of them means the fixture or gripper is in the way |
| `potato_center fit landed Nmm from the configured ...` | the sphere fit latched onto fixture or background | fix the configured `potato_center`; the fit refines, it does not search |
| coverage stalls well short of threshold | too few views, or the camera is marginal | run `scan_budget` with your `--scan-radius`; if the geometry says it should cover, suspect the camera |

### Detection

| you see | it means | do |
|---|---|---|
| `curvature_min=... sits below this scan's 90th percentile` | the threshold is set for a different point density | raise it toward the p99 the same line prints |
| `N eyes is outside the 2-15 a potato plausibly has` | thresholds wrong for this scan, not an unusual potato | check the percentile line above it first |
| `merged cloud has no colour` | the accumulator got a depth-only topic | see the `rgb` row above |
| plausible eye count, wrong places | soil clods read as pits — geometry cannot separate them | look at the logged `colour_contrast`; if eyes and clods separate, set `min_color_contrast` between them |
| `not enough points for eye detection` | the scan produced almost nothing | a scanning problem, not a detection one |

### Drilling

| you see | it means | do |
|---|---|---|
| `eye N: no_contact -- skipping` | fed the whole approach travel touching nothing | the eye pose or `potato_center` is wrong, **or the potato moved**. Not a force-tuning problem — raising `feed_force` cannot help |
| `eye N: force_limit` | hit `max_force` before depth | `max_force: 40.0` sits at a soft cultivar's puncture peak; dense ones need 60–90 N. Check the bit is sharp before raising it |
| `eye N: fixture_blocked` | every approach starts inside the pin's keep-out cone | re-seat the potato so that eye faces out. Widening the cone does not fix it, it just lets the arm hit the pin |
| `eye N: unreachable` | the arm refused every roll **and** tilt | a kinematics problem: check fixture placement, and whether `max_approach_tilt_deg` is larger than 0 |
| `reached` but nothing was removed | depth or cut shape | `max_depth` is 8 mm from **contact**; if the hole is right but the eye stays, try `cut_lateral_radius` |
| `widening pass: ... reached max_force` | the cut bound up | expected on a wide `cut_lateral_radius`; the bore is still cut, so it is a warning, not a failure |

### The one that hides

A run that finishes cleanly and reports success is still worth doubting until
step 5 has given you a number. Every expensive failure this project has had
looked exactly like a clean run: a well-formed dataset in which the target
never appeared, a drill that reported reaching depth while 15 mm clear of the
potato, a wrist driven through the potato's own volume. The guards that now
catch those are in the code, but the calliper number is what confirms the
whole chain rather than each link.


## Choices worth knowing about

**The cut shape is a parameter, not a decision.** `cut_lateral_radius: 0.0`
is a straight bore, exactly what the plunge cuts. Larger opens a cone on the
way out, and because that path ends at the surface, the widening pass *is*
the retraction. Bore-versus-scoop is therefore a number to measure or learn
rather than a design choice to make up front.

**Depth is measured from contact, not from the approach pose.** They differ
by `standoff`, and measuring from the approach pose meant a `max_depth`
smaller than `standoff` reported success with the bit still that far clear of
the potato, having touched nothing.

**The tool points into the surface.** `+Z = -normal`. A UR5e's wrist extends
back along the tool's `-Z`, so commanding `+Z = +normal` asks the wrist to
occupy the potato's own volume — measured at 0.0 mm clearance against a 35 mm
radius, versus 65 mm the right way round.

**The potato is on a pin, and the reachability search cannot see it** —
driving the arm into a pin is perfectly reachable. A keep-out cone about
`fixture_axis` rejects those approaches geometrically and reports them as
`fixture_blocked`, which is a re-seat-the-potato problem rather than the
force-tuning problem `unreachable` would suggest.

## How many scan views does it need?

```bash
python -m potato_scan.scan_budget --potatoes 8
```

Sweeps the raster's step sizes against coverage on procedurally generated
potatoes, using the same coverage grid the robot uses and a sampled
visibility model (`potato_surface.py`) — no robot, no simulator, no GPU.

**Twelve views reach full coverage on every potato tried; the shipped default
is 40.** Nine reach 99.7%, six fall to 85.8%. The scan's cost is dominated by
its view count, so that is a 3.3× reduction in the dominant term — but the
model is *geometric*. No sensor noise, no specularity, no dropout in the dark
pockets an eye actually is, no arm to refuse a pose. Every one of those argues
for more views than the geometric minimum, so read 12 as the floor and the
other 28 as a robustness margin worth choosing deliberately rather than
inheriting from round numbers.

The same sweep says something about the **Phase B / RL hook**: at 40 views
the raster already saturates, so there is no gap left to fill and
`rl/cpu_scan_env.py` has no problem to pose. That is structural — a
star-shaped surface has essentially no self-occlusion once grazing views are
discarded, and a dent is still star-shaped. The policy only has work below
about nine views. Where it would actually earn its place on the real cell is
on effects this model omits, not on geometry.

## Known gaps

- **Nothing here has been run against a REAL robot or camera.** Every number
  is from synthetic data, analytic surfaces, or stubbed backends. Step 4 is
  what changes that first, and cheaply. `isaac/isaac_scene.py` (the Isaac Sim
  stand-in) has now been run for the first time, and the scan half of the
  pipeline verified against it end-to-end (real accumulated point cloud ->
  real coverage/centre estimate) -- 2026-09-14, see "Running against Isaac
  Sim instead of real hardware" above and the fixes below.
- **`isaac_scene.py` crashed on shutdown -- found and fixed 2026-09-14.** A
  plain SIGTERM (e.g. from a process supervisor, or `timeout` in a smoke
  test) is not a `KeyboardInterrupt`: rclpy installs its own SIGTERM handler
  that calls `context.shutdown()` from inside the signal handler, racing
  whatever line happened to be executing. That left `rclpy.spin_once()`
  mid-call raising `RCLError` ("the given context is not valid"), which the
  old `except KeyboardInterrupt` did not catch -- so it propagated straight
  through the `finally` block's `rclpy.shutdown()` (raising a second,
  already-shut-down error) before ever reaching the Replicator cleanup below
  it, and `simulation_app.close()` then tore down the extension stack with a
  live pointcloud annotator and render product still attached -- a native
  SIGSEGV in `omni.graph.core.plugin`/`omni.syntheticdata.plugin` during
  `Py_FinalizeEx`, not a Python-catchable error. Fixed by catching `Exception`
  broadly around the loop and running each cleanup step
  (`destroy_node`/`rclpy.shutdown`/`annotator.detach`/`render_product.destroy`)
  independently in its own `try`, so one failing step can no longer skip the
  ones after it.
- **The drill tip's contact sensor never left its constructor default --
  found and fixed 2026-09-14.** `ContactForceReader.read()` kept reporting
  zero force no matter how the tip was driven; a standalone probe (query
  `isaacsim.sensors.physics._sensor`'s interface directly, bypassing this
  project's wrapper) showed `get_sensor_reading(...).is_valid` was `False`
  on every single call, so `get_current_frame()` was silently leaving the
  whole dict at `{"time": 0, "physics_step": 0}` forever rather than raising
  -- indistinguishable from "genuinely no contact" unless you check
  `is_valid` yourself. Root cause: PhysX parses contact-report registrations
  when the physics scene (re)starts, and `ContactSensor`/`add_drill_tip` are
  both constructed well after `isaac_scene.py`'s first `world.reset()` --
  so PhysX never saw either one. A second `world.reset()` +
  `robot.initialize()` right after `ContactForceReader(...)` is constructed
  (after every physics-relevant prim for the episode already exists) is what
  actually binds it; neither a `dt=`/`sensor_period=` change nor manually
  applying `PhysxContactReportAPI` to the rigid-body link were needed once
  that reset was in place. `rl_drill_train_env.py`'s `IsaacDrillEnv` looks
  like it already gets this for free by construction -- `__init__` builds
  `ContactForceReader` once, and Gymnasium always calls `reset()` (which
  does its own `world.reset()`/`robot.initialize()`) before the first
  `step()`, so the required second reset already happens -- but this is
  read from the code, not confirmed live: a standalone check crashed on
  startup from running a second concurrent Isaac Sim instance alongside the
  one already up for the `isaac_scene.py` testing above, not from anything
  in the script itself. Worth a clean, isolated re-check before trusting it.
- **Three files misread `PointCloud2` messages -- found and fixed
  2026-09-14, confirmed against `isaac_scene.py`'s real published cloud, not
  a mock.** `pointcloud_accumulator.py`, `scan_controller.py`, and
  `eye_detector.py` all did
  `raw = np.array(list(pc2.read_points(msg, field_names=(...), skip_nans=True)))`
  then sliced it as a plain `raw[:, :3]` -- but `read_points` returns a
  STRUCTURED array (one named dtype field per requested name), which is
  1-dimensional; indexing a second axis raises `IndexError: too many indices
  for array`, and casting the whole structured array to `float` (as
  `surface_coverage.estimate_center` did downstream) raises `TypeError:
  Cannot cast array data from dtype([('x','<f4'),...]) to dtype('float64')`.
  Every one of these was reached and crashed the owning node the first time
  it processed a real cloud. Fixed by indexing fields by name
  (`np.column_stack([raw['x'], raw['y'], raw['z']])`) instead of by position.
- **`scan_controller.py`/`drill_controller.py`/`calliper_check.py` crashed
  immediately under `robot_backend:=isaac_sim` -- found and fixed
  2026-09-14.** All three `import`ed `potato_scan.robot_interface` (which
  imports `rtde_control` at module level) unconditionally at the top of the
  file, even though the constructor only reaches that branch for
  `robot_backend:='rtde'`. Without `ur_rtde` installed -- correct per this
  README's own "Isaac backend does not need it" -- every one of the three
  died on startup with `ModuleNotFoundError: No module named 'rtde_control'`
  regardless of which backend was actually requested. Fixed by moving the
  `UR5eInterface` import into the `else` (rtde) branch in each file.
- **First successful scan against Isaac Sim, 2026-09-14, after the fixes
  above.** `ros2 launch potato_scan scan.launch.py robot_backend:=isaac_sim`
  against a running `isaac_scene.py`: `potato_center` converged to within a
  few mm to ~2.5cm of the configured value from real accumulated points
  (see the drift note below), and `estimated_potato_radius` converged near
  the mesh's actual ~35-57mm range.
- **Almost every "unreachable at every camera roll" view was a false
  positive from a concurrency bug, not a real workspace limit -- found and
  fixed 2026-09-14.** Initially looked exactly like the README's own
  documented explanation (orbit outside the arm's envelope): coverage
  plateaued around 53% with dozens of views across every azimuth failing.
  Debug instrumentation in `IsaacSimRobotInterface.move_to_pose` showed
  each failed attempt gave up at the full 6s `settle_timeout_s` with
  `pos_err`/`rot_err` nowhere near converging (5-27cm / 68-121deg,
  different every time) -- not the small, consistent residual a genuinely
  out-of-reach pose leaves. Cause: `ScanController`'s 0.5s `_run_step`
  timer was registered on the same `ReentrantCallbackGroup` as its
  subscriptions. `ReentrantCallbackGroup.can_execute()` returns True
  unconditionally, so the `MultiThreadedExecutor` re-armed and re-entered
  `_run_step` every 0.5s even while the previous call was still blocked
  inside `move_to_pose`'s up-to-6s wait -- multiple concurrent `_run_step`
  calls kept publishing different Cartesian targets to the same robot, so
  nothing ever got the several seconds RMPflow needs to actually converge
  (see vla_ur5e_ws's near-identical bug and fix, same root cause: a timer
  and its own blocking work sharing a reentrant group). Fixed by giving the
  timer its own `MutuallyExclusiveCallbackGroup`, leaving the TF/point-count
  subscriptions on the reentrant one so they keep refreshing while
  `_run_step` blocks. Effect measured directly: 40+ spurious failures
  covering every azimuth dropped to 1 genuine one, and a single raster view
  now reaches 24% coverage on its own (versus many thrashing "views" needed
  to reach a similar number before, none of which had actually settled).
  One side effect worth watching, not yet explained: `potato_center`
  drifted a real ~26mm from the configured value over the course of the
  first two (now properly-settled) views, plateauing rather than continuing
  to grow -- possibly expected re-fitting behavior as more of the surface
  is actually seen, possibly its own bug; not yet distinguished.
- **First successful full-pipeline run, 2026-09-14: scan -> detect -> drill
  chain confirmed end to end, drilling itself still blocked.** `ros2 launch
  potato_scan potato_drill.launch.py robot_backend:=isaac_sim` (not
  `scan.launch.py`, which does not include `eye_detector`/`drill_controller`
  at all -- confirm with `ros2 node list` before assuming a missing trigger
  is a bug). Manually publishing `/potato_scan/scan_complete` partway through
  a scan (a full 40+20-view raster is tens of minutes now that views
  actually settle -- see below) fed a real, partial merged cloud to
  `eye_detector`, which found 6 candidate eyes (3.5-13.6mm diameter, inside
  the plausible 2-15mm range) and `drill_controller` received all 6.
  Publishing `/potato_scan/start_drilling` then drove `_find_reachable_approach`
  for real: the first two eyes visited each exhausted all 24 roll x tilt
  candidates and were rejected as `unreachable`, at the genuine ~6s-per-candidate
  rate settle_timeout_s implies (not the sub-second thrashing the
  scan-side bug produced, so this is not that same bug back).
- **Root cause of the unreachable-approach eyes -- found 2026-09-14: the
  detected NORMAL was ~84 degrees off from the true outward direction, not
  a `drill_controller`/`approach_candidates` bug.** Added a one-line check
  (`normal_vs_radial_deg`, comparing the detected normal against
  `position - potato_center`, which a genuinely convex potato surface point
  should roughly agree with) and re-ran: the rejected eye's normal was
  83.6 degrees off, computed from a cluster of only 8 points. An approach
  built from a normal that wrong asks the arm to insert roughly TANGENT to
  the surface instead of into it -- unreachable at any roll/tilt, not
  because the arm or the approach-candidate math is wrong, but because the
  input it was given was. `drill_task_planner.approach_candidates` is
  cleared by this: it correctly turned a bad normal into a correctly
  unreachable pose. The real fix is upstream, distinguishing a real eye's
  normal (many points, low estimation variance) from a noise direction
  computed from too few -- not yet built; `cluster_min_points` (currently 8,
  the same value that just produced this) is the parameter to look at, and
  the fact that low-point clusters correlate with real drilling failures is
  now measured, not assumed.
- **Separately, `eye_detector` had no way to tell a real eye from anything
  else in the scene with a matching local shape -- found and fixed
  2026-09-14.** Against the same real cloud above, 4 of 11 "eyes" clustered
  near [0.04-0.07, 0.03-0.07, ...] -- nowhere near the potato (potato_center
  was ~[0.48, -0.01, 0.16]) -- because the ROBOT'S OWN BASE happened to
  satisfy the same curvature+shape-index window `find_eye_candidates` gates
  on. `potato_center` was already a parameter, but only ever used to orient
  normals (via `describe_surface`) -- it never restricted which points were
  even considered. Added `max_center_distance` (`find_eye_candidates`) /
  `max_expected_radius` (`eye_detector`'s ROS parameter, reusing
  `scan_controller`'s own name and default of 0.07m -- "larger than the
  largest potato you'd ever load" already meant exactly what this filter
  needed) to reject any candidate whose position lands farther than that
  from `potato_center`. Regression-tested with a synthetic lookalike placed
  far from the potato (`test_surface_curvature.py`,
  `test_max_center_distance_rejects_a_lookalike_far_from_the_potato`) --
  building it took care to place its dimple on the side facing away from
  `potato_center`, since a dimple facing TOWARD the wrong reference gets its
  normal auto-orientation flipped and disappears as a dome instead of a
  cup, which is itself the SAME degradation as the finding above, just
  encountered while constructing the test rather than in the field.
- **A full scan is now a multi-minute run, not the sub-minute one the
  pre-fix thrashing made it look like.** Properly-settled views take several
  real seconds each (RMPflow settle + point cloud settle), so the full
  40-view raster plus up to 20 gap-fill views, and then up to 24 approach
  candidates per eye in drilling, add up fast -- budget accordingly rather
  than assuming a long-running scan or drill attempt is stuck.
- **`curvature_min` is tuned on a synthetic**, not a real potato.
- **The bottom of the potato is not scanned.** The elevation band starts at
  −15°, so eyes near the pin are never found. Deliberate — that end carries
  few eyes, and it is where the fixture is — but it is a real blind spot, and
  the fix if it matters is a re-seat and a second scan.
- **Geometry-only detection has a ceiling** that colour lifts only partly.
  The visible band is a weak tuber-versus-soil discriminator on its own: good
  on wet material, doubtful when dry. A near-infrared band is the established
  answer if soil turns out to be a real problem.
- **`isaac_scene.py`'s procedural potato was carving eyes ~39mm across --
  found, root-caused and fixed 2026-09-15, running the "next Isaac Sim
  session" bring-up steps above against a live scene for the first time.**
  A real scan (10 raster views, 82% coverage, 8.9M merged points) came back
  with `eye_detector detected 0 potato eyes` -- and re-running with 3x more
  points changed nothing, which ruled out "not enough scan" before any
  threshold was touched. Pulling the merged cloud and `isaac_scene`'s own
  ground truth out via a throwaway subscriber and checking kappa AT each of
  the 7 real eye positions directly (not at whatever `find_eye_candidates`
  happened to cluster) showed why: kappa there topped out at 0.0001-0.0088,
  2-150x below `curvature_min=0.015`, unmoved by the extra points. Reading
  `make_potato_mesh`'s own eye formula found the actual cause: `dot > 0.85`
  is a 31.8-degree cone, which at `base_radius=35mm` carves a **~39mm-diameter**
  pit -- an order of magnitude past the 2-15mm `min/max_eye_diameter` the
  rest of the pipeline (this file's own force_drill_tuner section, the
  depth-collar sizing for a 3.25mm bit) is built around. Curvature over a
  30-point neighbourhood reads nearly flat on a bowl that wide; the two
  candidates that DID clear the filters each run (diameter 4.6-10mm, distance
  50mm+ from every real eye) were something else on the surface entirely,
  which is exactly what `detection_accuracy_check` reported them as:
  `spurious`, not a near-miss.

  Fixed by replacing the `dot > 0.85` linear-ramp cap with the SAME Gaussian
  dimple profile (`EYE_DEPTH_M=0.0035`, `EYE_SIGMA_RAD=0.09`, jittered
  ±15-30% per potato) this project's own `test_surface_curvature.py` already
  validates detection against at this exact `base_radius` -- reusing proven
  numbers rather than guessing new ones. That alone would carve a
  correctly-sized pit invisible to the mesh, though: with
  `SetSubdivisionSchemeAttr("none")`, every face renders perfectly flat, so
  curvature only ever appears at a vertex, and the original 24x36 grid
  (864 vertices, ~5mm spacing) can leave a 2-15mm pit with zero vertices
  inside it. `n_lat`/`n_lon` defaults raised to 60x90 (5400 vertices, ~2mm
  spacing) so a realistically-sized eye still spans several.

  **Confirmed on a fresh scene with 3 fewer raster views than the failing
  run (3 views, 43-49% coverage vs. the prior 10 views/82%):** 2/7 eyes
  found, 0 spurious, position error 1.05/3.19mm, normal error 1.9/12.9deg
  (both -- worst case included -- comfortably inside the 2mm/45deg bars
  bring-up step 1 checks against). Re-running detection again at 61%
  coverage found the same 2 and no more, which reads as "the other 5 eyes'
  neighbourhood isn't scanned yet" rather than a detection failure -- the two
  that ARE covered are found accurately and every run, zero false positives.
  Whether the remaining 5 come in with fuller coverage, and whether the
  now-working detector changes anything at bring-up steps 2-3 (pairing
  `normal_consistency` against `normal_vs_radial_deg`, then drilling), is
  what running the rest of "The next Isaac Sim session" order will show --
  not yet done as of this note.

  **Update, same session, 67% coverage (40 raster views + partial
  gap-filling):** 3/7 real eyes now found, position error down to 1.85mm
  mean / 2.94mm worst (inside the 2mm bar), normal error 4.3deg mean /
  9.4deg worst -- the fix holds up as more of the surface is scanned. But
  **8 spurious detections appeared where there were 0 at 43-49% coverage**,
  all high-confidence by the numbers that are supposed to separate a real
  eye from noise (`normal_consistency` 0.88-0.997, `shape_index` 0.18-0.25 --
  as cup-like as the real eyes). They cluster spatially (centroid ~34mm from
  `potato_center`, ~20-30mm spread) on the side of the potato away from the
  detected real eyes, rather than scattering evenly across the surface,
  which points at a specific cause rather than generic noise: `bump_dirs`
  (the mesh's 4-6 random low-frequency bumps, unrelated to the eye pits) can
  leave a genuinely concave valley where two bumps meet, and raising mesh
  resolution to resolve 2-15mm eye pits (this same fix) would have made
  those valleys resolvable too, for the first time. **Not confirmed** --
  `isaac_scene.py` doesn't currently expose `bump_dirs` to check candidate
  positions against them directly -- but the spatial clustering argues
  against plain sensor/reconstruction noise. If this holds up, the fix is
  likely on the bump side (lower `bumpiness`, or keep bumps low-frequency
  enough that adjacent ones can't create a sub-15mm concave valley) rather
  than the eye side touched here. Worth confirming with `bump_dirs` logged
  before spending time tuning `curvature_min`/`shape_index_max` against it.

  **Checked off-line, same day, and it does not hold.** The potato's
  geometry now exists outside Kit (`potato_scan/procedural_potato.py`: the
  identical random draws in the identical order, so a seed reproduces the
  potato Isaac built, plus a sampler that stands in for the depth camera
  at the accumulator's 1mm voxel), which made the hypothesis testable
  with the real `find_eye_candidates` and the real scoring, over as many
  potatoes as wanted (`python -m potato_scan.sim_detection_check`). With
  the eye pits carved AND with them removed, noise off, 30 potatoes
  produced **0 spurious candidates**: the bumps on their own make nothing
  the detector accepts. What does is the noise floor. kappa is a
  covariance ratio, so isotropic noise raises it everywhere: at 0.15mm
  its 90th percentile is ~0.007 and spurious stays 0; at 0.20mm, ~0.012
  and 18 appear across 30 eyeless potatoes; at 0.25mm, ~0.018 -- past
  `curvature_min` -- and every point on the potato is a candidate (2232
  spurious). A merged cloud that thickens as more views land on the same
  surface (TF residual, per-view registration) is indistinguishable from
  rising noise to this detector, which fits 0 spurious at 43-49% coverage
  and 8 at 67% without any valley. `eye_detector` already logs the kappa
  percentiles per run: the 67% run's p90 against the 43% run's is the
  number that decides it.

  Two more things fell out of the same tool. **The ground truth
  `isaac_scene.py` publishes was inside the potato:** `make_potato_mesh`
  placed each eye at `base_radius - eye_depth` along its direction,
  "ignoring the smaller bump contribution", and it was not small -- bumps
  add up to `bumpiness * base_radius` = 12mm of radius, so over 200 seeds
  the truth sat a median 6.2mm under the surface and beyond the 8mm
  matching tolerance for 37% of eyes. Each of those scored as one missed
  eye plus one spurious detection however well the detector did, and
  scoring the same detections against the corrected truth moved the
  position error from ~4.0mm mean/7.9mm worst to ~0.9/1.5mm. Fixed: the
  truth is now where the pit meets the actual surface, and `isaac_scene`
  builds its mesh from the same geometry object it publishes truth from,
  so the two cannot drift again. The normal stays the pit's axis, by
  measurement: the detector's normal tracks it at 6.0deg mean / 11.7deg
  worst on noiseless clouds versus 9.2 / 29.2 for the surface normal of
  the flank the pit sits on.

  And **recall is set by `curvature_min`, not by coverage.** A nominal pit
  (3.5mm deep, sigma 0.09rad) on a noiseless 1mm cloud measures kappa
  0.010-0.016 at its most curved -- straddling 0.015 -- so which eyes
  clear it comes down to the per-potato depth/sigma jitter (recall ran 6%
  for the shallowest-widest potatoes to 100% for the sharpest) and to the
  noise (13% with none, 32% at 0.15mm, 55% at 0.20mm, then the cliff
  above). 3/7 found at 67% coverage is what this threshold does at full
  coverage too; the remaining eyes were not unscanned, they were under
  the bar. This is the density dependence `surface_curvature`'s docstring
  already warns about, now with the number attached: on this cloud the
  detector is operating within a factor of ~1.5 of its own threshold on
  both sides, which is why a modest change in noise flips it from missing
  most eyes to accepting everything. Not changed here -- which gate
  replaces it is a detector design decision, and the noise level of a
  real camera is the input it needs -- but `--curvature-min` on
  `sim_detection_check` sweeps it in seconds, which is where to start.
