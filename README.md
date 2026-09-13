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

# the view-budget sweep: runs the real coverage grid against simulated potatoes
python -m potato_scan.scan_budget --potatoes 5

# is the camera even usable at the configured distance?
python -c "
from potato_scan.camera_ranges import check
print(check('d435', 0.15, 0.07)[1])"
```

If those run, the geometry half of the package is working and anything that
breaks later is ROS, hardware, or calibration.


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

- **Nothing here has been run against a robot or a camera.** Every number is
  from synthetic data, analytic surfaces, or stubbed backends. Step 4 is what
  changes that first, and cheaply.
- **`curvature_min` is tuned on a synthetic**, not a real potato.
- **The bottom of the potato is not scanned.** The elevation band starts at
  −15°, so eyes near the pin are never found. Deliberate — that end carries
  few eyes, and it is where the fixture is — but it is a real blind spot, and
  the fix if it matters is a re-seat and a second scan.
- **Geometry-only detection has a ceiling** that colour lifts only partly.
  The visible band is a weak tuber-versus-soil discriminator on its own: good
  on wet material, doubtful when dry. A near-infrared band is the established
  answer if soil turns out to be a real problem.
