# battery_size1 first autonomous grasp bring-up

Scope: first right-arm `battery_size1` grasp and lift only. No transfer, place,
release, snap or insertion is performed by these tools.

The bring-up deliberately refuses guessed geometry. Two values must be taught
on the competition robot before autonomous descent:

1. the `wrist_a` pixel occupied by the chosen battery feature when the jaws
   are mechanically aligned over the battery;
2. the measured `tip_r` Z at the desired battery grasp height.

The dedicated calibration file is:

```
calibration/battery_size1_grasp.json
```

The repository contains only
`calibration/battery_size1_grasp.template.json`; its physical fields are
intentionally null.

## Preconditions

Verified inputs used by the tools:

- working arm: RIGHT;
- TCP: `tip_r`;
- physical right wrist camera: `wrist_a`;
- CAN gripper scope: `right`;
- grip current: 1.0 A;
- grip speed: 240 deg/s;
- task floor guard: `configs/skills/vega.json:safety.min_tcp_z_m`; this numeric
  guard remains provisional from the earlier bring-up and is **not** recorded
  as a measured right-arm battery calibration until separate `tip_r` floor
  revalidation is performed;
- vertical claw orientation is preserved from the successful wrist-centering
  hover.

The jaw goal pixel is tied to the hover height and orientation where it was
taught. The calibration records both. A later jaw-centering run refuses a
hover whose Z differs by more than 5 mm or whose orientation differs by more
than 0.04 rad.

## Shortest teaching procedure

Do this only after the board-axis benchmark and the image-center
`vega_wrist_fine_center.py` validation have passed.

### 1. Create a clean right-arm calibration record

Battery calibration schema 2 records `working_arm=right`, `tcp_frame=tip_r`,
`wrist_camera=wrist_a`, and `gripper.scope=right`. Any pre-switch,
schema-1, provenance-missing, `tip_l`, `wrist_b`, or left-gripper artifact is
stale and must not be incrementally reused.

If a calibration file already exists from earlier bring-up, reset it before
teaching either physical section:

```bash
python3 tools/vega_battery_size1_calibrate.py init --force
```

For a new path with no existing file:

```bash
python3 tools/vega_battery_size1_calibrate.py init
```

Both the jaw/hover section and grasp section must then be retaught on the right
arm. Teaching commands refuse to update a file that was not created by this
clean right-arm initialization epoch.

### 2. Teach the jaw-alignment goal pixel

At the safe low hover, use operator-supervised small XY jogs if necessary until
the OPEN jaws are mechanically centered over `battery_size1`. Do not descend
to the grasp height yet. This is a calibration operation; external visual
inspection is acceptable.

Capture the verified RIGHT wrist image while the jaws are mechanically aligned:

```bash
python3 tools/vega_battery_size1_calibrate.py \
  capture-goal-image \
  --output runs/battery_size1_goal.png
```

Inspect `runs/battery_size1_goal.png` and choose a repeatable pixel on the
battery feature that the wrist tracker will follow. Record that exact U,V while
the arm has not moved:

```bash
python3 tools/vega_battery_size1_calibrate.py \
  record-goal-pixel \
  --goal-pixel U V \
  --source-image runs/battery_size1_goal.png \
  --confirm-read-current-tcp
```

This command does not command arm motion. It reads the live TCP so the goal
pixel is bound to the measured hover Z and tip orientation. Constructing
`Robot()` can move/home the head.

Do not substitute image center unless the mechanical teaching actually proves
that image center is the jaw-alignment pixel.

### 3. Teach the grasp TCP Z

Keep XY aligned and the claw vertical. With the jaws open, descend only in
small straight-Z increments using the already-bounded jog tool, for example:

```bash
python3 tools/vega_jog_tcp.py \
  --dz -0.005 \
  --speed-scale 0.10 \
  --confirm-head-motion \
  --confirm-physical-motion
```

Repeat only until the open jaws are at the intended physical grip height on
`battery_size1`. Do not close the jaws during this teaching step.

Record the measured current TCP Z:

```bash
python3 tools/vega_battery_size1_calibrate.py \
  record-grasp-z \
  --confirm-read-current-tcp
```

The tool refuses a Z below the measured task floor or a nonvertical tip pose.
When both values are present it prints:

```
CALIBRATION_COMPLETE = True
```

Raise straight back to the same safe hover before autonomous centering. The
number of +Z jogs depends on the taught grasp Z; do not invent it.

## First jaw-centering run

The generic wrist-fine benchmark proves the servo mechanics using image center.
For an actual battery grasp, center the selected battery feature to the
**taught jaw goal pixel** instead.

Use the coarse X/Y from the already-established safe hover. For the current
board-axis workflow, this should be the final measured `REACHED CENTER` X/Y,
not a commanded/stale board estimate.

In the initial `wrist_a` hover image identify the chosen battery feature and
supply its current pixel as `--feature U V`:

```bash
python3 tools/vega_battery_size1_center.py \
  --coarse-xy X Y \
  --feature U V \
  --confirm-physical-motion
```

This tool:
- uses only `wrist_a`;
- stays at fixed Z/orientation;
- fits the local 2x2 image Jacobian from measured reversible probes;
- converges to the taught jaw goal pixel;
- never connects the gripper or descends;
- does not assert software e-stop on camera/tracking/calibration failures;
- writes a `result.json` that cryptographically identifies the calibration
  used and records the final measured TCP pose.

Keep the arm where this successful run leaves it.

## First autonomous grasp-and-lift

Immediately pass that exact result file to:

```bash
python3 tools/vega_battery_size1_pick.py \
  --alignment-result runs/<battery_size1_center_run>/result.json \
  --confirm-physical-motion \
  --confirm-battery-ready
```

The pick tool refuses to move unless:

- the calibration file is complete;
- the calibration belongs to the current robot/base frame, explicitly records
  `working_arm=right`, `tcp_frame=tip_r`, `wrist_a`, and right gripper
  scope;
- runtime config still has right arm / `tip_r` / `wrist_a=right_wrist` /
  `gripper.scope=right`;
- the gripper config is exactly 1.0 A / 240 deg/s;
- the wrist alignment result converged to the taught goal pixel;
- the alignment result was generated from the current calibration contents;
- the live TCP still matches the successful aligned TCP within 3 mm / 0.025 rad;
- the live orientation matches the taught vertical grasp orientation;
- taught grasp Z is at/above the task floor;
- the safe hover has enough clearance for the configured pregrasp.

Default sequence:

```
open jaw
  -> pregrasp at grasp_Z + 25 mm       speed 0.70
  -> final vertical descent to grasp_Z speed 0.12
  -> current-limited grip               1.0 A / 240 deg/s
  -> if driver says gripped:
       vertical lift back to original hover, speed 0.70
  -> operator records whether battery actually stayed in jaws
```

If the gripper driver does not report `gripped=true`, the tool does **not**
lift. It never places or releases the battery.

The task floor is checked before motion and after descent. All arm motion in
this tool is vertical at the already wrist-centered X/Y, with the same
orientation. Actual arm-command failures are handled by `VegaAdapter`; a
calibration/gripper/operator failure does not cause a blanket software e-stop.

## What to return after the first physical run

From the jaw-centering run:

- `REFERENCE`;
- `PROBE_X`;
- both `RETURN_REFERENCE` observations;
- `PROBE_Y`;
- `CALIBRATED`;
- every `CORRECTION` / `MOTION`;
- `COMPLETE`;
- path to its `result.json`.

From the pick run:

- `GATES_PASSED`;
- `GRIPPER_OPEN`;
- `APPROACH_REACHED`;
- `DESCENT_REACHED`;
- `GRIP` result including `stopped_by`, `peak_current`, and `position`;
- `LIFT_REACHED` if lift happened;
- `RETENTION_CHECK`;
- final `BATTERY_SIZE1 PICK RESULT`.

Also report physically whether the jaws were centered on the battery at the
taught goal, whether the final descent stayed vertical, and whether the battery
remained securely held after the lift.

## Still unresolved physically

These tools do not assume or solve the following:

- the jaw goal pixel has not yet been taught;
- battery grasp TCP Z has not yet been taught;
- the best repeatable battery feature for template tracking is not yet fixed;
- the first battery feature pixel is operator-selected during bring-up rather
  than automatically identified;
- current/stall detection proves resistance at grip time, not post-lift
  retention; retention is explicitly operator verified;
- grasp Z repeatability assumes the physical board/table geometry has not moved;
- no destination/place Z has been taught yet;
- no force-based insertion logic is involved.

Offline tests live in `tests/test_battery_size1_grasp.py`.
