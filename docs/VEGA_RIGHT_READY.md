# Vega right-arm shoulder validation and manual RIGHT_READY route

On 2026-09-28 the supervised shoulder preparation step was run on
`dm/vgfcb66075ea-1u`:

```bash
python3 tools/vega_right_shoulder_step.py \
  --delta-rad 0.10 \
  --confirm-physical-motion
```

The measured state before the move was:

```text
[-2.8569905758, 0.0080302600, 0.0129189268, -2.5138814449,
  0.0899769589, 0.9653764963, -0.0006475172]
```

The measured state after the move was:

```text
[-2.7619554996, 0.0080529489, 0.0128700575, -2.5139060020,
  0.0899525210, 0.9653764963, -0.0006719517]
```

The tracked `tip_r` model predicted approximately `(+11.6 mm, 0 mm,
 +25.3 mm)` and the measured modeled pose changed by approximately
`(+11.1 mm, 0 mm, +24.0 mm)`. This validates the local positive `R_arm_j1`
direction and the tracked FK behavior for this small move.

This result does not validate a large shoulder trajectory, collision clearance,
the physical claw-to-`tip_r` transform, downward claw orientation, or the
right-arm table floor. Do not chain additional shoulder steps without a fresh
preview and operator inspection.

## Preferred next run

Use the manual-ready board calibration path. After the head image, it pauses so
the operator can manually put the right claw high above the board with the
physical jaws pointing down. It records the seven live right-arm joints and the
modeled `tip_r` pose, skips automatic post-image verticalization, and preserves
that measured quaternion for the small board calibration moves.

Pull, then run on the robot:

```bash
python3 tools/vega_board_manual_calibrate.py \
  --use-configured-right-ready \
  --forward-rise-angle-deg 0 \
  --confirm-physical-motion
```

The calibration now visits CENTER and BOARD_X_PLUS only. BOARD_Y_PLUS is
derived from the head-predicted planar Y axis and corrected CENTER, so the arm
does not have to reach the far Y reference. Add `--include-board-y-plus` only
when that physical reach is known to be safe and useful.

The output `calibration/vega_board_manual.json` contains a `right_ready` record
with `joint_names`, `joint_positions_rad`, and the `tip_r` pose. The zero rise
angle keeps the historical slope compensation disabled until a right-arm
board-parallel measurement is made.

The measured endpoints are also stored in `configs/robots/vega.json` as
`right_ready` and `right_camera_clear` joint presets. The camera-image-clear
phase uses `right_camera_clear` directly, avoiding Cartesian IK. To use the
measured ready endpoint after the head image, add:

```bash
--use-configured-right-ready
```

The preset move is still gated by the adapter's joint limits and maximum delta;
if the live state is too far away, the tool stops and asks for manual recovery
instead of inventing an IK path.

Manual board jogs also preflight the complete target. An overshoot that is not
IK-reachable is rejected before motion and returns to the jog prompt, allowing
the operator to use a smaller inverse jog without terminating calibration.

## Corner height survey

Use the supervised software-motion corner survey after the board calibration
record exists. It reuses the calibrated CENTER/axis frame, moves the robot to
TOP_RIGHT and BOTTOM_LEFT through the normal TCP planner, and pauses for the
operator to enter the measured height. It does not require manual claw control.

```bash
python3 tools/vega_board_corner_height_calibrate.py \
  --measurement-kind board_surface_z_mm \
  --confirm-physical-motion
```

The default offsets reuse the measured calibration reference distances. If a
corner target is outside the arm workspace, rerun with smaller explicit
`--x-offset-m` and `--y-offset-m` values. The tool preflights each target before
motion and writes `calibration/vega_board_corner_heights.json`.

For the board frame used by task-coordinate motion, run the five-point loop
after the initial head-camera read:

~~~bash
python3 tools/vega_board_five_point_calibrate.py \
  --confirm-physical-motion
~~~

This visits CENTER, TOP_RIGHT, BOTTOM_RIGHT, and BOTTOM_LEFT. At each point,
correct the camera estimate with the forward/back/left/right TCP jog prompt,
then enter the measured board-surface Z in millimetres in
`vega_1u_base_link`. The output fits the board surface plane and corrected
axes. Task-coordinate motion refuses to run until this schema is present.

## Operator competition pipeline

The menu entry point keeps calibration, position tests, and versioned task
experiments together:

~~~bash
python3 tools/vega_competition_pipeline.py \
  --confirm-head-motion \
  --confirm-physical-motion
~~~

Choose **Recalibrate** whenever the board has moved or rotated. The menu runs
the five-point camera/TCP/height procedure and reloads its output immediately.
The **Position tests** menu lists the corrected board references and every
published task point; selected targets are preflighted together before motion.
The **Competition task versions** menu preserves the battery and all-part
iterations as named entries. Only the calibrated battery approach-only entry
currently commands motion; later grasp/place versions remain visible scaffolds
until their approach, gripper, and verification gates are completed.

The menu can be skipped for scripted operator runs:

~~~bash
# Recalibrate directly
python3 tools/vega_competition_pipeline.py \
  --recalibrate \
  --confirm-head-motion \
  --confirm-physical-motion

# Test selected corrected references directly
python3 tools/vega_competition_pipeline.py \
  --test-positions board.center board.top_right task.battery_size1.pick \
  --confirm-head-motion \
  --confirm-physical-motion

# Run a named preserved task version directly
python3 tools/vega_competition_pipeline.py \
  --competition-task battery_size1_pick_v0 \
  --confirm-head-motion \
  --confirm-physical-motion
~~~

Use `--test-positions all` to select every board and task point. `--check-only`
can be combined with the test/task actions for local IK preflight without arm
motion.

Transient IK failures are automatically recovered. The runner closes the
current motion session, uses the validated camera-clear pose, captures a fresh
head-camera frame, and retries. Calibration repeats this startup sequence until
it succeeds or the operator presses Ctrl-C.

The final physical five-point result is preserved in
`calibration/vega_board_manual_fallback.json`. If the live calibration file is
absent, the pipeline uses that fallback automatically. The task-height planner
uses the fitted plane plus the measured local residual corrections; the final
fit residual is about 3.7 mm.

## Next productive bring-up: battery wrist centering

After updating the robot, first move to the calibrated coarse battery point:

~~~bash
python3 tools/vega_competition_pipeline.py \
  --test-positions task.battery_size1.pick \
  --confirm-head-motion \
  --confirm-physical-motion
~~~

Then initialize and teach the right wrist-camera battery calibration. The
existing tools use verified `wrist_a` and do not require manual claw control:

~~~bash
python3 tools/vega_battery_size1_calibrate.py init --force
python3 tools/vega_battery_size1_calibrate.py \
  capture-goal-image \
  --output runs/battery_size1_goal.png
~~~

Inspect that image, choose a repeatable battery feature pixel `(U,V)`, then
record it while the arm remains at the same hover:

~~~bash
python3 tools/vega_battery_size1_calibrate.py \
  record-goal-pixel \
  --goal-pixel U V \
  --source-image runs/battery_size1_goal.png \
  --confirm-read-current-tcp
~~~

For a first wrist-servo validation, use the coarse XY printed by the pipeline
and the current feature pixel:

~~~bash
python3 tools/vega_battery_size1_center.py \
  --coarse-xy X Y \
  --feature U V \
  --confirm-physical-motion
~~~

This keeps Z and orientation fixed, calibrates the local image Jacobian from
small measured probes, and stops before grasping. Only after that passes should
the grasp height be taught and `vega_battery_size1_pick.py` be attempted.
