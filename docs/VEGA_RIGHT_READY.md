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
