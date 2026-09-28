# Vega battery_size1 coarse-hover mover

This is the motion bridge between the operator-explicit battery source localizer
and `tools/vega_battery_size1_center.py`.

It is intentionally narrow. It consumes a source-localizer `localization.json`,
revalidates that file against the **same current completed manual board
calibration**, and moves the left TCP only in base-frame X/Y while preserving
the measured live low-hover Z and quaternion.

## Safety and provenance contract

The mover refuses to run unless all of these are true:

- localization is for `battery_size1` and the configured competition robot;
- localization came from an accepted battery source-localization method;
- battery identity was operator-explicit;
- localization recorded that the board had not moved since manual calibration;
- the current manual calibration SHA256 and timestamp exactly match the
  calibration provenance stored in localization;
- the operator again confirms the board has not moved since localization;
- the current TCP is above the configured hard floor;
- the current TCP Z is already within the allowed tolerance of the manual
  calibration's corrected `CENTER` hover Z;
- the current TCP orientation matches the orientation recorded at that manual
  `CENTER` hover.

The tool does **not** claim that the recorded manual-calibration quaternion is a
new physical tool-frame calibration. It uses that quaternion only as the
accepted hover-orientation reference, then preserves the **live** quaternion
exactly during motion.

There is no vertical repositioning. If the arm is high, low, or otherwise not
already at the accepted low hover, the tool refuses instead of descending or
raising.

The tool never connects the gripper, never invokes wrist servo, never performs a
grasp descent, and has no placement or insertion path.

## Run

Use the localization file produced by
`tools/vega_battery_size1_source_localize.py`:

```bash
python3 tools/vega_battery_size1_coarse_hover.py \
  --localization runs/<battery_source_run>/localization.json \
  --confirm-board-unchanged-since-localization \
  --confirm-head-motion \
  --confirm-physical-motion
```

`--confirm-head-motion` is required because constructing the Vega SDK
`Robot()` homes/moves the head as a side effect.

Default coarse speed is 0.70. The configured task floor is enforced on the
complete segmented Cartesian move.

## Output and wrist-centering handoff

On success the tool writes an audit `result.json` and prints:

```text
BATTERY_SIZE1 COARSE HOVER PASS
MEASURED FINAL XYZ = X Y Z
CENTER ARGUMENT = --coarse-xy X Y
RESULT FILE = ...
```

Use the **measured** final X/Y, not the source-localizer's requested X/Y, for the
next gate:

```bash
python3 tools/vega_battery_size1_center.py \
  --coarse-xy X Y \
  --feature WRIST_U WRIST_V \
  --confirm-physical-motion
```

`WRIST_U WRIST_V` is the battery feature in the initial `wrist_b` image after
the coarse-hover move. It is not the head-image battery pixel.

The mover records the localization file/hash, source-image hash, manual
calibration file/hash/timestamp, operator board-unchanged confirmation, start
TCP, requested/measured motion stage, measured final TCP, and endpoint errors.
