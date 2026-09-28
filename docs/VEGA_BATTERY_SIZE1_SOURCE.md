# battery_size1 source localization from manual board calibration

This bridge is intentionally operator-assisted for the first battery pick.

It makes **no** assumptions about:
- how many parts are present;
- whether gears touch;
- source/final layout ordering;
- battery identity from spatial position;
- a 400 mm board;
- head-camera depth/extrinsics.

The operator explicitly identifies `battery_size1`.

## Representation used

The preferred first-bring-up representation is an **explicit four-corner planar
homography** from one head RGB image.

That is more reliable for this task than projecting one head pixel through the
current head-camera model because:
- the manual board calibration already gives the board center and +X/+Y axes in
  Vega base XY;
- the four corners and battery pixel come from the **same image**;
- no head intrinsics, depth, head pose or inherited camera extrinsics are needed;
- arbitrary board image rotation/perspective is handled by the homography.

The homography produces normalized board fractions. Converting those fractions
to metres requires both physical board dimensions.

Verified physical measurement:
- board width = **383 mm**.

The second dimension has **not** been supplied. The tool has no default for it.
It also does not guess whether the measured 383 mm width is manual-board +X or
+Y: `--width-axis x|y` is explicit.

If the second dimension is still unavailable, use the `board-offset` fallback
and give the battery center's measured board-relative offset directly in mm.

## 1. Manual board calibration must already exist

Expected file:

```
calibration/vega_board_manual.json
```

The source-localization tool checks:
- schema;
- competition robot name;
- base frame;
- TCP frame;
- corrected board center/+X/+Y vectors;
- reference distances;
- calibration timestamp.

Default maximum age is 720 minutes. More importantly, localization requires:

```
--confirm-board-unchanged-since-calibration
```

because software cannot detect someone physically moving the board after the
manual calibration.

## 2. Capture a head image

This capture does not construct `Robot()` and does not move the arm/head:

```bash
python3 tools/vega_battery_size1_source_localize.py capture \
  --output runs/battery_source_head
```

Output:

```
runs/battery_source_head/head_source.png
runs/battery_source_head/capture.json
```

The image is used only for explicit operator selections.

## 3A. Preferred: four-corner homography

Inspect the image and record:
- battery_size1 center pixel `BAT_U BAT_V`;
- physical board corner pixels corresponding to:
  - `xm_ym` = board -X,-Y;
  - `xp_ym` = board +X,-Y;
  - `xp_yp` = board +X,+Y;
  - `xm_yp` = board -X,+Y.

These names are **physical board-axis signs**, not screen TL/TR/BR/BL. Use the
directions established during manual board calibration.

Also determine:
- which board axis spans the measured 383 mm width: `x` or `y`;
- the measured second board dimension in mm.

Then:

```bash
python3 tools/vega_battery_size1_source_localize.py homography \
  --manual-calibration calibration/vega_board_manual.json \
  --image runs/battery_source_head/head_source.png \
  --battery-pixel BAT_U BAT_V \
  --corner-xm-ym U1 V1 \
  --corner-xp-ym U2 V2 \
  --corner-xp-yp U3 V3 \
  --corner-xm-yp U4 V4 \
  --width-axis x \
  --other-dimension-mm MEASURED_OTHER_DIM_MM \
  --confirm-board-unchanged-since-calibration
```

Use `--width-axis y` instead if that is the physical relationship established
onsite. `--board-width-mm` defaults to the verified **383**, not 400.

The tool prints:

```
BATTERY_SIZE1 COARSE BASE XY = X.XXXXXX Y.YYYYYY
CENTER ARGUMENT = --coarse-xy X.XXXXXX Y.YYYYYY
LOCALIZATION FILE = ...
```

It saves:
- a copy of the source image;
- an overlay showing the four selected corners and battery selection;
- manual-calibration path/hash;
- source-image hash;
- selected battery pixel;
- explicit physical dimensions;
- normalized board fraction;
- metric board offset from center;
- resulting base-frame XY.

## 3B. Fallback while second dimension is unknown

If the operator can physically measure the battery center relative to the
manual board center along board +X/+Y, use:

```bash
python3 tools/vega_battery_size1_source_localize.py board-offset \
  --manual-calibration calibration/vega_board_manual.json \
  --image runs/battery_source_head/head_source.png \
  --battery-pixel BAT_U BAT_V \
  --board-offset-mm DX_MM DY_MM \
  --confirm-board-unchanged-since-calibration
```

Here the selected head pixel records **which object the operator identified**,
while the explicit board-relative millimetres provide the geometry.

This mode intentionally records:
- measured width = 383 mm;
- width-axis = unknown;
- second dimension = unknown;

because neither physical dimension is needed when the operator supplies the
metric board offset directly.

## 4. Hand off to coarse arm motion

The output XY is a **base-frame coarse battery center**.

The source-localization tool performs no robot motion. The arm must next be
moved to an established safe low hover at that X/Y, preserving the vertical
claw orientation.

Important: `tools/vega_battery_size1_center.py --coarse-xy` verifies that the
TCP is already sitting near that XY. It is not a navigation command.

Once the main motion path has reached the coarse battery hover, use the measured
hover X/Y as:

```bash
python3 tools/vega_battery_size1_center.py \
  --coarse-xy X Y \
  --feature WRIST_U WRIST_V \
  --confirm-physical-motion
```

`WRIST_U WRIST_V` is the battery feature pixel in the **initial wrist_a
image at the coarse hover**. It is not the head-image `BAT_U BAT_V`.

The existing battery centering tool then converges that wrist feature to the
operator-taught jaw-alignment goal pixel.

If successful, pass its exact `result.json` to:

```bash
python3 tools/vega_battery_size1_pick.py \
  --alignment-result runs/<battery_size1_center_run>/result.json \
  --confirm-physical-motion \
  --confirm-battery-ready
```

## Remaining physical assumptions

First acquisition can still fail if:
- the manual board calibration is physically wrong;
- the board moves after manual calibration;
- the operator assigns the 383 mm width to the wrong manual board axis;
- the second board dimension used for homography is measured incorrectly;
- one of the four physical corner labels is assigned with the wrong axis sign;
- battery center is selected incorrectly in the head image;
- the resulting coarse XY is outside the right arm's reachable safe-hover region;
- head-to-board visibility does not show all four corners clearly;
- after coarse motion, `wrist_a` cannot see a trackable battery feature.

None of those are hidden behind part-count/layout heuristics.
