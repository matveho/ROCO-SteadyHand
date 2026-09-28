# Day 2 Execution Plan — Vega U

## Objective

Get one autonomous scoring part working as early as possible, then generalize.

The preferred competition architecture is:

```text
head RGB
  -> board corners
  -> board frame in robot base
  -> board-relative source ROI / destination
  -> coarse arm XY
  -> left wrist camera fine centering
  -> fixed per-part grasp/place Z
  -> grip / transfer / release
```

Do not make full simulation-policy integration, head depth, perfect camera
extrinsics, or literal four-corner reach prerequisites for scoring.

## Working assumptions

- Working arm: left.
- Physical TCP/IK frame: `tip_l`.
- Keep the gripper physically vertical, like a 3D-printer nozzle.
- Normal low working plane should remain roughly 6–10 cm above the configured
  task floor except where a grasp/place descent intentionally goes lower.
- Head camera is for coarse global registration.
- Wrist camera is for local precision.
- Known board layout should be exploited. Identification can initially come
  from expected board-relative ROIs rather than general object recognition.
- The submitted `policy.py` is a deterministic simulator skill executor, not a
  learned visual policy. Its useful sequencing ideas already exist in the
  physical executor.

## Execution order

### 1. Competition procedure
Status: BLOCKED UNTIL REPRESENTATIVES ARE AVAILABLE.

Questions live in `docs/COMPETITION_QUESTIONS.md`.

Do not stop engineering work while waiting.

### 2. Identify the board in head RGB

**Live integration now implemented**

Canonical command:
```bash
python3 tools/vega_scene_perception.py
```

It:
- ensures the known `dexsensor launch --sensor head_camera` publisher is running;
- constructs the robot and commands the head to `[head_j1,head_j2,head_j3] = [0.55,0,0]`;
- reads RGB-only from the ZED left camera;
- runs the exact same board + dark-part detector as the offline tool;
- prints board corners/center plus each part center in image pixels, 400-mm board
  coordinates, and coarse `vega_1u_base_link` coordinates;
- saves raw RGB, `scene.json`, and an annotated overlay.

Use `--frames 0` for continuous re-identification. Use `--layout final` only
when the pieces are in the operator-confirmed assembled/final arrangement; the
pre-pick/source layout will get its own identity map after a fresh source-state
capture.


Status: PRIOR REAL IMAGE PASSES; fresh-image robustness check next.

Goal: robustly recover TL/TR/BR/BL and the nine dark task parts from a saved downward-looking image.

Procedure:
1. Capture one fresh downward head RGB snapshot on the robot.
2. Copy the snapshot off the robot.
3. Iterate on board detection offline against the saved image.
4. Save an overlay and machine-readable corner JSON.
5. Repeat on several images / small lighting changes before declaring robust.

Existing detector:
`steadyhand.vision.board.detect_white_board_corners`.

Offline inspection tool:
`tools/identify_board_snapshot.py`.

Current prior-image result:
- board corners: TL=(421,366), TR=(575,366), BR=(621,517), BL=(377,518);
- all 9 visible dark parts detected after rectification;
- touching gears require one board-specific merged-component split.

Exit criterion:
- all four corners visually land on the physical board boundary in several
  saved images;
- all nine task parts are boxed without false positives;
- no arm motion is required during detector tuning.

### 3. Join board coordinates to arm/base coordinates
Goal: obtain coarse `T_base_board`.

Use live camera intrinsics + head joint readback + current coarse head extrinsic.
This only needs to be accurate enough to place the wrist camera over the
correct local region.

Validate physically at:
- board center;
- one or two interior offsets.

Do not use literal outer corners as a blocking milestone.

### 4. Join task coordinates to board coordinates
Define every source ROI and destination in board coordinates.

Prefer normalized board coordinates or metres in the board frame. Do not store
raw camera pixels or raw base-frame XY as the task definition.

Deliverable:
- one table/config mapping each part to:
  - expected source ROI;
  - destination XY;
  - release type;
  - initial grasp/place height guesses.

### 5. Identify one piece
Start with one easy/open-release piece, currently `battery_size1` unless
competition scoring/procedure indicates a better choice.

Within its expected source ROI:
- segment plausible foreground/blob;
- choose centroid;
- output coarse board-relative XY.

Do not build a general classifier unless location-based identification fails.

### 6. Arm to piece
Move the vertical claw to low hover above the coarse XY.

Motion requirements:
- fixed vertical orientation;
- fixed low working Z where possible;
- smooth long Cartesian motion rather than many stop/start waypoints;
- speed scale >= 0.45 for ordinary testing and ~0.9 for clear free space.

### 7. Teach one complete part
Manual teaching is calibration, not the final method.

For the first piece record:
- wrist stream corresponding to physical left wrist;
- jaw alignment goal pixel;
- grasp Z;
- place Z;
- grip settings;
- board-relative destination;
- any required release offset.

Immediately automate this one part after teaching it.

### 8. Autonomous one-part benchmark
Target sequence:

```text
detect board
 -> find source ROI
 -> blob centroid
 -> coarse hover
 -> wrist-center
 -> descend fixed Z
 -> grip
 -> lift
 -> board-relative destination
 -> optional wrist correction
 -> descend fixed Z
 -> release
```

Exit criterion:
- one full run without operator-provided XY;
- then target 5 consecutive successful runs before broadening.

### 9. Scale to remaining parts
First expand to other open-release parts.

For each additional part, prefer configuration:
- ROI;
- detector/template parameters;
- grasp Z;
- place Z;
- grip settings.

Only precision/insertion parts should trigger additional placement correction,
search, or contact logic.

## Development discipline

- Robot time is for acquiring data and validating motion, not tuning CV.
- Save raw images from every useful vision run.
- Every vision algorithm should be runnable against saved data offline.
- Validate CLI code and targeted tests before pushing robot-facing changes.
- Treat transforms explicitly as `T_destination_source`.
- Record every verified hardware fact in `docs/FOR_LLMS.txt`.
- If a milestone does not increase scoring capability, question whether it is
  worth robot time.

## Immediate next milestone

**Physically validate board center, +100 mm board-X, and +100 mm board-Y at low vertical hover.**


## Day-2 wrist-camera blocker

Secondary live robot inspection on 2026-09-28 found no usable wrist-camera
runtime path:
- `tools/vega_wrists_probe.py` cannot import `wrist_cameras` in either system
  or Conda Python;
- only the two ZED head-camera video devices are exposed;
- documented wrist runtime/install directories are absent;
- no driver/service changes were made.

Do not burn robot time guessing at vendor camera setup. Continue head-camera
board/part localization and arm-frame integration. Ask the supplier/organizer
to confirm wrist capture-board wiring/power and the supported Sony ISX031
driver/API installation. Once restored, identify physical left/right by covering
one lens and capturing again.

Until then, the contingency path is to use head-derived board-relative part
centroids for coarse pick calibration and keep the wrist-servo interface ready
to slot in later.
