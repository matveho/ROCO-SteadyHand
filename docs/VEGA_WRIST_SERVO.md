# First wrist XY loop on competition Vega

Use the system interpreter onboard. This benchmark uses the left arm, tip_l,
fixed hover Z and fixed orientation. It never connects or homes the CAN gripper.
`--confirm-physical-motion` also acknowledges dexcontrol Robot() head homing.

Pull/deploy main first, then run in `~/ROCO-SteadyHand-live`.

1. Reach only board center (skip this if already there and vertical):

   ```bash
   python3 tools/vega_board_benchmark.py --reuse-registration --center-only --execute --confirm-physical-motion
   ```

2. Capture both live wrist views, without constructing Robot():

   ```bash
   python3 tools/vega_wrist_servo.py
   ```

   It prints a new `runs/wrist_servo_<UTC>/` directory containing
   `000_wrist_a.png`, `000_wrist_b.png`, and frame metadata. The competition
   mapping is verified: `wrist_a=RIGHT`, `wrist_b=LEFT`. Startup frames can
   be black after boot; the tool rejects near-black frames. The camera wrapper
   uses `wrist_cameras.WristCameras.get_obs(fresh=True)`; no depth, TensorRT,
   or precise wrist extrinsics are required.

3. Run the local loop using the verified left wrist (wrist_b is the default):

   ```bash
   python3 tools/vega_wrist_servo.py --execute --confirm-physical-motion
   ```

   The initial feature is a textured corner near image center. To track a
   specific mark or part, append `--feature U V` using its pixel coordinates in
   the current hover image. Choose a stationary board/part feature, not the
   gripper. If the tool needs a coarse approach first, append `--move-to-board`;
   it loads the saved registration and plans only center. `--board-xy X Y`
   overrides that coarse approach with a reachable base-frame region, and
   requires `--move-to-board`. Do not reuse pre-move pixel coordinates after
   changing the view.

Coarse approach defaults to z=0.55 m with fixed yaw and re-aims head_j1=+0.55
after Robot() connection. Defaults: speed_scale=0.45, +X/+Y probes=12 mm, correction gain=0.65,
maximum correction=15 mm, local radius=60 mm, at most 8 corrections,
convergence within 5 pixels. Each probe returns to the initial pose before the
next. Calibration uses measured TCP XY, including cross-axis error. The wrist
stage restores the normal IK configuration after coarse navigation and tightens
position tolerance to 0.7 mm; coarse 10 mm tolerance cannot resolve these steps.

The tool prints every observation and correction, saves both RGB views at each
sample, adds `_tracked.png` overlays (green feature, red goal), and writes
`events.jsonl` and `result.json`. Success means `status: converged`. Exceptions
record `status: stopped`, return nonzero, and issue no automatic recovery move.
Tracking loss/ambiguity, stale frame identity, negligible probe image motion,
poor Jacobian conditioning, TCP drift, travel limits, divergence and stalled
convergence stop further corrections. Arm motion failures retain the existing
adapter's stop behavior.

This is an image-centering benchmark. Image center is not yet a calibrated jaw
contact point. Before battery descent, teach the appropriate `--goal-pixel U V`
for the chosen orientation and hover height, plus the part's grasp/place Z.
Re-identify the local Jacobian after changing height or orientation. This
benchmark deliberately does not descend or infer a successful grasp.

Dependencies are NumPy and OpenCV plus the robot's existing SDK/camera modules.
Missing OpenCV fails before robot connection. Check with:

```bash
python3 -c 'import numpy, cv2, wrist_cameras; print("wrist dependencies ready")'
```

Offline validation: `python3 -m unittest discover -s tests -p test_wrist_servo.py -v`.
The numerical/image tests require NumPy and OpenCV; standard-library-only runs
skip them explicitly. Synthetic validation is not evidence of physical success.
