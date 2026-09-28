# Wrist XY benchmark

Live inspection on 2026-09-27 found no `wrist_cameras` module in either
system Python 3.10 or Conda Python 3.13 on dm/vgfcb66075ea-1u. Only ZED
video0/video1 were enumerated; `/opt/wrist-cameras` was absent. The Sony
capture path needs the sponsor's hardware/driver setup before this can run.
No driver, boot, or kernel changes were made.

Once Sony capture works, cover the physical left wrist to identify its API
label. Run the benchmark in an environment with both `wrist_cameras` and
`dexcontrol`. A split-interpreter bridge is not implemented.

First establish the corrected vertical claw at a clear board-region hover
3-10 cm above the configured task floor. The benchmark does not approach
the board or descend. Run from the deployed repository:

```bash
python3 tools/vega_wrist_servo.py \
  --left-camera wrist_a --output runs/wrist-attempt-01 \
  --speed-scale 0.45 --confirm-head-motion --confirm-physical-motion
```

Replace `wrist_a` with the physically verified label. The output directory
must be new. View each saved PPM on the robot desktop or copy it to the laptop;
enter the same stationary feature's u (column), v (row) each time.
Do not move the feature or head/arm manually during the attempt.

The tool captures reference, +X, and +Y images, returning to reference between
axes. Jacobian estimation uses measured TCP displacements. It then makes up
to five corrections, each <=10 mm, within 40 mm of the initial XY, at fixed
Z and orientation. It rejects missing/repeated frame IDs, poor calibration,
tracking error, and increasing pixel error. Failure stops without auto-return.

This proves centering in the camera view only. The camera center is not the
gripping-pad center: calibrate a jaw-aligned desired pixel at the relevant
height before using this for picking. Re-estimate the local Jacobian after
changing orientation or working height. No object detector is included yet.
