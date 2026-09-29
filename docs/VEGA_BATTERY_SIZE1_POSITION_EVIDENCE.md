# Vega battery_size1 Position Evidence

**Purpose:** provide an objective, reproducible comparison between a live Vega right-arm state and the checked-in board and wrist calibration used for the battery_size1 pickup path.

This record documents measured values and software provenance. It does not, by itself, establish how another team produced its code or establish plagiarism. The live snapshot below was supplied from the competition robot; repository values were recomputed from the files named in this document.

## 1. Live read-only snapshot

The snapshot was captured without constructing the normal motion Robot() object. It used the repository's ReadOnlyVegaTipReader, which subscribes to right-arm state and computes local FK. No arm, gripper, camera, or E-stop command was issued.

Operator capture metadata:

~~~text
captured_at_utc     = 2026-09-12T16:24:41.294033+00:00
robot_name          = dm/vgfcb66075ea-1u
git_sha             = f0d64ce4c3e5b450fce6df6d4a2726f5a429a2db
joint_timestamp_ns  = 1789230280913859018
tcp_frame           = tip_r
~~~

Measured joint state, in configured SDK order:

| Joint | Radians | Degrees |
|---|---:|---:|
| R_arm_j1 | 0.2783939838409424 | 15.9508003159 |
| R_arm_j2 | -0.7670617103576660 | -43.9493986296 |
| R_arm_j3 | -1.9423315525054932 | -111.2874003737 |
| R_arm_j4 | -2.1447443962097170 | -122.8848020372 |
| R_arm_j5 | 2.0234405994415283 | 115.9346064434 |
| R_arm_j6 | -1.2766804695129395 | -73.1484026899 |
| R_arm_j7 | -0.4091662764549255 | -23.4435007600 |

Modeled live tip_r pose from those measured joints:

~~~text
position_m       = (0.35060428549583184, -0.14050284919594436, 0.6044448153805213)
quaternion_wxyz  = (0.815564602688782, 0.052608933897955644,
                    0.07161210043124001, 0.5718027509439209)
~~~

The complete raw JSON output is the primary evidence record. It includes the joint timestamp, full-precision radians/degrees, FK pose, and comparisons to the two measured transit presets.

## 2. Saved battery pickup calibration

The saved profile is calibration/wrist_part_profiles.json, part battery_size1. Relevant fields are:

~~~text
working_arm           = right
tcp_frame             = tip_r
wrist_camera          = wrist_a
coarse_xy_m           = (0.3558604585319267, -0.1432632399164138)
hover_clearance_m     = 0.1000000000000000
grasp_clearance_m     = 0.0080000000000000
yaw_deg               = 0.0
gripper_open_fraction = 0.18999957274999468
grasp_verified        = true
place_verified        = true
goal_source           = operator_confirmed_current_pose
feature_tracking_mode = strict
calibration_sha256    = 41233a67a52630d9b3d7670b54b31093db05a53d4118702927dba1ee31aff21f
~~~

The profile coarse_xy_m is the Cartesian XY position retained for the pickup fallback. During no-CV competition pickup, the implementation reconstructs:

~~~text
target_x          = coarse_xy_m[0]
target_y          = coarse_xy_m[1]
target_z          = calibrated_surface_z(target_x, target_y) + hover_clearance_m
target_quaternion = measured RIGHT_READY tip_r quaternion
~~~

This is implemented in tools/vega_wrist_part_calibrate.py, lines 588–616. Profile creation records coarse_xy_m, feature/goal pixels, clearance, yaw, gripper opening, template hash, and grasp result at lines 1205–1232.

## 3. Board and height calibration

The canonical permanent calibration is calibration/vega_board_manual_fallback.json. The live and fallback files are byte-identical and have this SHA-256:

~~~text
41233a67a52630d9b3d7670b54b31093db05a53d4118702927dba1ee31aff21f
~~~

Identity and board measurements:

~~~text
robot_name          = dm/vgfcb66075ea-1u
base_frame          = vega_1u_base_link
tcp_frame           = tip_r
board width         = 0.386 m
board motion model  = horizontal_translation_only_fixed_table_plane
floor guard         = 0.4560002716867571 m
initial hover       = 0.5360002716867571 m
forward rise angle  = 12.418744602 degrees
permanent_fallback  = true
~~~

The four operator-confirmed calibration samples were each measured at 100 mm TCP-to-board clearance. The fitted board surface is:

~~~text
z_m = -0.2169866276788319*x_m
      +0.002782350024095798*y_m
      +0.5890807137080113
~~~

Maximum fitted residual is 0.510347 mm. Corrected board axes are:

~~~text
board_x_unit_base_xy = ( 0.01091155147271351, -0.9999404672501551)
board_y_unit_base_xy = (-0.9999404672501551, -0.01091155147271351)
raw axis angle error  = 1.8445894682438237 degrees
~~~

Permanent camera target corrections, stored in the fallback JSON and configs/robots/vega.json:

~~~text
CENTER       = (-0.046518656, -0.014059528) m
TOP_RIGHT    = (-0.045435466,  0.006742256) m
BOTTOM_RIGHT = (-0.054762737,  0.024019439) m
BOTTOM_LEFT  = (-0.050445634, -0.042835330) m
~~~

These are corrected TCP XY minus raw camera coarse XY. They are operator-measured five-point correction data.

## 4. Verticalization and camera registration path

The board image path has two distinct stages:

1. tools/vega_competition_pipeline.py, lines 295–354, calls move_camera_clear_for_image, verifies the head reaches downward-view target [0.55, 0.0, 0.0] within 0.03 rad, captures the head image, and registers the board scene.
2. steadyhand/vega_camera_clear.py, lines 333–380, contains the post-image verticalization sequence. It finds an IK-reachable vertical-claw waypoint, solves it from the measured current seed, and sends one server-smoothed joint trajectory rather than interpolating unreachable Cartesian orientations.

The vertical-claw model is defined in steadyhand/vega_camera_clear.py, lines 40–47. At zero yaw it uses a tip_r quaternion corresponding to a physical downward gripper axis. The camera-clear planner respects the calibrated floor and searches only validated high poses.

The board image does not replace physical height calibration. A fresh image may update live horizontal board registration only when its geometry signature is compatible with the calibration image; otherwise the pipeline requires a new full five-point calibration. This contract is implemented in _runtime_from_board_scene, lines 357–375 of tools/vega_competition_pipeline.py.

## 5. Task-coordinate transform

The source task coordinates are configs/task_coordinates.json. The runtime transform:

- uses the 386 mm board width;
- applies the configured 180-degree task-coordinate orientation;
- maps board-relative axes into measured base-frame axes;
- recomputes Z from the fitted plane so hover remains parallel to the board;
- applies the onsite 12 mm forward correction (+base X) after registration.

The implementation is tools/vega_competition_pipeline.py, lines 216–245. The 12 mm correction is the explicit constant TASK_FORWARD_OFFSET_M = 0.012 at lines 52–55.

The task transform is upstream of the saved wrist profile. The wrist profile then preserves the operator-confirmed coarse hover for the part and can be used when wrist CV is unavailable.

For auditability, the battery source point and the independently generated task audit entry are:

~~~text
source task coordinate (configs/task_coordinates.json) = (0.08757302905790931, 0.08527389315586605, 0.0)
derived task base XY (calibration/task_geometry_audit.json) = (0.33308487569817746, -0.14400469510721214)
derived task 100 mm hover Z = 0.6165279691685872 m
task geometry audit SHA-256 = 4c619ec514395b2bcaacbb6231bbae9d5d959ab7f29685f7d4b341bd4efd9771
~~~

That upstream audit target is distinct from the later operator-confirmed wrist teaching coarse XY (0.3558604585319267, -0.1432632399164138). The no-CV pickup path intentionally uses the saved wrist teaching coarse XY when it is available, then falls back to the live task target only if that recorded hover is unreachable.

## 6. Numerical comparison

Using the saved battery profile XY and calibrated plane:

~~~text
profile surface z at coarse XY = 0.5114651444078923 m
expected 100 mm hover           = 0.6114651444078922 m
~~~

Measured live state:

~~~text
live tip_r = (0.35060428549583184, -0.14050284919594436, 0.6044448153805213) m
~~~

Live minus saved battery hover:

~~~text
delta X = -5.256173 mm
delta Y = +2.760391 mm
delta Z = -7.020329 mm
horizontal error = 5.936927 mm
3-D error        = 9.194136 mm
~~~

At the live XY itself, the calibrated plane predicts z_surface = 0.5126133440426786 m, so live TCP clearance is 91.831471 mm. Relative to the 100 mm convention, the live point is 8.168529 mm low.

The live quaternion differs from the measured RIGHT_READY quaternion by approximately 0.0526 degrees. The live pose is orientation-consistent with the calibrated vertical tool orientation even though its position is offset from the saved battery hover.

The saved profile does not contain a battery-specific seven-joint teaching vector; it contains Cartesian coarse XY plus calibrated surface/hover clearance. Therefore, the defensible position comparison is the FK tip_r position above. The right-arm joint vector is retained in the raw snapshot:

~~~text
[ 0.2783939838409424,
 -0.7670617103576660,
 -1.9423315525054932,
 -2.1447443962097170,
  2.0234405994415283,
 -1.2766804695129395,
 -0.4091662764549255 ] rad
~~~

For context, live state is not expected to equal RIGHT_READY or RIGHT_CAMERA_CLEAR; those are transit/camera presets, not the battery pickup pose. Largest live-to-RIGHT_READY joint difference is 0.4979198 rad (28.5287 degrees); largest live-to-RIGHT_CAMERA_CLEAR difference is 0.4982793 rad (28.5493 degrees).

## 7. Repository provenance

The checked-in repository used for this record is currently:

~~~text
HEAD = 778940d  Refresh task geometry audit after USB correction
~~~

Relevant commits:

~~~text
95c5fec  Record onsite Vega calibration and wrist profiles
ddb283e  Migrate untracked onsite artifacts during deploy
c384ce6  Preserve onsite calibration artifacts and correct USB target
858504b  Make competition pickup independent of wrist centering
41a4bb7  Add forward offset and no-CV pickup retries
ecc9f03  Preserve manual grasp hover during descent
~~~

The robot snapshot reports runtime SHA f0d64ce4c3e5b450fce6df6d4a2726f5a429a2db, the onsite status-bundle commit recorded in the snapshot. This distinction should be preserved when comparing the robot checkout with the later repository consolidation commits.

SHA-256 values for principal checked-in inputs:

~~~text
calibration/vega_board_manual.json                    41233a67a52630d9b3d7670b54b31093db05a53d4118702927dba1ee31aff21f
calibration/vega_board_manual_fallback.json            41233a67a52630d9b3d7670b54b31093db05a53d4118702927dba1ee31aff21f
calibration/wrist_part_profiles.json                   7bf7d6633a519af3b9760b8becc715f1e073465d40ff47d2cac888812c96aba3
configs/robots/vega.json                                ffdbc6e18993c3b34bd851c3d62fd126871fa95bf4db6ce45883eadbba04fd7a
configs/task_coordinates.json                           acc14c9c6941835a140d9339559cab52803ee385a6d5a43d45b270672387fed3
configs/task_board.json                                 02ef5e45368719bfe4d2d805d94c3946a7222e2c531404f1b2e81188a618197f
configs/robots/vega_1u_competition_kinematics.urdf      9160d2861696161f1cbc280cc5d7a3ff68212c8b1cd1f0ec99d0bacaa96335fd
~~~

The exact configs/robots/vega.json SHA-256 is ffdbc6e18993c3b34bd851c3d62fd126871fa95bf4db6ce45883eadbba04fd7a.

## 8. Reproduction procedure for an independent reviewer

From the repository root:

~~~bash
python3 - <<'PY'
import json
from pathlib import Path

profile = json.loads(Path("calibration/wrist_part_profiles.json").read_text())
cal = json.loads(Path("calibration/vega_board_manual_fallback.json").read_text())
p = profile["parts"]["battery_size1"]
a, b, c = (cal["board_surface_plane_base"]["coefficients"][k] for k in ("a", "b", "c"))
x, y = p["coarse_xy_m"]
surface = a*x + b*y + c
print("battery coarse XY:", (x, y))
print("surface z:", surface)
print("100 mm hover z:", surface + p["hover_clearance_m"])
PY
~~~

The live snapshot can be reproduced onsite with the no-motion ReadOnlyVegaTipReader capture command. The resulting tip_r FK pose and the calculation in Section 6 are sufficient to verify the stated error without commanding a new position.
