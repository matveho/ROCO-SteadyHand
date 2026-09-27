# Vega live path

This document tracks the competition Vega actually assigned to Team SteadyHand.
The shipped config remains deliberately gated for facts that are not yet
physically verified. `policy.py` remains the submitted simulation policy.

## Verified competition-unit identity

Verified onsite Sep 27:

- hostname: `vega-1u`
- local robot Ethernet address used from the team laptop: `192.168.50.20`
- `ROBOT_NAME=dm/vgfcb66075ea-1u`
- robot model: `vega_1u`
- Ubuntu 22.04.5 / kernel 5.15.185-tegra / aarch64
- active arm/control Python: Conda Python 3.13
- `dexcontrol==0.5.0`
- robot reports firmware/system version `0.5.1`, minimum client `0.5.0`
- Zenoh config:
  `/home/dexmate/.dexmate/comm/zenoh/dm_internal_061826.dzcfg`
- live components: left arm, right arm, head, e-stop, heartbeat
- no live torso or chassis component
- both live arm joint names/orders and limits match `configs/robots/vega.json`
- software/physical e-stop state can be read and was clear during the probe
- both arm state timestamps advance
- wrench reads returned six zeros while unloaded; units/frame remain unverified

The vendor `Robot()` constructor was successfully used on this unit. It can
initialize the head, so keep the head workspace clear before constructing it.

## Python/environment split

Use the Conda `python3` for arm/control work because that is where
`dexcontrol 0.5.0` is installed.

The wrist-camera/GStreamer path was previously documented against
`/usr/bin/python3` (Python 3.10). Do not assume one interpreter can run both
stacks until tested.

Do not import `dexmotion` just to inspect it on this unit. A direct
`import dexmotion` caused a native segmentation fault during onsite
inspection. Read its installed source files directly if needed.

## Installed robot files

Local dexcontrol checkout:

~~~
/home/dexmate/dexcontrol
~~~

Verified gripper-equipped URDF:

~~~
/home/dexmate/miniconda3/lib/python3.13/site-packages/dexmate_urdf/robots/humanoid/vega_1u/vega_1u_gripper.urdf
~~~

The live `Vega1UConfig` itself points to the standard
`vega_1u.urdf`. SteadyHand needs the gripper-equipped URDF for physical
tool geometry.

The previously assumed paths below were not present during initial inspection:

~~~
~/Downloads/Dexmate
/home/dexmate/gripper.py
~~~

Do not use those paths unless they later appear for a documented reason.

## URDF chain facts

The installed gripper URDF contains:

- fixed root frame `base_link`
- `L_ee`, `R_ee`
- `L_gripper_base`, `R_gripper_base`
- no `tip_l` / `tip_r` frame in the inspected standard gripper URDF
- `L_ee` and `L_gripper_base` parent joint: `L_arm_j7`
- `R_ee` and `R_gripper_base` parent joint: `R_arm_j7`

It also models two movable joints upstream of both arms even though the live
robot exposes neither as a controllable component:

- `Lift`: prismatic +Z, neutral 0.0, limits 0.0..0.4 m
- `torso_flip`: revolute +X, neutral 0.0, limits 0.0..1.0 rad

Those neutral values are **model defaults, not yet verified physical values**.
Do not silently set `kinematics.fixed_joint_values` to zero until DexMate or
an independent physical/model check confirms that convention for this unit.

## Verified arm command semantics

The installed dexcontrol 0.5.0 source resolves the previous `wait_time`
ambiguity:

- `set_joint_pos(..., wait_time=0)` sends one raw position setpoint and must be
  called repeatedly in a high-frequency loop (e.g. 100 Hz; documented maximum
  500 Hz).
- `set_joint_pos(..., wait_time>0)` performs client-side interpolation and is
  deprecated.
- `move_to_joint_pos(..., velocity_scale=...)` sends a target to the
  robot-server motion plugin. The plugin handles trajectory generation,
  smoothing, and gravity compensation and returns a `MotionHandle`.

The live arm config reports `default_control_hz=100`.

On the competition robot, a left-arm `move_to_joint_pos()` command whose
target was the arm's **current measured pose**, with
`velocity_scale=0.05`, returned a handle that reached `finished` within
five seconds. No intentional arm displacement was requested.

SteadyHand now uses `move_to_joint_pos()` for physical arm motion. It keeps its
own small waypoint bound and verifies fresh measured state after every
controller-managed waypoint.

## No-motion checks

From the competition Jetson:

~~~bash
cd ~/ROCO-SteadyHand
python3 -m unittest discover -s tests -v
python3 tools/vega_sdk_probe.py --confirm-head-motion
~~~

The SDK probe constructs `Robot()` and therefore is not literally zero-motion
for the head, but it intentionally sends no arm or gripper command.

Wrist-camera tests may need the system interpreter:

~~~bash
/usr/bin/python3 tools/vega_wrists_probe.py
~~~

## Configuration still required before SteadyHand Cartesian motion

Already recorded in the template:

- `robot_name=dm/vgfcb66075ea-1u`
- host `192.168.50.20`
- dexcontrol `0.5.0`
- verified arm joint names/limits
- gripper-URDF candidate path
- live arm default control rate 100 Hz (informational; the motion plugin owns
  trajectory execution)

Still intentionally unresolved:

- `working_arm`
- final `urdf_path` selection for the run
- `kinematics.base_frame`
- `kinematics.ee_frame`
- physical `Lift` / `torso_flip` values for Pinocchio
- `motion.joint_reached_tolerance_rad`
- `motion.joint_timeout_s`
- actual competition gripper-driver path
- gripper side/current/homing state
- camera extrinsics and board registration
- per-part `T_part_tcp`

The old `max_joint_speed_rad_s`, `motion.step_wait_time_s`, and
`motion.control_hz` fields are no longer required by VegaAdapter's physical
motion path; `speed_scale` is passed to the robot-server plugin as
`velocity_scale`.

## Validate FK/IK before Cartesian motion

Once the physical fixed model values and frames are known:

~~~bash
python3 tools/vega_ik_check.py \
  --urdf /home/dexmate/miniconda3/lib/python3.13/site-packages/dexmate_urdf/robots/humanoid/vega_1u/vega_1u_gripper.urdf \
  --base-frame <verified-fixed-base-frame> \
  --ee-frame <verified-ee-frame> \
  --arm left \
  --fixed Lift=<verified-value> \
  --fixed torso_flip=<verified-value> \
  --q q1 q2 q3 q4 q5 q6 q7 \
  --dz 0.01
~~~

The checker fails rather than inventing values for movable joints in the EE
chain. Compare the FK pose against an independent vendor/physical reference
before trusting IK.

## First physical arm motion

We have verified only a current-pose command so far. The next physical milestone
is one very small, reversible free-space joint motion with e-stop reachable.

Use the gated SteadyHand tool after filling the remaining motion config:

~~~bash
python3 tools/vega_joint_check.py \
  --robot-config <onsite-vega.json> \
  --joint-index 7 --delta-rad 0.005 \
  --speed-scale 0.05 \
  --confirm-head-motion --confirm-physical-motion
~~~

The tool pauses before the move and before return. A successful joint move does
not validate Cartesian IK or collision safety.

## Gripper status

The organizer-linked competition driver contract is known:

- `Grippers()`
- independent `g.left` / `g.right`
- per-side `home/open/close/grip/current/position`
- `grip(current=...)` returns a dictionary including `gripped`

But the expected local file `/home/dexmate/gripper.py` was not present during
initial inspection. The actual competition-unit gripper driver/CAN setup is
still unresolved. Do not install system software or run a homing command until
the sponsor/technician confirms the intended driver and CAN procedure.

## Targets and tool geometry

Runtime targets must include `base_frame`, `position_units=m`, and
`quaternion_order=wxyz`. Pick/place values are object/reference poses, not TCP
poses. Prefer a measured part-relative grasp transform:

~~~
T_base_tcp = T_base_part * T_part_tcp
~~~

Do not promote simulation-era world offsets or source-robot calibration to
competition-unit calibration without measurement.

## First pick/place

After joint motion, IK, TCP direction, and gripper are independently verified,
use one easy open-release part first (for example battery_size1):

~~~bash
python3 tools/vega_run_part.py \
  --robot-config <onsite-vega.json> --skills <onsite-skills.json> \
  --targets <runtime_targets.json> --part battery_size1 --operator <name> \
  --speed-scale 0.05 --confirm-head-motion --confirm-physical-motion
~~~

Do not begin with the insertion parts.

## Remaining major limitations

- collision checking is incomplete
- automatic board/part perception is not yet implemented
- camera-to-base calibration is unresolved
- grasp transforms are not calibrated
- placement verification is incomplete
- wrench units/frame remain unknown
- insertion success logic is not yet robust
- gripper driver/location is still unresolved
- wrist camera physical A/B mapping remains unresolved

A completed trajectory is not evidence of successful assembly.
