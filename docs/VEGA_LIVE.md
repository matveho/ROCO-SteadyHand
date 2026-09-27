# Vega live path

The shipped config is intentionally non-runnable. Copy the robot and skill JSON
to an onsite session directory, fill only measured values, and keep the originals
as templates. `policy.py` remains the submitted simulation policy.

## No-motion checks

~~~bash
git pull --ff-only
python -m unittest discover -s tests -v
/usr/bin/python3 tools/capture_environment.py > <session>/environment.json
/usr/bin/python3 tools/vega_preflight.py --robot-config <onsite-vega.json>
export ROBOT_NAME='<verified-name>'
/usr/bin/python3 tools/vega_head_probe.py
/usr/bin/python3 tools/vega_wrists_probe.py
~~~

`vega_preflight.py` imports no robot API and never constructs `Robot()`. Verify
the wrist A/B physical mapping visually. Do not use the normal Vega URDF: it
omits the physical grippers.

## Required measured configuration

In the onsite robot config fill:

- `robot_name`, `working_arm`, and the gripper-equipped `urdf_path`;
- `kinematics.base_frame`, a named fixed URDF frame that is also the coordinate
  frame of every runtime target;
- `kinematics.ee_frame`, the frame whose pose is actually commanded;
- every non-arm movable ancestor such as `Lift` or `torso_flip` in
  `kinematics.fixed_joint_values`, using the physical value;
- `max_joint_speed_rad_s`, `motion.control_hz`,
  `motion.step_wait_time_s`, `motion.joint_reached_tolerance_rad`, and
  `motion.joint_timeout_s`, from the installed dexcontrol example/robot test;
- `gripper.scope` set to the selected working arm, the actual driver path, and a verified
  `gripper.grip_current_a`.

`step_wait_time_s` is extra endpoint settle time. It is not a speed control.
The adapter sends `max_joint_speed_rad_s * speed_scale` as dexcontrol
`wait_kwargs.max_vel` and checks a fresh timestamp plus measured endpoint after
every segment. `control_hz` must be the verified vendor loop rate in 100–500 Hz.

Validate FK/IK without the robot:

~~~bash
/usr/bin/python3 tools/vega_ik_check.py \
  --urdf <vega_1u_gripper.urdf> \
  --base-frame <fixed-base-frame> --ee-frame <tcp-frame> --arm left \
  --fixed Lift=<measured> --fixed torso_flip=<measured> \
  --q q1 q2 q3 q4 q5 q6 q7 --dz 0.01
~~~

The checker fails if the EE is not downstream of all seven configured joints,
if a movable ancestor is unknown, or if a seed/fixed value violates the
effective URDF/config limits. Compare its initial FK pose to an independent
vendor/physical pose before trusting IK.

## Targets and tool geometry

Runtime targets must include `base_frame`, `position_units=m`, and
`quaternion_order=wxyz`. Pick and place are object/reference poses, not TCP
poses. For open-release parts, place is the desired release pose; it may be
above the final settled/grading height. `T_part_tcp` must use the same object
reference at pick and place.

The legacy simulation offset is a world-axis addition. The simulation
orientation also passed through Lula's USD-to-URDF correction, so `[0,1,0,0]`
is not proven to be the physical URDF TCP orientation. The runner refuses the
legacy values until `legacy_geometry_verified=true`, and only after they are
measured for the chosen EE frame. Prefer a measured `T_part_tcp`.

First validate every input without constructing `Robot()` or CAN:

~~~bash
/usr/bin/python3 tools/vega_run_part.py \
  --robot-config <onsite-vega.json> --skills <onsite-skills.json> \
  --targets <runtime_targets.json> --part battery_size1 --operator <name> \
  --check-only
~~~

## First connection and tiny motion

Run connection-only with the head workspace clear and physical e-stop ready:

~~~bash
/usr/bin/python3 tools/vega_joint_check.py \
  --robot-config <onsite-vega.json> --connect-only \
  --confirm-head-motion --confirm-physical-motion
~~~

Then select one unobstructed joint and make no more than a 0.02 rad reversible
move. The tool pauses before moving and before returning:

~~~bash
/usr/bin/python3 tools/vega_joint_check.py \
  --robot-config <onsite-vega.json> --joint-index 7 --delta-rad 0.01 \
  --speed-scale 0.1 --confirm-head-motion --confirm-physical-motion
~~~

Bring `can1` up with the supplied robot instructions and run the unmodified
`/home/dexmate/gripper.py` demo before SteadyHand. The organizer-linked driver
exposes `g.left` and `g.right`; the wrapper homes and moves only the configured
working-arm motor during a one-arm attempt. Verify that the installed file has
the same API.

## First pick/place

Use one open-release part, an interactive terminal, and direct observation:

~~~bash
/usr/bin/python3 tools/vega_run_part.py \
  --robot-config <onsite-vega.json> --skills <onsite-skills.json> \
  --targets <runtime_targets.json> --part battery_size1 --operator <name> \
  --speed-scale 0.1 --confirm-head-motion --confirm-physical-motion
~~~

The runner loads all local config, targets, geometry, gripper gates, and IK
before `Robot()` can move the head. It then homes CAN, executes segmented TCP
motion, checks fresh joint readback after every segment, and requires the
operator to type `yes` after grasp, after lift, and after placement. Any
exception, Ctrl-C, failed verification, stale state, tracking error, or limit
violation activates software e-stop and halts CAN. `close()` still attempts
robot shutdown if camera or CAN cleanup fails.

## Insertion remains a later step

Insertion is disabled until these are measured: `force_delta_limit`,
`wrench_units="N,Nm"`, `wrench_frame`, and `max_contact_step_m`. Force is only
checked between blocking joint moves, not continuously. A contact-limit event
stops without blind automatic retract. TCP arrival is never insertion success;
the operator must confirm physical seating before release. There is no bypass
for running without a force guard.

Full sequence execution defaults to `battery_size1`. More than one explicit
part requires a physically validated `--return-home-q`; the home path itself is
not collision checked.

Automatic target perception and collision checking are not implemented. Keep
using measured `runtime_targets.json` and visually clear every segmented path.
