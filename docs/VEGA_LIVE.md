# Vega live path

This is the shortest path from fresh checkout to a physical one-part attempt.
Do not skip the vendor health/e-stop checks in BOOTCAMP.md.

## 1. Pull and inspect

~~~bash
git pull
python onsite.py doctor
python onsite.py check-config --robot vega
/usr/bin/python3 tools/vega_preflight.py
~~~

doctor never contacts hardware. vega_preflight checks local modules/files only.

## 2. Verify cameras without motion

~~~bash
export ROBOT_NAME=<onsite-name>
/usr/bin/python3 tools/vega_head_probe.py
/usr/bin/python3 tools/vega_wrists_probe.py
/usr/bin/python3 tools/vega_capture_snapshot.py --output ~/vega-first-snapshot
~~~

The camera path does not construct dexcontrol Robot().

## 3. Resolve IK facts

Obtain/verify:

- gripper-equipped URDF path;
- selected physical arm;
- actual EE/TCP frame name in that URDF;
- every non-arm movable joint in the EE chain and its physical value;
- a safe step wait time from a vendor/example move.

Use current 7 arm joints from the existing inspect_joints.py and run:

~~~bash
/usr/bin/python3 tools/vega_ik_check.py \
  --urdf <gripper-urdf> \
  --ee-frame <verified-frame> \
  --arm left \
  --fixed <non-arm-joint>=<verified-value> \
  --q q1 q2 q3 q4 q5 q6 q7 \
  --dz 0.01
~~~

This performs FK + a one-centimetre IK solve and sends no robot commands.
If the URDF chain has an unconfigured movable joint, the checker names it.

## 4. Bring up CAN gripper separately

Use the supplied robot instructions to bring up can1, then validate the existing
/home/dexmate/gripper.py demo before using SteadyHand. Homing is expected at the
start of a new powered session. Keep hands clear.

SteadyHand intentionally uses the documented dual-gripper API only. If the
competition setup requires single-side CAN commands, inspect the actual library
and add that mapping rather than guessing an attribute name.

## 5. Create current target poses

Copy configs/runtime_targets.template.json into a session-specific file.

The poses are OBJECT/DESTINATION poses in robot_base, not TCP poses. The
executor applies configs/skills/vega.json to convert them into TCP goals.

The preferred long-term representation is a calibrated T_part_tcp per part.
Until then, the skill config carries the submitted policy's world-axis EE
offset/orientation as a bring-up fallback.

Do not paste simulation world XYZ directly into this file.

## 6. First physical part

Start with one open-release part, not a connector insertion.

~~~bash
/usr/bin/python3 tools/vega_run_part.py \
  --targets <current-targets.json> \
  --part battery_size1 \
  --operator Matvey \
  --working-arm left \
  --robot-name '<onsite-name>' \
  --urdf <gripper-urdf> \
  --ee-frame <verified-frame> \
  --step-wait <verified-seconds> \
  --gripper-scope both \
  --confirm-head-motion \
  --confirm-physical-motion
~~~

The runner:

1. snapshots config/targets/skill settings;
2. constructs Robot() (which may move the head);
3. loads Pinocchio against the gripper-equipped URDF;
4. homes/connects the CAN grippers;
5. opens the gripper;
6. approaches the pick through segmented Cartesian targets;
7. descends and current-limited grips;
8. lifts and transfers through segmented Cartesian targets;
9. places/releases/retracts;
10. logs every phase/waypoint;
11. software-e-stops on an unexpected exception.

A completed sequence is not automatically recorded as a successful placement.
Verification remains separate.

## 7. Snap/insertion parts

The executor ports the simulation XY search patterns, but the simulator's snap
teleport is gone. Physical insertion uses:

- a pre-insert pose above the destination;
- center-out XY candidates;
- short Cartesian descent waypoints;
- live TCP-reached checking;
- an optional relative wrist-force guard;
- retract before the next XY candidate.

By default connector insertion REFUSES to run until force_delta_limit is
measured/verified onsite. The wrench units/frame are currently unknown.

There is an override, --allow-snap-without-force-guard, but it is intentionally
explicit.

## 8. Full sequence

Only after individual parts work:

~~~bash
/usr/bin/python3 tools/vega_run_sequence.py <same hardware arguments> \
  --targets <current-targets.json> \
  --operator Matvey \
  --return-home-q q1 q2 q3 q4 q5 q6 q7 \
  --confirm-head-motion \
  --confirm-physical-motion
~~~

Use --return-home-q only after physically validating that pose/path. This
preserves the submitted policy's strategy of returning to a consistent IK seed
between parts.

## Remaining hard gap

Automatic runtime target generation is still the major missing Vega layer:
camera images must become current board/part poses. Everything below those poses
is now represented in code, but perception/calibration cannot be honestly
completed without real board imagery and verified camera/base extrinsics.
