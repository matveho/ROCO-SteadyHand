# Connecting the assembly policy to North

`policy.py` is a simulation entry point. It obtains a Lula controller from
`env_info.L_controller`, target poses from `PartTarget`, and insertion success
from `obs.snap_fired`. Those are simulator services, not North SDK services.
Preserve this file and reuse the pick/approach/grasp/lift/transfer/insert/release
sequence through the existing physical executor.

The hardware path should be:

```text
measured board/part poses in the robot frame
  -> SteadyHand task phases and segmented TCP targets
  -> North-specific IK, seeded from measured joints
  -> bounded joint targets + feedback/fault/mode checks
  -> NorthClient.publish_single_action()
  -> inference/action -> native controller -> arm drivers
```

## What is implemented

* `tools/sharpa_observe.py`: command-free live observation checks, including
  advancing source stamps for every required component. Images remain in RAM.
* `--check-sdk`: serialize and decode an identity target with the installed
  organizer SDK, without constructing its client, transport or sender thread.
* `steadyhand.adapters.north_commands.joint_step`: pure command construction
  in SDK joint order. It holds both arms and body/neck, optionally changing one
  arm within explicit joint limits and a per-step bound. It never publishes.
* The separate native SDK copy can carry the arm-index correction under
  `patches/sharpa-north`. This fixes invalid indexing; it does not establish
  physical tracking accuracy or change stop behavior.

The joint interface uses radians with these SDK keys:

```python
{
    "/action/left_arm/joint_angle": left_arm_7,
    "/action/right_arm/joint_angle": right_arm_7,
    "/action/motor/joint_angle": body_and_neck_7,  # motor IDs 1 through 7
}
```

Hold the unselected arm and body from a fresh, verified initial observation.
Do not manufacture missing groups as zeros. Omit hand/chassis commands for an
arm-only test. The serializer should use `using_eef_control=False` and
`static_lowbody=False`; the latter avoids SDK body-target rebasing.

When a sender is implemented, use one synchronous sender calling
`publish_single_action()`. Do not mix it with the SDK's buffered `step()` sender.
The SDK's `shutdown()` stops its background sending thread; `cleanup()` also
disconnects transport. Neither establishes a physical stop.

## Required controller checks before enabling a sender

In the inspected native controller, `actionSendLoop()` keeps the last arm/body
target. `action_timeout_ms` clears chassis commands only. Python buffer clearing,
closing a connection, and Ctrl-C therefore cannot be advertised as stopping
the arms. The `isDaggerTransition()` predicate also permits cached targets in
MOVING with the action switch off, without testing the Dagger-enabled setting.
Resolve the supported hold/stop procedure with the organizer and test it before
relying on software aborts. A stop must account for targets already held by the
arm driver, not just empty a Python queue.

The inspected joint-command path calls the stability checker with overwriting
disabled. Do not assume it will correct or reject an unsafe but well-formed
joint target; motion limits and collision validation remain necessary.

F6 changes operation mode. F2 transitions through Init, Standby and Moving;
transitions into Standby invoke homing. F2 is not a pause-at-current-pose command.
Keep the operator in charge of those transitions, and verify measured arrival
and native homing diagnostics before treating Standby as ready. A status label
alone does not prove homing succeeded.

The first physical test should initialize all target groups from fresh feedback,
hold those values, and then perform a bounded out-and-back change on one verified
joint while checking measured tracking. Stop on stale feedback, faults, wrong
mode, unexpected motion of held joints, or excessive tracking error. A requested
offset such as 0.005 rad (0.29 degrees) is only a proposed test size, not a
collision-clearance or tracking validation. Do not execute it from this document
as a ready-made motion script.

## Completing Cartesian execution

The native SDK includes `NorthKinematics`, Python bindings named
`north_kinematics_py`, and `CalcArmIkLeft`/`CalcArmIkRight` C++ overloads that accept
a reference configuration. Inspect the active controller's `urdf_file`; multiple
models are shipped. Use the active model, verify its joint/frame mapping, and
test FK -> IK -> FK residuals offline before sending results. The inspected
Python arm-IK bindings do not expose the reference-configuration overload;
check continuity explicitly or add a wrapper in the separate SDK copy.

The repository's `PinocchioArmKinematics` is another candidate, but its existing
Vega configuration is not North calibration. North's movable torso values and
TCP frame must be supplied and verified. Never feed simulator joint vectors
directly to either hardware arm.

Implement `SharpaAdapter.move_joints`, then `get_tcp_pose`/`move_tcp`, with the
verified stop contract and motion guards. Finally supply measured board-to-robot
and hand/TCP transforms and implement physical grasp/placement verification.
Use vision/contact/tactile evidence in place of the simulator's `snap_fired`.
Camera transport working does not establish those calibrations.

Until these pieces are verified, Sharpa motion capabilities remain disabled and
the existing executor must fail rather than claim a successful physical trial.
