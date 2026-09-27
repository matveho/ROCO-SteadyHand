# Architecture

## Design rule

Share task intent, not raw motor commands.

The shared layer should know that a part must be picked, transferred, inserted
or placed. Each robot layer decides how its own arm and end effector accomplish
that task.

~~~text
camera / task input
       |
board + part poses
       |
robot-independent PartGoal
       |
shared assembly phases
       |
RobotAdapter capability boundary
       |                     |
       v                     v
VegaAdapter              SharpaAdapter
DexMate arm +            North arm/body +
parallel gripper         Wave dexterous hand
~~~

## Shared data

steadyhand/models.py defines Pose and PartGoal. Poses are metres plus wxyz
quaternions. PartGoal intentionally contains no robot-specific joint values.

steadyhand/skills.py defines the common manipulation vocabulary:

1. open_gripper
2. approach_pick
3. descend_pick
4. grasp
5. lift
6. verify_grasp
7. transfer
8. approach_place
9. place
10. release
11. retreat
12. verify_place

This is a sequencing scaffold, not yet a physical controller. The current
dry-run command only logs these phases.

## Hardware boundary

steadyhand/adapters/base.py defines RobotAdapter. Vendor SDK imports belong
inside the robot-specific adapter and should not leak into task, calibration,
configuration or logging code.

The contract covers connection, stop, observation, joint motion, TCP motion,
end-effector commands and optional success verification.

Not every SDK natively exposes every capability. Vega, for example, exposes
joint-position control publicly, so TCP motion will require a validated local
IK layer rather than pretending the vendor provides Cartesian commands.

## Vega

VegaAdapter should eventually translate the common contract into:

- dexcontrol joint-state reads and joint-position commands;
- head and wrist camera reads;
- wrist wrench reads after units and frame are verified;
- the competition gripper's actual CAN driver;
- local IK and collision checking using the gripper-equipped robot model.

Do not use a robot model that omits the physical grippers for collision
clearance.

## Sharpa North

SharpaAdapter stays conservative until the onsite North interface is known.
Public Wave-hand resources and the RoCo demonstration dataset do not define the
full North arm/body API.

Once the onsite interface is available, record joint ordering and limits,
command semantics and rate, Wave hand representation, camera/tactile APIs,
stop behavior, and the provided robot description. Then implement this adapter
without rewriting the task-level logic.

## Coordinate frames

Calibration matrices use T_destination_source. T_base_camera therefore maps a
point expressed in the camera frame into the robot base frame.

Unknown transforms stay null. Never copy simulation coordinates into a real
robot configuration unless the physical frame relationship has been measured.

## Simulation versus reality

The submitted policy receives exact PartTarget information and simulator snap
logic. Real hardware must replace those conveniences with perception/board
registration, measured transforms, real grasp/insertion behavior, explicit
success verification and recovery.


## Vega camera layer

steadyhand/cameras/vega.py is intentionally usable without the motion-control
adapter. It implements:

- head ZED X Mini rectified left/right RGB and depth acquisition;
- both Sony ISX031 wrist RGB streams through WristCameras;
- runtime intrinsic extraction from get_camera_info();
- depth-to-XYZ reconstruction when a point cloud is needed.

The head is configured for 1920x1200 at 30 fps; competition guidance reports
approximately 24 Hz observed publication. The point cloud is reconstructed from
depth and runtime intrinsics rather than relying on a native point-cloud stream.

The wrist API exposes wrist_a and wrist_b. Do not map those labels to physical
left/right until verified on the competition unit.

Head and wrist timestamps are not assumed to share a clock or exposure trigger.
