# SteadyHand onsite workspace

Use the onsite-2026 branch for competition development. The exact original
submission is preserved on submission-baseline at commit
705e4e03ed8e3427cfc5f58a20de02afc3687e6b.

Generated runs, local data, and private planning in llm/ are gitignored; back
them up before switching machines.

## Offline commands

Run from the repository root. Python 3.10+ is enough for the current scaffold.

~~~bash
python onsite.py doctor
python onsite.py check-config --robot vega
python onsite.py dry-run --robot vega --part battery_size1
python onsite.py dry-run --robot sharpa --part pin --fail-at verify_grasp
python onsite.py new-session --robot vega --operator "Matvey Okoneshnikov"
python -m unittest discover -s tests -v
~~~

check-config accepts unfinished templates and lists missing fields. Add
--require-ready to fail when setup is incomplete. Passing that check means only
that the configuration is structurally complete.

dry-run never connects to hardware and does not simulate physics.

## Source of truth for the board task

configs/task_board.json is pinned to organizer repository commit
45dd6ad6e0792faf3450bdd2f81bb143b11bc43f, file task/param_config.py.

At that commit the code's part_order starts:

~~~text
gear_60teeth, gear_20teeth, rod_16mm, ...
~~~

The organizer's hand-maintained PARTS.md lists the two gears in the opposite
order. Their documentation says param_config.py is the source of truth, so this
repo follows the code. Re-check the pinned commit if the organizer repository
changes onsite.

Simulation snap labels are retained as task metadata only. They do not mean a
real robot receives the simulator's teleport/fixed-joint success mechanism.

## Repository map

| Location | Purpose |
|---|---|
| policy.py | Original submitted Vega simulation policy |
| onsite.py | Offline command-line entry point |
| steadyhand/models.py | Shared robot-independent poses and goals |
| steadyhand/skills.py | Shared assembly phase vocabulary |
| steadyhand/adapters/base.py | Common robot capability contract |
| steadyhand/adapters/vega.py | Vega hardware integration |
| steadyhand/adapters/sharpa.py | Sharpa North integration |
| configs/robots/ | Connection settings, joint order, motion limits |
| configs/task_board.json | Pinned task metadata and pose placeholders |
| configs/task_coordinates.json | Organizer task XY/Z data and 386 mm board registration metadata |
| tools/vega_task_coordinate_reachability.py | Supervised no-gripper reachability test for selected task points |
| calibration/ | Measured transforms for each robot |
| runs/ | Generated session snapshots and logs |
| docs/ARCHITECTURE.md | Shared-code / two-adapter design |
| docs/ARRIVAL.md | First physical robot session checklist |

## Session records

Each session contains session.json, a configuration snapshot, trials.csv, and
notes.md. Dry runs additionally write events.jsonl and result.json, explicitly
marked as mock results.

All task poses use metres and wxyz quaternions. Calibration matrices are
T_destination_source; T_base_camera maps camera coordinates into the robot base
frame. Unknown values remain null.

## Hardware enablement rule

The adapters are deliberately disabled. Before enabling one:

1. verify the installed SDK/version and robot identity;
2. verify joint ordering, limits, units, and control rate;
3. demonstrate stop/e-stop behavior;
4. read state without commanding motion;
5. make one small, unobstructed motion with the onsite engineer;
6. only then add a hardware-run path.

Do not make doctor, check-config, or imports contact a robot.


## Prepared onsite utilities

~~~bash
# Reproducibility snapshot; does not contact hardware.
python tools/capture_environment.py > environment.json

# Vega Jetson only: head stereo/depth subscription without Robot().
/usr/bin/python3 tools/vega_head_probe.py

# Vega Jetson only: local dual-wrist capture.
/usr/bin/python3 tools/vega_wrists_probe.py
~~~

The head probe intentionally uses the sensor manager rather than constructing
Robot(), because the supplied Vega field manual documents automatic head motion
during Robot() initialization. The wrist probe reports wrist_a/wrist_b only;
verify their physical left/right assignment onsite.

See docs/BOOTCAMP.md for the first-access order and docs/UPSTREAMS.md for pinned
public reference repositories.


## Vega camera API from SteadyHand

On the robot's system Python, camera-only access can now be used without
enabling arm/gripper control:

~~~python
from steadyhand.adapters.vega import connect_cameras
from steadyhand.cameras.vega import depth_to_point_cloud

robot = connect_cameras({}, head=True, wrists=True)
try:
    frames = robot.read_cameras()
    head = frames["head"]
    wrists = frames["wrists"]

    # Optional: sparse XYZ cloud in the rectified head-camera frame.
    xyz = depth_to_point_cloud(
        head.depth_m,
        head.camera_info,
        stride=4,
        max_depth_m=2.0,
    )
finally:
    robot.close()
~~~

Do not interpret this as synchronized four-camera capture: the head and wrist
timestamps use different time bases. Also verify wrist_a/wrist_b against the
physical left/right wrists before storing that mapping in config.
