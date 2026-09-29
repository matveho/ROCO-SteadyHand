# SteadyHand

Our entry for the Industrial Board Assembly track at RoCo @ IROS 2026 in Pittsburgh.

policy.py is the submitted Vega simulation policy. Real-robot integration lives
in the steadyhand package and is deliberately separated from the submission.

## Branches

- submission-baseline: exact original submission snapshot at
  705e4e03ed8e3427cfc5f58a20de02afc3687e6b. Never develop on this branch.
- onsite-2026: active competition integration work.
- main: kept fast-forwarded to the current integration head for easy cloning.

The original policy.py remains unchanged from the submission snapshot.

## Architecture

Shared code owns task definitions, calibration, logging and the manipulation
vocabulary. Robot-specific code is isolated behind one adapter contract.

~~~text
task / perception / calibration
            |
       shared skills
            |
      RobotAdapter
       /        \
   Vega          Sharpa
 dexcontrol     North/Wave
~~~

See docs/ARCHITECTURE.md.

## Offline preparation

~~~bash
python onsite.py doctor
python onsite.py check-config --robot vega
python onsite.py dry-run --robot vega --part battery_size1
python -m unittest discover -s tests -v
~~~

The hardware adapters are intentionally disabled until each competition robot's
installed control path and stop behavior have been verified. A dry run never
contacts hardware and never claims physical success.

See docs/ONSITE.md for the remaining repository-wide workflow notes.

## Offline board annotation

For a supervised initial/final photograph measurement workflow, use
[`tools/board_annotation_tool.py`](tools/board_annotation_tool.py). It creates
an audit project and a board-local `task_coordinates` export that can be
reviewed before applying the existing live board calibration. The complete
workflow and coordinate convention are documented in
[`docs/BOARD_ANNOTATION_TOOL.md`](docs/BOARD_ANNOTATION_TOOL.md).


## Vega live stack

The Vega path is now implemented below perception/target generation:

~~~text
runtime object/target poses
        |
Vega skill config
        |
physical executor
        |
segmented Cartesian targets
        |
Pinocchio IK
        |
stepped 7-DoF joint targets
        |
dexcontrol 0.5.0
        |
physical Vega arm

separate paths:
  Sensors/WristCameras -> RGB-D + wrist RGB
  gripper.py/SocketCAN -> third-party parallel grippers
~~~

Start with [docs/FOR_LLMS.txt](docs/FOR_LLMS.txt), which is the compact current
Vega handoff and runbook. The single-part runner is `tools/vega_run_part.py`;
`tools/vega_run_sequence.py` is a later multi-part scaffold.

The active physical path is the calibrated right-arm competition pipeline:
head-camera board registration, flat 100 mm TCP hover, and the `wrist_a`
right-wrist precision stage. Battery grasp/place remains gated on physical
teaching and validation; no unverified pick/place result is implied.
