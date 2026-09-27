"""Vega integration TODOs.

Confirm the installed dexcontrol version and actual robot name/configuration.
Retrieve and inspect the onboard gripper.py; the fitted CAN grippers may not
use robot.left_hand. Choose the working arm after checking hardware condition.
The linked field manual says Robot() initialization moves the head.

Implement separately: observation reads, joint motion, gripper control, and stop.
Use calibrated frames and the actual gripper-equipped robot model.
"""


def connect(config):
    raise NotImplementedError(
        "Vega hardware integration is not implemented. Use the offline dry-run."
    )
