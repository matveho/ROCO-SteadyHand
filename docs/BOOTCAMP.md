# First hardware access checklist

The goal of the first session is not a board attempt. It is to leave with a
known-good control path, exact interfaces, and one reproducible small motion.

## Before touching either robot

1. Pull main/onsite-2026 and record the current commit.
2. Create a session folder.
3. Run tools/capture_environment.py into the session folder.
4. Ask the engineer to demonstrate the physical e-stop and software stop.
5. Ask which files/examples they consider the canonical competition interface.
6. Copy those examples and record their paths/versions before editing anything.

## Vega U

First collect facts:

- exact ROBOT_NAME and current network path;
- installed dexcontrol version;
- which arm/gripper is healthy and intended for the board;
- actual gripper driver/example;
- gripper-equipped URDF/model path;
- current head/wrist camera status;
- permitted onboard vs workstation execution.

Then proceed in this order:

1. Run vendor health checks.
2. Run tools/vega_head_probe.py. This does not construct Robot().
3. Run tools/vega_wrists_probe.py; identify physical wrist_a/wrist_b.
4. With the engineer present, construct Robot() and note any automatic head
   motion.
5. Read joints/wrench/buttons.
6. Demonstrate stop.
7. Command one very small unobstructed joint-position change at low speed.
8. Return to a known safe pose.
9. Only then test gripper open/close with the arm stationary.
10. Record every confirmed API call in VegaAdapter.

Do not start by porting the full policy.

## Sharpa North

The first objective is to resolve the missing hardware contract.

Ask for or locate:

- North robot description / URDF;
- canonical Python control example;
- arm/body joint ordering and units;
- action representation and command frequency;
- Wave hand joint ordering and command units;
- head/wrist camera APIs;
- tactile API and force/torque meaning;
- startup/homing procedure;
- stop/e-stop API;
- whether the 65-D dataset state/action vector maps directly to the live API
  or requires conversion.

Then:

1. run the provided example unmodified;
2. print state only;
3. identify every state/action field;
4. demonstrate stop;
5. make one small safe arm motion;
6. make one simple hand open/close or posture command;
7. record the exact interface in SharpaAdapter.

Do not assume the public Wave SDK is the complete North controller.

## Before leaving each session

Record:

- working commit;
- SDK versions;
- robot model paths;
- joint names/order;
- confirmed physical frames;
- camera resolutions and timestamps;
- control rate;
- stop behavior;
- one known-good command;
- current blocker;
- next single experiment.

Back up the session folder before changing calibration or switching machines.
