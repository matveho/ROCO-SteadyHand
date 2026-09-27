# Upstream references

Snapshot checked 2026-09-27 UTC. These are references, not vendored
dependencies. Clone/download the pieces you actually need before relying on
venue Wi-Fi.

| Resource | Pinned reference | Why it matters |
|---|---|---|
| rocochallenge/RoCo_TaskBoardAssembly | 45dd6ad6e0792faf3450bdd2f81bb143b11bc43f | Released board task and simulation source of truth |
| intelligent-control-lab/dexmate-setup | 4be4d16140b25f730673d664b49c58280b07ef46 | Organizer-linked competition-specific Vega setup, drivers, gripper API, URDFs, calibration/reference files, pose tools |
| sharpa-robotics/sharpa-wave-sdk | release v5.0.11, published 2026-09-24 | Public Wave hand SDK packages |
| sharpa-robotics/sharpa-urdf-usd-xml | 0d19cac602f46456b819e4b6a2c09a74982c9a3e | Wave hand robot descriptions / collision models |
| sharpa-robotics/sharpa-rl-lab | 95ccda3d948801bb5da4cb7ffea766e03067a63b | Sharpa sim-to-real / deployment examples |
| ZhuoyangLiu2005/T-Rex | f88e10c61da123c68bf0927cf4860bc97a0381f3 | Real dexterous manipulation stack using Vega arms + Wave hands |

## Sharpa Wave SDK release

The current public release observed before onsite is v5.0.11. The release
provides amd64 Debian packages and an ARM zip. Do not install an arbitrary
package onto the competition robot; first inspect the installed version and
ask the onsite engineer whether the robot image may be modified.

## What is still not represented here

These public resources do not establish the complete Sharpa North full-body
control interface. The North-specific arm/body API, joint ordering, control
rate, robot description and competition runner must still be obtained or
inspected onsite.

## Offline copies

Prefer a sibling directory outside this repository, for example:

~~~text
roco-offline/
  RoCo_TaskBoardAssembly/
  sharpa-wave-sdk/
  sharpa-urdf-usd-xml/
  sharpa-rl-lab/
  T-Rex/
  dataset-metadata/
~~~

Record the exact commit/tag beside every copied resource. Do not commit large
SDK archives, model checkpoints, datasets or generated robot recordings into
ROCO-SteadyHand.


## Competition-specific DexMate Vega setup

The organizer explicitly directed teams on Sep 27 to review
intelligent-control-lab/dexmate-setup because the official competition Vega
configuration has been modified.

Treat this repository as the primary competition-specific Vega setup reference,
ahead of older generic assumptions. Important confirmed details include:

- system target: Orin Nano / JetPack-L4T stack described by the repo;
- CAN grippers use can1 at 1 Mbit/s;
- the official gripper driver exposes independent g.left / g.right Motor APIs;
- object gripping is Motor.grip(current=...), which returns a dict including
  a boolean gripped field;
- halt() stops gripper motion while preserving calibration; release() destroys
  the multi-turn reference and requires re-homing later;
- competition setup assets include gripper-equipped URDFs, calibration files,
  reference transforms, camera support, and safe pose utilities.

Verify the physical competition unit before copying source-robot calibration
values directly into live configuration.
