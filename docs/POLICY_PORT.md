# Submitted policy to physical Vega

`policy.py` is preserved unchanged. The physical stack keeps its strategy but
replaces simulator-only services:

| Submitted policy | Physical implementation | Status |
|---|---|---|
| `PartTarget.pick_pos/place_pos` | measured runtime object/release poses | legitimate sim-to-real difference |
| `ee_offset` added in world axes | verified legacy offset or measured `T_part_tcp` | legacy semantics preserved; calibration gated |
| `ee_orientation` | measured URDF TCP orientation | must not copy raw simulation quaternion |
| relative world +Z hover/lift/retract | `offset_z` | preserved |
| Lula IK | Pinocchio LOCAL log/Jlog damped IK | replacement, with explicit base/EE/joint mapping |
| EE path steps | Cartesian interpolation, live IK reseed, segmented joint commands | replacement |
| simulation gripper scalar | verified CAN current and full selected-side open | legitimate; no unit conversion exists |
| nominal place then XY snap grid | nominal candidate followed by center-out grid | preserved ordering |
| `snap_fired` | force guard plus explicit physical seating confirmation | legitimate difference |
| return home before every part after first | validated `--return-home-q` before later parts | preserved when supplied |

The original Lula wrapper multiplies the user target quaternion by a 180-degree
USD-stage correction before solving against the URDF. Therefore the submitted
`[0,1,0,0]` cannot be sent directly to Pinocchio for an arbitrary Vega EE
frame. No correction is guessed in this repository.

The original `place_pos` is a release pose. For batteries and other open-release
parts it can differ from the final settled/grading pose. Runtime target capture
must preserve release clearance. For insertion parts the physical target is the
seated reference, approached through `preinsert_m`.

The submitted even-size bolt grid has no zero sample, but its phase builder
first performs a nominal place descent. The physical executor explicitly tries
zero before the configured grid to retain that behavior.

`usb_a` retract is 0.02 m because the submitted `FINAL_HEIGHT=None` falls back
to that part's `init_height`; the previous physical 0.1 m value was accidental
and has been corrected.

Unavoidable differences are physical force/contact, measured frames and target
poses, blocking dexcontrol joint trajectories, CAN homing/current, operator
verification, and the absence of simulator teleport/snap. These are explicit
gates rather than implicit success assumptions.
