# Submitted policy -> physical Vega mapping

The physical runner preserves the structure of policy.py rather than attempting
to execute the Isaac/Lula policy class on real hardware.

| Submitted simulation policy | Physical SteadyHand |
|---|---|
| PartTarget.pick_pos / place_pos | per-attempt runtime_targets.json object poses |
| target.ee_orientation | Vega skill grasp orientation / calibrated T_part_tcp |
| target.extra.ee_offset | Vega skill legacy EE offset / calibrated T_part_tcp |
| EEPathFollower | executor.move_tcp_segmented |
| Lula L_controller | PinocchioArmKinematics + dexcontrol joint commands |
| hover / descend / lift / transfer | executor physical phases |
| gripper_close simulation scalar | current-limited CAN gripper grip() |
| snap XY search | physical center-out XY insertion search |
| obs.snap_fired | no equivalent; force/TCP/vision success must replace it |
| return_home_q | optional validated --return-home-q |
| exact simulator target knowledge | perception/calibration -> runtime target poses |

## What is intentionally not copied

Simulation gripper radians are not sent to the real CAN grippers. The real
driver has its own calibrated 0..1 stroke and a current-limited object grip.

Simulation snap success is not copied. In simulation it can attach/teleport the
part; on hardware insertion is real contact and needs physical verification.

Simulation world coordinates are not copied. The onsite board and parts can
move, so current robot-base poses must be generated for each layout.

## Bring-up fallback versus final geometry

configs/skills/vega.json currently contains the submitted policy's EE offsets
and top-down orientation as a bring-up fallback. These reproduce the old policy
most closely when the real board orientation is similar.

The stronger representation is T_part_tcp: a rigid transform from each part
frame to the desired TCP. Once calibrated, it naturally rotates with a moved or
rotated board/part and should replace the legacy world-axis offset.
