# TODAY — Must-Do Checklist (Sep 27)

Goal: leave today with **both robots controllable from our code**, one repeatable Vega part working end-to-end, and the Sharpa control contract fully captured. Do not spend time on cleanup, refactors, or model training until these are true.

Official onsite schedule today: prep 10:00–14:00, **Sharpa bootcamp 14:00–15:00**, prep 15:00–16:00, **DexMate bootcamp 16:00–17:00**, prep 17:00–20:00.

## 1. Before bootcamps — repo + bring-up ready

- [ ] Pull latest `main`; run `python onsite.py doctor` and tests.
- [ ] Run `/usr/bin/python3 tools/vega_preflight.py` on the Vega computer if accessible.
- [ ] Have `docs/BOOTCAMP.md`, `docs/VEGA_LIVE.md`, and this checklist open.
- [ ] Copy `configs/runtime_targets.template.json` to a working targets file.
- [ ] Prepare one session folder and start recording every SDK version, path, frame name, command, failure, and successful setting.
- [ ] Decide the first Vega success target: **one easy open-release part** (battery first unless hardware geometry says otherwise).

## 2. Sharpa bootcamp 14:00–15:00 — do not leave without the control contract

- [ ] Obtain/copy the **North SDK/control example, URDF/robot description, and competition runner/example**.
- [ ] Record exact environment/setup: Python, packages, startup/homing, network/host, command process.
- [ ] Record exact **joint/state/action ordering, units, limits, command rate, and control semantics**.
- [ ] Record Wave hand command representation, joint ordering, limits, and tactile API.
- [ ] Record camera APIs/streams and timestamps.
- [ ] Physically identify **e-stop + software stop + recovery procedure**.
- [ ] Run vendor example unmodified.
- [ ] Read state only.
- [ ] Execute **one small safe arm motion**.
- [ ] Execute **one simple hand open/close or posture command**.
- [ ] Immediately fill `steadyhand/adapters/sharpa.py` with the confirmed API; do not leave it as a guessed stub.
- [ ] Save all organizer/vendor files locally before relying on venue internet.

**Sharpa exit criterion today:** our code can at minimum connect, read state, stop, send one safe arm command, and send one hand command.

## 3. DexMate bootcamp 16:00–17:00 — resolve every remaining Vega unknown

- [ ] Confirm exact `ROBOT_NAME`, installed `dexcontrol` version, and canonical vendor example.
- [ ] Confirm working arm and healthy gripper(s).
- [ ] Confirm gripper-equipped URDF path and **actual EE/TCP frame name**.
- [ ] Run/inspect joint-dump tool; record current joint order, limits, and any live Lift/torso value needed by IK.
- [ ] Confirm a safe nonzero `set_joint_pos(..., wait_time=...)` bring-up value.
- [ ] Demonstrate e-stop/software stop and recovery.
- [ ] Inspect `/home/dexmate/gripper.py`; implement the **per-side gripper API** if available.
- [ ] Verify CAN setup and homing behavior.
- [ ] Confirm wrench read works; determine or empirically calibrate frame/units enough to choose a conservative relative force threshold.

## 4. Vega no-motion validation — must pass before arm commands

- [ ] Head camera probe works: left RGB, right RGB, depth, runtime intrinsics.
- [ ] Wrist camera probe works; map `wrist_a` / `wrist_b` to physical wrists.
- [ ] Capture a full RGB-D + wrist snapshot with `tools/vega_capture_snapshot.py`.
- [ ] Run `tools/vega_ik_check.py` from the **actual current 7-joint state**.
- [ ] Verify FK pose is physically plausible.
- [ ] Verify a +1 cm IK target returns a nearby joint solution.
- [ ] Resolve every non-arm joint the IK chain requires; never leave Pinocchio using an assumed neutral value.

**Stop if FK/IK frame semantics are not clearly correct. Do not compensate with offsets by trial-and-error yet.**

## 5. Vega first physical motion — isolate layers

- [ ] With engineer/e-stop ready, command **one tiny unobstructed joint change**.
- [ ] Return to the known safe pose.
- [ ] Validate gripper homing, open, current-limited grip, and release with arm stationary.
- [ ] Implement/verify simple grasp detection from gripper position/status/current if the driver supports it.
- [ ] Confirm `VegaAdapter` state → command → feedback path behaves as expected.

## 6. Calibration/perception — produce usable runtime targets

- [ ] Establish/verify `T_base_camera` and the board frame.
- [ ] Produce a current **robot-base pick pose and place pose** for the first easy part.
- [ ] Replace legacy fixed-world EE offset with a calibrated `T_part_tcp` for that part as soon as practical.
- [ ] Confirm the generated TCP pose visually/kinematically before descending to the object.
- [ ] Save the working target/calibration snapshot in the session.

Do not attempt all nine objects before one target pipeline is trustworthy.

## 7. Get one Vega part reliable today

- [ ] Run `tools/vega_run_part.py` on the easy open-release part.
- [ ] Fix only the first failed layer: target pose → IK → trajectory → grasp → place.
- [ ] Repeat until the same part succeeds **at least 5 consecutive times** without manual intervention.
- [ ] Record failures and tune only measured parameters.
- [ ] Add recovery: failed grasp → retract/retry once; persistent failure → skip rather than jam.

**Vega exit criterion #1:** one easy part is boringly repeatable.

## 8. Expand score, not complexity

After the first reliable part:

- [ ] Bring up the other easy/open-release parts first: batteries/gears.
- [ ] Protect anything already reliable; do not destabilize good parts for marginal gains.
- [ ] Validate a safe return-home joint pose and use it between parts to stabilize IK branches.
- [ ] Run a short multi-part sequence and confirm failures do not destroy the rest of the run.

## 9. Physical insertion — only after open-release parts work

- [ ] Verify/debias wrench signal and set a conservative **relative** force limit.
- [ ] Pick one insertion part and validate pre-insert → small XY search → controlled Z descent → retract-on-contact.
- [ ] Add a real success criterion; **TCP reaching commanded depth alone is not enough**.
- [ ] Test repeated insertions before adding the remaining connector parts.
- [ ] Never defeat the force guard just to make the sequence continue.

## 10. Before leaving tonight

- [ ] Vega: at least one repeatable physical part, plus saved working config/targets/calibration.
- [ ] Sharpa: confirmed control API + state read + arm command + hand command implemented in adapter.
- [ ] Commit all confirmed code/config changes.
- [ ] Back up `runs/`, calibration, captured images, vendor examples/SDKs, URDFs, and notes to a second machine/location.
- [ ] Write the **single current blocker** for each robot and the first experiment for tomorrow morning.
- [ ] Re-run CI/tests after final commit.

## Do NOT spend today on

- broad refactoring or code prettification;
- training a new model before hardware control is proven;
- full automatic perception before one manual/semiautomatic target path works;
- speed optimization before reliability;
- tuning all nine parts in parallel;
- guessing missing SDK/frame/control values;
- making Sharpa a tomorrow problem — **both platforms count 1:1**.

## End-of-day definition of success

1. **Vega:** sensors + FK/IK + joint control + gripper + one repeatable part are real and logged.
2. **Sharpa:** full control contract is known and a minimal live adapter has performed a safe arm and hand command.
3. **Tomorrow starts with tuning/scaling**, not figuring out how either robot is controlled.
