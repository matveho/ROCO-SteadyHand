# MATVEY — DexMate Vega U Must-Do Today

**Owner: Matvey**  
**Robot: DexMate Vega U only**  
**Goal for tonight:** the complete Vega path is real, repeatable, and ready to scale tomorrow.

Do not spend today helping with Sharpa unless Eunice has a blocker that only you can resolve. Eunice owns North.

Your job is to turn this chain into something physically proven:

```text
camera / current layout
        ↓
board + part pose
        ↓
runtime target
        ↓
part → TCP grasp transform
        ↓
Cartesian waypoints
        ↓
Pinocchio IK
        ↓
safe 7-joint commands
        ↓
dexcontrol
        ↓
CAN gripper
        ↓
physical pick/place
        ↓
verification + recovery
```

The day is successful if this is no longer a diagram: **at least one easy part works repeatedly end-to-end, and every layer below it is understood and logged.**

---

## 0. Rules for today

1. **Fix the first broken layer only.** Do not tune grasping if FK is wrong.
2. **One arm, one easy part first.** Use a battery/open-release part unless the physical layout makes another part clearly easier.
3. **No guessed frames or dimensions.** Measure or get them from the robot/vendor.
4. **Reliability before speed.**
5. **Never defeat a safety gate just to continue.**
6. **Save every working state.** Commit + config + calibration + target file + notes.
7. Once something succeeds repeatedly, protect it. Do not casually rewrite it.

---

# PHASE 1 — Establish the exact robot interface

## Read this organizer-linked repo first

The organizers explicitly said the official Vega configuration was modified and
directed teams to:

`https://github.com/intelligent-control-lab/dexmate-setup`

Pinned reference checked today:

`4be4d16140b25f730673d664b49c58280b07ef46`

Clone/copy it locally before relying on venue internet.

This repo already confirms several things we previously planned to ask:

- CAN adapter is `can1` at 1 Mbit/s;
- motor 1 = left gripper, motor 2 = right gripper;
- `Grippers().left` and `.right` are independent Motor objects;
- use `Motor.grip(current=...)` for objects;
- `grip()` returns a dict including `gripped`;
- `halt()` stops while preserving calibration;
- `release()` de-energizes and destroys the multi-turn position reference;
- gripper-equipped URDFs and calibration/reference files are included;
- vendor pose/safety scripts and joint-inspection tools are included.

Our SteadyHand gripper wrapper has been updated to this API. Still verify the
actual competition unit before motion.

Do this before serious motion.

## Ask/verify at the DexMate bootcamp

Write the answers directly into your notes/config.

- [ ] Exact `ROBOT_NAME`
- [ ] Confirm installed `dexcontrol` version is `0.5.0` or record the actual version
- [ ] Canonical vendor Python example for arm state and joint-position motion
- [ ] Which physical arm should we use?
- [ ] Which gripper(s) are healthy?
- [ ] Exact gripper-equipped URDF path
- [ ] Exact EE/TCP frame name in that URDF
- [ ] Exact 7 arm joint names/order
- [ ] Any non-arm movable joints in the EE kinematic chain, especially Lift/torso-like joints
- [ ] Their **actual physical values**
- [ ] Safe nonzero `set_joint_pos(..., wait_time=...)` value for bring-up
- [ ] Physical e-stop
- [ ] Software stop
- [ ] Recovery after e-stop
- [ ] What happens if Python crashes mid-command?
- [ ] Exact CAN interface/startup command for grippers
- [x] Per-side gripper API exists in the organizer-linked driver (`g.left` / `g.right`); verify the same file/API is installed on the competition unit
- [ ] Wrench units/reference frame if vendor knows them
- [ ] Whether code should run directly on the Jetson or from a workstation

### Do not leave the bootcamp with vague answers

Bad note:

```text
"gripper works through CAN"
```

Good note:

```text
CAN interface: can1
driver: /home/dexmate/gripper.py
class: Grippers
startup: ...
left-only open call: ...
left-only grip call: ...
home required after power cycle: yes
```

---

# PHASE 2 — Software and sensors with ZERO intended arm motion

From the repo root:

```bash
git pull
git status
git rev-parse HEAD
python onsite.py doctor
python onsite.py check-config --robot vega
/usr/bin/python3 tools/vega_preflight.py
```

- [ ] Record the current Git commit.
- [ ] Fix import/version/path failures before controlling hardware.
- [ ] Do not modify `submission-baseline`.

## Head camera

```bash
export ROBOT_NAME=<actual-name>
/usr/bin/python3 tools/vega_head_probe.py
```

Confirm:

- [ ] left RGB
- [ ] right RGB
- [ ] depth
- [ ] runtime camera info/intrinsics
- [ ] image shapes make sense

Then:

```bash
/usr/bin/python3 tools/vega_capture_snapshot.py --output ~/vega-first-snapshot
```

Keep this snapshot. It is your first real perception fixture.

## Wrist cameras

```bash
/usr/bin/python3 tools/vega_wrists_probe.py
```

- [ ] Both streams work.
- [ ] Physically cover one wrist camera.
- [ ] Determine exactly which physical wrist is `wrist_a` and which is `wrist_b`.
- [ ] Record that mapping.

### Gate to continue

Do not proceed until sensors can be read predictably and you know the e-stop.

---

# PHASE 3 — Prove the kinematic model before using it to move

This is one of today's highest-risk steps.

You need:

- actual current 7-joint arm state;
- actual gripper-equipped URDF;
- actual EE/TCP frame;
- actual values for every non-arm movable joint in the chain.

Use the existing robot joint inspection tool/vendor example to read the actual q.

Then run **offline/no motion**:

```bash
/usr/bin/python3 tools/vega_ik_check.py \
  --urdf <actual-gripper-urdf> \
  --ee-frame <verified-frame> \
  --arm <left-or-right> \
  --fixed <extra-joint>=<actual-value> \
  --q q1 q2 q3 q4 q5 q6 q7 \
  --dz 0.01
```

Check manually:

- [ ] FK position/orientation is physically plausible.
- [ ] +1 cm Z target produces a **nearby**, not wildly different, q.
- [ ] IK solution remains inside joint limits.
- [ ] FK of the solved q lands near the requested +1 cm target.
- [ ] No hidden Lift/torso joint is silently assumed.

### STOP CONDITION

If FK says the gripper is somewhere physically absurd, or +1 cm produces a large/strange arm reconfiguration:

**do not command that IK solution.**

Resolve frame/joint/model semantics first.

---

# PHASE 4 — First intentional arm motion

Engineer nearby. Workspace empty. Physical e-stop immediately reachable.

First:

- [ ] Construct the vendor `Robot()` only after acknowledging that initialization may move the head.
- [ ] Read current joint state.
- [ ] Read current wrench.
- [ ] Verify software e-stop call.
- [ ] Verify recovery from software e-stop.

Then command only:

```text
current q
→ current q + tiny safe change in ONE unobstructed joint
```

- [ ] Low speed / conservative wait time.
- [ ] Observe physical direction.
- [ ] Confirm reported joint state follows command.
- [ ] Return to the original safe q.

### Gate to continue

You must trust:

```text
our q ordering
→ dexcontrol
→ expected physical joint
→ feedback
```

before any TCP/IK motion.

---

# PHASE 5 — Make the CAN gripper boring

Arm stationary and safely parked.

First run the known vendor gripper example unmodified.

Then determine:

- [ ] startup/CAN bring-up
- [ ] homing behavior
- [ ] open
- [ ] current-limited object grip
- [ ] release
- [ ] halt
- [ ] position read
- [ ] current/status read
- [x] Organizer repo exposes left-only/right-only `Motor` API; verify it on the physical unit

Update:

```text
steadyhand/grippers/vega.py
```

if the actual API differs.

## Add real grasp verification

Do not leave `verify_grasp()` permanently as unknown if the driver exposes useful data.

Simple useful logic is enough:

```text
command grip
↓
read jaw position/status/current
↓
did jaws stop on an object rather than fully close?
```

- [ ] Calibrate empty-close reading.
- [ ] Calibrate known-object grip reading.
- [ ] Implement a conservative grasp-present check if data permits.

### Gate to continue

You can deliberately:

```text
open → grip object → detect object → release
```

with the arm stationary.

---

# PHASE 6 — Prove TCP motion in free space

Before touching a competition part:

- [ ] Use current q → FK to get current TCP.
- [ ] Request a small free-space TCP displacement, e.g. +1 cm in a clearly safe direction.
- [ ] Run through `VegaAdapter.move_tcp()`.
- [ ] Confirm the physical gripper moves in the expected Cartesian direction.
- [ ] Return.
- [ ] Repeat along another axis if safe.

Verify that segmented Cartesian motion does not produce strange elbow/joint jumps.

### Important limitation

Current code does not provide full collision-aware planning.

Therefore:

- keep unused arm parked;
- keep generous board clearance;
- use conservative hover heights;
- do not trust a valid IK solution to imply a collision-free path.

---

# PHASE 7 — Generate one trustworthy real target

Do not attack all nine objects.

Pick one easy open-release part.

You need:

```text
T_base_part_pick
T_base_part_place
```

and a desired grasp/TCP transform.

## First acceptable version

A manual/semiautomatic target is fine for bring-up.

The point is to prove the entire control stack, not finish automatic perception first.

Copy:

```text
configs/runtime_targets.template.json
```

into a session-specific target file.

Fill only the chosen part first.

- [ ] Pick pose is in **robot_base**, not simulator/world coordinates.
- [ ] Place pose is in **robot_base**.
- [ ] Verify the pose numerically.
- [ ] Verify it visually against camera/robot geometry.

## Improve grasp geometry

The current config can fall back to old simulation-style world-axis offset/orientation.

That is only a bring-up approximation.

Prefer to calibrate:

```text
T_part_tcp
```

for the first part.

Why:

```text
part rotates
↓
T_part_tcp rotates with it
↓
grasp remains correct
```

A fixed XYZ world offset does not have that property.

---

# PHASE 8 — First physical pick/place

Use an open-release part first.

Run:

```bash
/usr/bin/python3 tools/vega_run_part.py \
  --targets <current-targets.json> \
  --part battery_size1 \
  --operator Matvey \
  --working-arm <left-or-right> \
  --robot-name '<actual-name>' \
  --urdf <actual-gripper-urdf> \
  --ee-frame <actual-frame> \
  --step-wait <verified-value> \
  --gripper-scope <left-or-right> \
  --confirm-head-motion \
  --confirm-physical-motion
```

If single-side gripper control has been implemented, use the updated interface/config instead.

## Debug in this order only

If it fails, identify the first wrong layer:

1. **Target wrong?**
2. **TCP transform wrong?**
3. **IK wrong?**
4. **Path wrong?**
5. **Grasp wrong?**
6. **Transfer wrong?**
7. **Place wrong?**
8. **Verification wrong?**

Do not compensate for a wrong target by changing IK tolerances.

Do not compensate for wrong FK by changing grasp offsets.

---

# PHASE 9 — Turn one success into reliability

One success is not the milestone.

The milestone is:

**5 consecutive successful runs of the same part without manual rescue.**

For each failure record:

- exact target file;
- calibration;
- code commit;
- failure phase;
- whether grasp verification passed;
- physical reason;
- corrective change.

Tune the minimum parameter that explains the failure.

## Add recovery

Before moving to a full sequence:

- [ ] failed grasp → retract
- [ ] retry once
- [ ] still failed → safe skip/abort behavior
- [ ] never continue carrying an object you do not believe you grasped

---

# PHASE 10 — Expand score in the easiest order

Only after one part is stable.

Suggested initial progression based on manipulation complexity:

1. batteries
2. gears
3. rod / other precise placement
4. pin / bolt
5. USB / HDMI

Actual physical observations override this order.

For each new part:

```text
target
→ T_part_tcp
→ isolated run
→ repeated success
→ add to sequence
```

Do not tune all parts at once.

---

# PHASE 11 — Validate return-home and multi-part execution

The original submitted policy benefited from a consistent IK seed.

- [ ] Identify a physically collision-safe return-home q.
- [ ] Validate path to/from it.
- [ ] Save exact q.
- [ ] Run two known-good parts sequentially.
- [ ] Confirm the first part does not poison the second part's IK branch.
- [ ] Confirm a failed part does not destroy the rest of the sequence.

Then use:

```text
tools/vega_run_sequence.py
```

only when isolated skills are already trustworthy.

---

# PHASE 12 — Contact/insertion work LAST

Only after open-release scoring works.

For wrench:

- [ ] record unloaded baseline;
- [ ] move arm through free space and observe drift;
- [ ] gently make controlled contact with engineer approval;
- [ ] determine which force axes respond;
- [ ] determine usable relative threshold;
- [ ] keep it conservative.

Then test exactly one insertion part:

```text
pre-insert
→ center XY
→ controlled Z
→ force check
→ retract on excessive contact
→ small XY candidate
→ retry
```

### Critical

**TCP reached commanded depth is not sufficient proof of successful insertion.**

Add a real success indicator if possible:

- final visual pose;
- expected insertion depth + force signature;
- gripper release behavior;
- another physically meaningful observation.

Never disable the force guard just because the connector is difficult.

---

# What NOT to spend time on today

- Sharpa integration — Eunice owns it.
- General code cleanup.
- Rewriting the architecture.
- New learned policy training before deterministic Vega control works.
- Speed optimization before reliability.
- Sophisticated automatic perception before one real target pipeline works.
- All nine parts at once.
- Aesthetic documentation work.

---

# Required end-of-day deliverables

Do not leave until you have as many of these as physically possible.

## MUST

- [ ] Exact Vega SDK/environment recorded
- [ ] Actual gripper URDF + EE frame identified
- [ ] FK/IK validated against the physical arm
- [ ] One safe joint command proven
- [ ] Gripper open/grip/release proven
- [ ] Head RGB/depth captured
- [ ] Wrist cameras mapped
- [ ] One real runtime target produced
- [ ] One physical open-release part completed end-to-end
- [ ] Same part repeated successfully multiple times
- [ ] Working config/targets/calibration committed/backed up

## STRONG TARGET

- [ ] 5/5 consecutive on first part
- [ ] second/third easy parts working
- [ ] real grasp verification
- [ ] safe return-home q
- [ ] short multi-part sequence
- [ ] first force-gated insertion test

---

# What to hand tomorrow morning to yourself

One note containing only:

```text
WORKING COMMIT:
ROBOT_NAME:
WORKING ARM:
DEXCONTROL VERSION:
URDF:
EE FRAME:
FIXED CHAIN JOINT VALUES:
STEP WAIT:
GRIPPER API:
SAFE HOME Q:
CAMERA MAPPING:
CALIBRATION FILE:
TARGET FILE:
RELIABLE PARTS + SUCCESS RATE:
KNOWN FAILURES:
NEXT SINGLE EXPERIMENT:
```

If that sheet is complete, tomorrow is a tuning/scoring day rather than another integration day.
