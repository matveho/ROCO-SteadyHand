# EUNICE — Sharpa North Must-Do Today

**Venue:** Exhibit Hall, COMP7. **Sharpa training: 2:00–3:00 PM.** Sponsor technicians are available throughout the afternoon, so continue asking them concrete interface questions after the formal session if needed.

**Owner: Eunice**  
**Robot: Sharpa North only**  
**Goal for tonight:** remove the mystery around North and leave us with a minimal, real, tested hardware adapter.

You do **not** need to solve the whole board today.

Your most valuable job is to answer:

> **Exactly how do we safely observe and command this specific North robot in the competition?**

Right now our repo intentionally refuses to command North because we do not yet trust the full-body interface. That is good. Your job is to replace unknowns with confirmed facts.

---

# 1. Mental model: what you are trying to uncover

Think of the software chain as:

```text
our Python code
      ↓
Sharpa/North SDK or competition runner
      ↓
arm/body commands + Wave hand commands
      ↓
robot
      ↓
joint state + cameras + tactile
```

We currently know useful things about the **Wave hand** and have a RoCo demonstration dataset.

That does **not** prove how the complete **North robot** is controlled.

Do not assume:

- public Wave SDK = complete North SDK;
- dataset state/action ordering = live robot ordering;
- demonstration values can be copied directly into live commands;
- a public North product DOF number = competition action-vector length.

Everything live must be confirmed from the actual competition system or vendor example.

---

# 2. Your end-of-day success criterion

By tonight we want our repo to be able to do, from code we understand:

```text
connect
  ↓
read robot state
  ↓
print named joints
  ↓
read hand/tactile state
  ↓
STOP safely
  ↓
send ONE tiny safe arm command
  ↓
send ONE simple Wave-hand command
  ↓
read the resulting state
  ↓
shutdown cleanly
```

If you achieve that, today was successful even if North has not picked up a board part yet.

A very strong day would additionally get one simple grasp or one easy part attempt.

---

# 3. Before the Sharpa bootcamp

Open these files:

```text
steadyhand/adapters/sharpa.py
configs/robots/sharpa.json
steadyhand/adapters/base.py
docs/BOOTCAMP.md
docs/EUNICE_SHARPA_TODAY.md
```

Current `sharpa.py` is a deliberate safety stub. It should fail rather than guess.

Do not copy values into it until they are confirmed.

Create a notes file/session and record everything as you learn it.

For every fact, record **source + exact value**.

Example:

Bad:

```text
"actions are joint positions"
```

Good:

```text
Source: vendor demo north_control.py
Function: robot.send_action(...)
Action length: 65
Units: ...
Order: [ ... ]
Rate: 30 Hz
Blocking/nonblocking: ...
Safe stop: ...
```

---

# 4. At the Sharpa bootcamp — ask for these FIRST

Do not passively listen. Get files and concrete examples.

## A. Complete competition software package

Ask:

> "What is the canonical example that you expect competitors to use to control North?"

Obtain/copy:

- [ ] North SDK/package
- [ ] canonical Python control example
- [ ] competition deployment/runner example
- [ ] North URDF or equivalent robot description
- [ ] configuration files
- [ ] environment/requirements/install instructions
- [ ] any launch/startup script
- [ ] any provided calibration
- [ ] any provided task-specific example

Record exact versions/commits if available.

Save them locally immediately.

Do not trust venue internet to remain perfect.

---

# 5. Determine exactly where code runs

Ask and record:

- [ ] Run code onboard North or from our laptop?
- [ ] Required OS/Python version
- [ ] Required virtualenv/conda environment
- [ ] Required host/IP/network variables
- [ ] Required credentials/certificates if any
- [ ] Startup process
- [ ] Homing process
- [ ] Required background services
- [ ] Shutdown process

You want to be able to reproduce startup from a fresh terminal.

Write the exact command sequence.

Example structure:

```bash
source ...
export ...
python ...
```

Do not paraphrase it if you can copy the actual commands.

---

# 6. SAFETY — do not continue without this

Have the engineer physically show you:

- [ ] physical e-stop;
- [ ] software stop;
- [ ] what happens when commands stop arriving;
- [ ] what happens if Python crashes;
- [ ] how to recover after e-stop;
- [ ] safe/home posture;
- [ ] forbidden poses/regions if any.

Then test the software stop with the engineer.

You should know which exact call our adapter's:

```python
stop()
```

must invoke.

If you do not know how to stop it, do not send motion.

---

# 7. Understand the STATE vector completely

Run the vendor example **unmodified** first.

Then print/read state only.

Find:

- [ ] total state dimension
- [ ] exact joint names
- [ ] exact order
- [ ] units
- [ ] left/right arm dimensions
- [ ] Wave hand dimensions
- [ ] body/torso/head dimensions
- [ ] velocity fields
- [ ] effort/torque fields
- [ ] timestamps
- [ ] update frequency

If state is a flat vector, construct a table:

| Index | Name | Physical joint/signal | Units |
|---|---|---|---|
| 0 | ... | ... | ... |
| 1 | ... | ... | ... |

Do not stop at "65 values."

We need to know what each value means.

---

# 8. Understand the ACTION vector completely

This is even more important than state.

Find:

- [ ] exact action dimension
- [ ] exact ordering
- [ ] units
- [ ] absolute vs relative
- [ ] position vs velocity vs other control
- [ ] required command rate
- [ ] blocking vs streaming
- [ ] whether all joints must be sent every command
- [ ] whether arm and hand are separate commands
- [ ] interpolation behavior
- [ ] command timeout/watchdog behavior
- [ ] joint limits
- [ ] speed/step limits
- [ ] safe way to command a tiny motion

Make another table:

| Action index | Commanded joint | Units | Min | Max |
|---|---|---|---|---|

### Critical dataset question

Ask explicitly:

> "Does the released RoCo Sharpa dataset's state/action vector map directly to the live North control API? If not, what conversion is required?"

Record the answer.

Do **not** infer this from matching vector lengths.

---

# 9. Understand the Wave hand

North's dexterous hand is not a simple open/close gripper.

Find exactly:

- [ ] hand joint names/order
- [ ] number of commanded hand values
- [ ] units
- [ ] joint limits
- [ ] default/open posture
- [ ] simple safe grasp posture
- [ ] whether there are high-level named postures
- [ ] whether commands are raw joint positions or another representation
- [ ] command rate
- [ ] whether vendor SDK performs collision/current limiting
- [ ] how to stop hand motion

Your minimum test today:

```text
known safe open posture
→ small/simple closing posture
→ open again
```

Do not start with a complicated power grasp.

---

# 10. Understand tactile data

Find:

- [ ] API call
- [ ] number of tactile sensors
- [ ] which sensor corresponds to which fingertip
- [ ] fields per sensor
- [ ] units
- [ ] coordinate frame
- [ ] rate
- [ ] timestamp behavior
- [ ] whether data are already calibrated/bias-corrected

The released dataset suggests rich fingertip information, but the **live API is authoritative**.

Do one intuitive experiment:

1. read tactile with no contact;
2. gently touch one fingertip;
3. identify which values change;
4. record that fingertip mapping.

We do not need sophisticated tactile control today.

We do need to know that our code can read it correctly.

---

# 11. Cameras

Find:

- [ ] head camera API
- [ ] wrist camera API(s)
- [ ] resolutions
- [ ] RGB/BGR convention
- [ ] depth availability
- [ ] intrinsics
- [ ] timestamps
- [ ] synchronization with robot state
- [ ] camera frame names/extrinsics if supplied

Capture at least one frame from every competition camera.

Save the samples locally.

---

# 12. Robot description / kinematics

Obtain the actual North description.

Record:

- [ ] URDF/USD/XML path
- [ ] root/base frame
- [ ] left/right arm joint names
- [ ] wrist/hand frames
- [ ] fingertip frames if present
- [ ] camera frames
- [ ] collision geometry availability
- [ ] joint limits

Do not use a Wave-hand-only URDF as if it described the full North robot.

---

# 13. First safe ARM motion

Only after:

- e-stop known;
- state mapping known;
- action mapping known;
- limits known;
- engineer present.

Procedure:

1. read current arm q;
2. copy it;
3. alter **one clearly safe joint** by a very small amount;
4. send the command using vendor's canonical method;
5. observe physical direction;
6. verify state changed correspondingly;
7. return to original q.

Record the exact working code.

If the physical joint does not match the index/name you expected:

**stop immediately and fix the mapping.**

---

# 14. First safe HAND motion

Use a known vendor-provided posture if possible.

Procedure:

1. read current hand state;
2. send safe open posture or tiny finger change;
3. observe;
4. read state again;
5. return/open.

Then, if vendor approves:

- simple close/open;
- touch a harmless object;
- observe tactile response.

Do not invent a 22-joint grasp vector from the dataset.

---

# 15. Convert confirmed information into our repo

Once the vendor example works, start replacing the stub.

Primary files:

```text
configs/robots/sharpa.json
steadyhand/adapters/sharpa.py
```

You may add focused tools such as:

```text
tools/sharpa_preflight.py
tools/sharpa_state_probe.py
tools/sharpa_small_motion.py
tools/sharpa_hand_probe.py
tools/sharpa_camera_probe.py
```

Avoid changing shared Vega code unless absolutely necessary.

## Fill `configs/robots/sharpa.json`

Populate only facts you verified:

- robot_name
- host
- sdk_version
- urdf_path
- joint_names
- limits
- command rate
- hand joint names
- hand command units
- any required environment/config paths

If a fact remains unknown, leave it null and document the blocker.

## Implement `SharpaAdapter`

Map our common contract onto the real API:

```python
connect()
observe()
stop()
move_joints(...)
close()
```

For hands, our current generic `open_gripper/close_gripper` naming is not ideal for a dexterous hand.

For today, it is acceptable to map these to **verified safe named/open/simple grasp postures** if that is genuinely what the SDK supports.

Do not pretend arbitrary dexterous manipulation fits a binary gripper model.

If the SDK exposes a better hand method, add a Sharpa-specific method cleanly.

---

# 16. Add a NO-MOTION probe before a full runner

Before creating any task policy, make a script that does only:

```text
connect
print versions
print named joints/state
print hand state
print tactile summary
optional camera metadata
disconnect
```

It should send no motion.

Run it multiple times.

This becomes tomorrow's first diagnostic command.

---

# 17. Create one tiny motion test

Make a deliberately separate script that requires an explicit confirmation flag.

It should:

1. connect;
2. read current q;
3. command one tiny verified motion;
4. read q;
5. return;
6. stop/shutdown cleanly on exception.

This proves our adapter, not only the vendor demo.

---

# 18. If time remains: one simple object interaction

Only after the control contract is solid.

Do **not** try to build the final sophisticated manipulation policy today.

Choose something simple:

```text
move hand above easy object
→ open posture
→ lower
→ simple close/grasp posture
→ lift slightly
→ release
```

The purpose is to learn:

- arm positioning accuracy;
- whether the hand posture grasps the competition parts;
- whether tactile is useful;
- where the main physical difficulty lies.

Log everything.

---

# 19. How to use the released RoCo Sharpa dataset

The dataset is useful **after** the live mapping is known.

Possible uses:

- compare live joint/state ranges against demonstrations;
- extract typical arm configurations near board parts;
- inspect typical Wave hand postures;
- learn which fingertips contact which objects;
- later train/finetune or imitation-learn.

But first create an explicit mapping:

```text
dataset field/index
        ↓
meaning
        ↓
live SDK field/index
```

If you cannot prove the mapping, do not send dataset actions to hardware.

---

# 20. Questions to ask the Sharpa engineer verbatim

If conversation is rushed, use these.

1. **"Can you show me the exact Python example you expect teams to use to control the full North robot?"**
2. **"Which file/package is the authoritative full-body North SDK? Is the public Wave SDK only for the hand?"**
3. **"What exactly is the live state vector? Can you give me the joint names and order?"**
4. **"What exactly is the live action vector? Absolute positions? What units and update rate?"**
5. **"Does the RoCo dataset action vector map directly to this live API?"**
6. **"Which full North URDF/description should we use for FK/IK and collision geometry?"**
7. **"What is the safest way to command one tiny arm motion from the current pose?"**
8. **"What is the safest known open and simple grasp posture for the Wave hand?"**
9. **"How do I read fingertip tactile data and which entries correspond to which fingers?"**
10. **"What software call immediately stops motion, and what happens if our process dies?"**
11. **"Are cameras synchronized with joint/tactile state? What are their APIs and frames?"**
12. **"Is there a competition-specific runner/config we should use instead of the public examples?"**

Do not leave with "it's in the docs" if you can instead get the exact file/path/example.

---

# 21. What NOT to do today

- Do not work on DexMate. Matvey owns it.
- Do not train a model before we can command North safely.
- Do not spend the bootcamp reading docs silently; ask concrete interface questions.
- Do not guess action ordering.
- Do not paste 65-D dataset actions into hardware.
- Do not assume North = Wave hand API only.
- Do not try full board assembly before state/action mapping works.
- Do not make broad shared-architecture refactors.
- Do not hide unknowns by putting plausible-looking numbers in config.

---

# Required end-of-day deliverables

## MUST HAVE

- [ ] North SDK/control package copied locally
- [ ] Canonical full-robot Python example copied locally
- [ ] Full North robot description/URDF copied locally
- [ ] Startup/environment commands written down
- [ ] Physical + software stop procedure proven
- [ ] Exact state ordering understood
- [ ] Exact action ordering understood
- [ ] Units and control rate understood
- [ ] Wave hand command format understood
- [ ] Tactile read demonstrated
- [ ] One safe arm motion demonstrated
- [ ] One safe hand motion demonstrated
- [ ] `configs/robots/sharpa.json` populated with verified facts
- [ ] `steadyhand/adapters/sharpa.py` uses the real API instead of the safety stub
- [ ] A no-motion state probe exists and works
- [ ] All vendor files/examples backed up

## STRONG TARGET

- [ ] Camera capture works
- [ ] Tiny-motion tool works through our adapter
- [ ] Simple hand grasp posture identified
- [ ] One easy object picked/lifted/released
- [ ] Dataset ↔ live state/action mapping documented

---

# Send Matvey this handoff before leaving

Copy/paste this template and fill every line:

```text
SHARPA NORTH HANDOFF

SDK/package:
SDK version/commit:
Python/environment:
Startup command:
Host/network:
Canonical example:
North URDF:
Base frame:

STATE
dimension:
rate:
units:
joint order saved at:

ACTION
dimension:
rate:
control mode:
absolute/relative:
units:
action order saved at:

LEFT ARM:
RIGHT ARM:
HAND representation:
safe open posture:
simple grasp posture:

TACTILE
API:
rate:
finger mapping:
units/frame:

CAMERAS
APIs:
streams:
frames/timestamps:

SAFETY
physical e-stop:
software stop call:
process-crash behavior:
recovery:

PROVEN TESTS
state read:
tiny arm motion:
hand motion:
tactile read:
camera read:
simple grasp:

REPO
working commit:
files changed:
known blocker:
first experiment tomorrow:
```

If this handoff is complete, Matvey should be able to understand the North control path without sitting through the bootcamp himself.
