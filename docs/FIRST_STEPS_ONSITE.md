# Vega first steps onsite

This is the short operational runbook for the current right-arm competition stack.

Use the hardware E-stop during physical work. The software has IK/floor/provenance
guards but **no general collision checker**.

The normal path is:

```text
pull/deploy
-> offline checks
-> fresh board registration + high-clearance position checks
-> teach one wrist profile
-> verify pick
-> verify pick/place
-> add verified parts to the competition run
```

Do **not** change the permanent board calibration to correct an individual part
miss. Part-specific tuning belongs in the wrist profile / competition plan.

## 0. Pull / deploy

Pull/deploy with the normal tool.

On the Vega:

```bash
cd ~/ROCO-SteadyHand-live
```

Optional but useful for every handoff:

```bash
git rev-parse --short HEAD
```

Record that SHA with any run you send back for debugging.

---

## 1. Run software checks before motion

First run the tests:

```bash
python3 -m unittest discover -s tests -v
```

### Look for

- final result is `OK`;
- expected skips are acceptable;
- there are no failures/errors.

### If tests fail

Do not start physical manipulation.

Send back:

- the failing test names;
- complete traceback;
- current commit SHA.

Do not alter calibration to work around a test failure.

Now run the no-motion Vega preflight:

```bash
python3 tools/vega_preflight.py
```

This does **not** connect to the robot.

### Required preflight items

The important values should show usable/true values for:

- `dexcontrol_module`
- `pinocchio_module`
- `numpy_module`
- `python_can_module`
- `urdf_exists`
- `gripper_driver_exists`
- `working_arm: right`
- `ee_frame: tip_r`
- `base_frame: vega_1u_base_link`
- positive joint tolerance / timeout / grip current
- numeric `min_tcp_z_m`

### If preflight returns nonzero

Do not start motion.

Common cases:

| Output | Action |
| --- | --- |
| missing `dexcontrol`, Pinocchio, NumPy or python-can | software/environment problem; send the full preflight output |
| `urdf_exists: False` | deployment/path problem; do not bypass it |
| `gripper_driver_exists: False` | deployment/vendor-driver problem; do not attempt a grasp |
| wrong `working_arm` / `ee_frame` | stop; the competition path must remain RIGHT / `tip_r` |
| SDK version differs | send the full preflight output before changing config |
| wrist camera module is missing | arm position testing may still be diagnosable, but do not start wrist-profile/grasp work |

---

## 2. Run the offline geometry audit

Before moving to newly measured task coordinates:

```bash
python3 tools/vega_task_geometry_audit.py   --clearance-mm 100   --check-ik
```

### Look for

- all reviewed pick/place coordinates are inside the 386 mm board extent;
- no target violates the hard floor;
- the selected high-clearance targets solve IK;
- the report is written under `calibration/`.

### If geometry/IK fails

Do not compensate with a large wrist correction.

Send:

- terminal output;
- `calibration/task_geometry_audit.json`;
- current SHA.

If a task point is geometrically outside the board, fix the task coordinate.
If a reasonable task point is inside the board but fails IK, inspect that target
before changing calibration or joint limits.

---

## 3. Check competition readiness without motion

Run:

```bash
python3 tools/vega_competition_pipeline.py   --check-only   --competition-plan priority_pick_place
```

On a fresh onsite state with no physically taught wrist profiles, it is
**expected** to report that no verified competition actions are ready.

That is not a reason to edit the calibration. It means menu 3/4 still need to
teach and validate parts.

---

## 4. Start the main operator menu

For physical work:

```bash
python3 tools/vega_competition_pipeline.py   --confirm-head-motion   --confirm-physical-motion
```

Current menu:

```text
1. Recalibrate moved board
2. Test calibrated board/task positions
3. Wrist camera calibration
4. Task tests
5. Competition run sequence
6. Preserved legacy task versions
7. Reload operator settings / show readiness
0. Exit
```

Start with **7**.

### Look for

- competition plan loads;
- task coordinates validate;
- speed/clearance are reasonable;
- wrist profile count is shown.

The operator-editable scoring settings are in:

```text
configs/competition_plan.json
```

Use menu **7** after editing that file; it reloads/validates settings without
restarting the menu.

### If menu 7 says `Operator settings rejected`

Fix the configuration error it prints. Do not continue with a partially parsed
competition plan.

---

## 5. Validate board registration and coarse task XY at 100 mm

Use menu **2** before teaching a grasp.

Every physical location session should:

1. move the right arm camera-clear;
2. capture a fresh downward board image;
3. rebuild live XY registration;
4. return to RIGHT_READY;
5. preflight the requested target;
6. move only after those checks pass.

Start with:

- `board.center`;
- `board.top_right`;
- `board.bottom_right`;
- `board.bottom_left`;
- `task.battery_size1.pick`;
- `task.battery_size1.place`.

Then check other task points before teaching those parts.

### What to look for

At board reference points:

- TCP/claw should be over the intended board location;
- height should remain a safe high hover.

At task points:

- the wrist/claw should be coarsely over the correct physical part/socket;
- small residual error is acceptable for wrist centering;
- a large systematic miss is **not** a wrist-servo problem.

Record misses as physical offsets where possible:

```text
point:
left/right error: ___ mm
far/toward-robot error: ___ mm
```

### If the board image is rejected

Typical message:

```text
board image retake is not compatible with the calibrated board geometry;
run full five-point calibration
```

If the board truly moved/rotated outside the accepted retake geometry, use menu
**1** for a full calibration.

Do not repeatedly retry a visibly wrong board registration.

### If IK fails

The position tool may offer one recovery cycle through camera-clear and a fresh
image.

If the physical scene is understood and unobstructed, one supervised retry is
reasonable. If the same target fails again, answer **no**, save the output, and
inspect the target. Do not weaken limits.

### If one task coordinate is badly wrong but board references are right

The board calibration is probably not the first thing to change.

Send:

- task name;
- expected physical point;
- observed offset in mm;
- terminal target + measured `TIP_R`;
- fresh board scene/run artifact.

---

## 6. Teach the first part wrist profile

Start with the first part in the current competition plan, normally
`battery_size1`.

Direct command:

```bash
python3 tools/vega_competition_pipeline.py   --wrist-calibrate battery_size1   --confirm-head-motion   --confirm-physical-motion
```

Equivalent: main menu **3**.

The tool takes a fresh board image, returns to RIGHT_READY, moves to the coarse
pick hover, captures `wrist_a`, and prompts for a visible part feature and jaw
goal.

### Important interactive commands

```text
forward/back/left/right N   local XY adjustment in mm
yaw N                       yaw relative to READY, degrees
center                      run wrist visual centering
feature                     reselect the tracked feature
image                       capture/inspect another wrist image
depth N                     N mm down from the 100 mm hover
grab                        open, descend, grip, lift
return                      return a held test part to the source
place-config X Y DEPTH YAW  destination XY correction mm, depth mm, yaw deg
undo
save
abort
```

### Feature selection

Choose a visible textured/edge feature on the **part**, not the gripper.

If you get:

```text
feature is too close to the wrist image edge for a saved template
```

choose a different feature farther from the image edge.

If matching is weak/ambiguous, do not force it. Reselect a more distinctive
feature or improve the coarse position.

### Coarse-position failure

If the tool says:

```text
Adjustment exceeds 60 mm local radius; fix coarse coordinates
```

stop adjusting the wrist profile. The coarse task coordinate/registration is
wrong enough that it should be fixed upstream.

### Teaching depth

`depth N` means **N millimetres below the 100 mm hover**.

The tool prints the resulting clearance above the calibrated surface. Do not
guess a large descent. Tune under direct physical supervision and use `grab`
only when the XY/yaw/depth look correct.

### If it reports a floor violation

Examples:

```text
Target intersects the configured TCP floor + 5 mm
Grasp intersects TCP floor + 5 mm
Place intersects TCP floor + 5 mm
```

Do not lower the safety floor to make the command pass.

Recheck the taught depth, destination correction and physical geometry.

### Saving

Before `save`, do not leave the part held. Use `return` if appropriate.

A successful save writes:

- `calibration/wrist_part_profiles.json`;
- a hashed wrist template under `calibration/wrist_templates/`;
- a run folder / handoff JSON printed by the tool.

Preserve those files.

---

## 7. Run a pick-only test before pick/place

After saving the profile:

```bash
python3 tools/vega_competition_pipeline.py   --task-test battery_size1.pick   --confirm-head-motion   --confirm-physical-motion
```

The individual test asks for explicit confirmation before the grasp.

### Success criteria

- saved template reacquires the intended part feature;
- wrist centering converges;
- descent follows the taught profile;
- gripper reports a verified grip;
- part is visibly retained after lift.

A successful pick marks `grasp_verified` for that profile.

### Expected blockers

| Message | Action |
| --- | --- |
| `No taught wrist profile. Use menu 3 first.` | teach/save the profile |
| `Board calibration changed since wrist teaching` | re-teach this part against the current calibration |
| `Wrist template hash changed` | do not edit/replace the template manually; re-teach |
| `Wrist resolution changed` | re-teach using the current wrist camera mode |
| `Grasp depth has not been taught` | menu 3 → teach `depth N` |
| `Grip was not verified` | stop and physically inspect; do not assume the claw is empty |
| part may still be held | do not start another automatic action; inspect/recover first |

If the part is held after the pick test, follow the tool's `return` path before
continuing unless you intentionally stop the session to inspect it.

---

## 8. Teach/place-config and validate pick/place

In menu **3**, teach the destination settings if they are not already present:

```text
place-config X Y DEPTH YAW
```

- X/Y are board-frame destination corrections in mm;
- DEPTH is mm down from the 100 mm destination hover;
- YAW is degrees relative to READY.

Then run:

```bash
python3 tools/vega_competition_pipeline.py   --task-test battery_size1.pick_place   --confirm-head-motion   --confirm-physical-motion
```

### Look for

- verified pick/lift first;
- transfer to the intended destination;
- safe controlled release;
- retract;
- prompt asking whether placement was correct.

Only answer **yes** if the physical placement is actually acceptable. That is
what marks the profile `place_verified` for competition use.

### If placement is systematically offset

Adjust the **part's place configuration**, not the board calibration, unless
board reference tests are also wrong.

---

## 9. Check readiness, then run verified parts only

After at least one successful pick/place:

```bash
python3 tools/vega_competition_pipeline.py   --check-only   --competition-plan priority_pick_place
```

Now the verified part should appear eligible.

To run the verified priority plan:

```bash
python3 tools/vega_competition_pipeline.py   --competition-plan priority_pick_place   --confirm-head-motion   --confirm-physical-motion
```

Or use menu **5**.

The plan skips profiles that are not physically verified.

### Failure/retry behavior

A failed action is not blindly retried.

After inspection, the runner can allow the bounded operator-confirmed retry from
`configs/competition_plan.json`, or you can skip/stop.

If `run_summary.json` reports:

```text
holding_may_be_true: true
```

the runner blocks automatic retry. Physically inspect/recover before any next
part.

---

## 10. What to send back after a failure

For useful remote adjustment, send the **whole run folder**, not only a short
description.

Most useful files are:

- `run_summary.json`;
- `events.jsonl`;
- board scene JSON/image;
- raw wrist images;
- tracked/overlay wrist images;
- handoff JSON;
- current `calibration/wrist_part_profiles.json`;
- exact terminal error;
- commit SHA.

Also state:

```text
part/action:
board moved after last image? yes/no
expected physical result:
actual physical result:
XY miss if visible: ___ mm / direction
grip retained after lift? yes/no/unknown
part may currently be held? yes/no/unknown
```

If the failure is a coordinate miss, include the printed requested target and
measured `TIP_R`.

---

## Error quick reference

| Error / symptom | First response |
| --- | --- |
| tests fail | stop physical work; send traceback |
| preflight nonzero | fix deployment/environment first |
| no usable five-point calibration | inspect why; use menu 1 only if full recalibration is actually required |
| board retake geometry incompatible | full five-point calibration if board pose truly changed |
| repeated IK failure | stop; inspect target/scene, do not weaken safety limits |
| task hover is far from the part but board references are correct | fix task coordinate, not calibration |
| wrist correction wants >60 mm | fix coarse task coordinate |
| no wrist profile | menu 3 |
| calibration/template/resolution stale | re-teach that part |
| ambiguous/weak wrist match | stop/reselect feature; do not guess |
| missing grasp depth | menu 3 → `depth N` |
| missing place settings | menu 3 → `place-config ...` |
| floor + 5 mm violation | do not lower the floor; revise requested depth/geometry |
| grip not verified | inspect immediately; assume held state may be uncertain |
| `holding_may_be_true` | no automated retry/next part |
| no verified actions in priority plan | expected until individual pick/place tests are physically validated |

---

## Recommended progression for the day

Do not try to teach all nine parts before proving the workflow.

For each part:

```text
high-hover task XY correct
-> teach wrist profile
-> center reliably
-> pick test verified
-> teach/adjust place
-> pick/place verified
-> now eligible for competition plan
```

Then move to the next part.

This makes every successful physical test directly usable by the final
competition sequence.
