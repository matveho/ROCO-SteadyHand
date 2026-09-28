# Competition Questions To Resolve

Use this as the shared checklist when organizers / competition representatives
become available. Record answers verbatim where possible, including who answered
and when. Do not block current engineering work while waiting.

## Trial procedure and scoring

- What exactly starts and ends an official trial?
- Is there one fixed part order, or may teams choose part order?
- Is partial credit awarded independently per assembled part?
- Are any parts weighted differently?
- What constitutes successful placement for each open-release part?
- What constitutes successful insertion for rod, bolt, USB, HDMI, and pin?
- Is completion judged automatically, visually, manually, or by fixture state?
- Is speed only a tie-breaker, or does elapsed time affect the numerical score?
- What happens when a part is dropped outside its source/destination area?
- May a run intentionally skip a difficult part and continue?
- What are the retry / restart rules within the team's attempt window?
- After terminating a failed run, what state is reset before another attempt?

## Human intervention

- Confirm that no human intervention is allowed during an active trial except
  terminating it.
- What exactly counts as intervention: touching the board, robot, computer,
  keyboard, e-stop, or software UI?
- Is choosing a strategy / part order before pressing start permitted?
- Are operators allowed to stop a failed part and continue the same trial, or
  must the whole run terminate?
- May a physical e-stop be used without forfeiting later attempts?

## Calibration and data collection

- What calibration may be performed immediately before an official run?
- May the robot take one or more head/wrist images before the timed trial?
- May board pose be estimated after the official start but before the first
  manipulation?
- May we store calibration constants between attempts?
- May we collect and retain RGB images / robot telemetry from practice and
  official attempts?
- May code be modified between attempts?
- May manually taught heights / wrist goal pixels be retained between runs?

## Board / scene randomization

- Does the entire task board translate and/or rotate between attempts?
- Approximately what translation and rotation range should we expect?
- Are the board corners always fully visible to the head camera?
- Do individual source pieces move independently of the board?
- For each part, what is the allowed source-position and orientation variation?
- Are destination fixtures fixed relative to the board?
- Can parts start touching / overlapping anything else?
- Is lighting intentionally varied?
- Is the table / board height fixed across practice and competition setups?

## Initial robot state

- Is the Vega initial arm pose fixed across attempts?
- Is head pose reset before each attempt?
- Are grippers homed before the attempt, or is that the team's responsibility?
- Are there restrictions on moving the head during the trial?
- Is the task board always placed in the same approximate Vega workspace?
- Is any required robot homing included in timed trial duration?

## Hardware and execution constraints

- Must the submitted code run onboard the Vega, or may a connected laptop drive
  computation/control?
- Is local network communication to the robot allowed throughout the run?
- Is internet access prohibited, permitted, or simply not guaranteed during
  official attempts?
- Are there restrictions on CPU/GPU use or launching additional camera
  processes?
- Are teams permitted to restart `dexsensor` / camera services between
  attempts if needed?
- Are there any prohibited robot APIs or controller modes?

## Task-specific manipulation

- Are top-down grasps valid for every part, or are any parts expected to require
  a different approach orientation?
- For batteries/gears, how much placement tolerance is accepted?
- For connector/insertion parts, is there mechanical compliance in the fixture?
- Are the simulator snap/search semantics intended to approximate any physical
  fixture behavior?
- Is force/torque sensing expected or recommended for insertion?
- Are there known part-specific issues on the competition hardware that teams
  should account for?

## Strategy clarification

- Is using known board-relative source regions / part identities explicitly
  allowed?
- Are teams expected to generalize to different part identities/layouts, or only
  pose/orientation variation of the known board?
- Can the destination layout be treated as fixed relative to the board?
- Is template-based / deterministic perception fully acceptable?

## Record answers here

| Date/time | Representative | Question | Answer | Confidence / notes |
|---|---|---|---|---|
| | | | | |
