# First onsite session

Onsite participants: Matvey Okoneshnikov and Eunice Ding.
Original submission: Yifei Jiang.

## Before robot access

1. Get late check-in and the missed briefing material. Confirm the actual access slots.
2. Obtain the Sharpa simulation package, hardware SDK, and hand-control example.
3. Confirm Industrial Board scoring, task order, retries, and permitted pre-trial calibration.
4. Confirm whether our executor may run on the robot's onboard computer.

## At each robot

1. Ask the engineer to demonstrate the current working control path and stop behavior.
2. Record SDK versions, working arm/hand, camera interfaces, and available helper scripts.
3. Check current Vega gripper condition; the manual's left-cable fault is historical.
4. Create a session folder before changing configuration or attempting a task.
5. Capture observations; measure frames, tool offsets, and gripper behavior.
6. Attempt one reachable pick-and-place with the engineer, then repeat and record outcomes.

The toolkit currently has no hardware-run command. Its adapters must be implemented
and validated first. A mock run is useful for checking workflow and logs only.

## Record each attempt

In the generated `trials.csv`, use `true`, `false`, or blank (unknown) for success
fields. Record measured duration and the first failed stage. Link the corresponding
video. A completed motion command alone is not proof of a successful placement.

Keep dry-run folders separate from physical session notes. The configuration
snapshot preserves inputs at session creation; create a new session after changing
calibration or record the exact change in the session notes.

## Before leaving a session

- Note the next concrete experiment and the current blocker.
- Save the working configuration and software revision.
- Back up `runs/`, local files in `data/`, and `llm/`: Git excludes these.
