# TODAY — Split Ownership (Sep 27)

Two people, two robots. **Do not both debug the same platform today.**

## Matvey → DexMate Vega U

Primary checklist:

**[MATVEY_DEXMATE_TODAY.md](./MATVEY_DEXMATE_TODAY.md)**

Mission:

```text
make the full Vega stack physically real
→ sensors
→ frames
→ FK/IK
→ joint control
→ CAN gripper
→ target generation
→ one repeatable part
→ then expand score
```

End-of-day minimum: one open-release part runs end-to-end repeatedly, with the working calibration/config/targets saved.

## Eunice → Sharpa North

Primary checklist:

**[EUNICE_SHARPA_TODAY.md](./EUNICE_SHARPA_TODAY.md)**

Mission:

```text
turn North from an unknown platform into a known control contract
→ get SDK + full robot model
→ map state/action
→ prove stop
→ prove one arm command
→ prove one hand command
→ implement real Sharpa adapter
```

End-of-day minimum: our code can connect/read state/stop/send one safe arm command/send one safe hand command, and every important interface fact is documented.

## Coordination points only

Meet briefly after each bootcamp / major breakthrough and exchange only:

1. working commit;
2. what is now proven;
3. current blocker;
4. next experiment;
5. any shared-code change needed.

Avoid editing the same shared files simultaneously. Robot-specific work should stay primarily in:

```text
Matvey:
  steadyhand/adapters/vega.py
  steadyhand/cameras/vega.py
  steadyhand/grippers/vega.py
  steadyhand/kinematics/*
  configs/robots/vega.json
  configs/skills/vega.json
  tools/vega_*

Eunice:
  steadyhand/adapters/sharpa.py
  configs/robots/sharpa.json
  tools/sharpa_*
  Sharpa-specific docs/config added today
```

If either person needs to change shared modules such as:

```text
steadyhand/adapters/base.py
steadyhand/config.py
steadyhand/models.py
steadyhand/executor.py
```

tell the other person first to avoid conflicting edits.

## Tonight's shared finish line

- [ ] Vega: full lower stack proven + one repeatable physical part.
- [ ] Sharpa: full control contract known + minimal live adapter proven.
- [ ] Both: vendor files, robot descriptions, configs, logs, and working commits backed up.
- [ ] Both: one clearly written blocker + first experiment for tomorrow.
