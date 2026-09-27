# SteadyHand onsite workspace

The toolkit lives at the repository root alongside the original `policy.py` submission.
Code, configuration templates, tests, and docs belong in Git. Generated runs, local data,
and private planning in `llm/` are gitignored; back them up before switching machines.

Run these commands from the repository root. Python 3.10+ is enough; no packages are required yet.

```powershell
python onsite.py doctor
python onsite.py check-config --robot vega
python onsite.py dry-run --robot vega --part battery_size1
python onsite.py dry-run --robot sharpa --part pin --fail-at verify_grasp
python onsite.py new-session --robot vega --operator "Matvey Okoneshnikov"
```

`check-config` accepts unfinished templates and lists what's missing. Add `--require-ready`
to return an error for missing setup. Passing it checks configuration completeness only.
`dry-run` uses a mock, never connects to a robot, and does not simulate physics.
The injected-failure example intentionally exits with an error and still saves its log.

| Location | What goes here |
|---|---|
| `onsite.py` | Command-line entry point |
| `steadyhand/` | Shared configuration, session logging, and dry-run code |
| `steadyhand/adapters/` | Separate Vega and Sharpa integration files |
| `configs/robots/` | Connection settings, joint order, and motion limits |
| `configs/task_board.json` | Part order and pick/place pose placeholders |
| `calibration/` | Measured transforms for each robot |
| `data/` | Recordings and selected reference material |
| `runs/` | Generated sessions, configuration snapshots, and trial logs |
| `docs/ARRIVAL.md` | First robot session and trial-recording notes |
| `tests/` | Offline checks |
| `llm/ONSITE_PLAN.md` | Private preparation plan and source links (local only) |

For a new session, substitute `sharpa` or operator `Eunice Ding` as appropriate.
Each session includes `session.json`, configuration snapshots, `trials.csv`, and
`notes.md`. Dry runs also write `events.jsonl` and `result.json`, explicitly marked
as mock results. Record physical attempts in a separate `new-session` folder.

All task poses use metres and **wxyz** quaternions. Calibration matrices are
`T_destination_source`: for example, `T_base_camera` maps camera coordinates into
the robot base frame. Unknown values stay `null`; do not substitute guessed poses.

The hardware adapter files are intentionally disabled. Complete and validate them
against the installed SDKs onsite before adding a hardware-run command.

```powershell
python -m unittest discover -s tests -v
```
