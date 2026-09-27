# SteadyHand

Our entry for the Industrial Board Assembly track at RoCo @ IROS 2026 in Pittsburgh.

policy.py is the submitted Vega simulation policy. Real-robot integration lives
in the steadyhand package and is deliberately separated from the submission.

## Branches

- submission-baseline: exact original submission snapshot at
  705e4e03ed8e3427cfc5f58a20de02afc3687e6b. Never develop on this branch.
- onsite-2026: active competition integration work.
- main: kept fast-forwarded to the current integration head for easy cloning.

The original policy.py remains unchanged from the submission snapshot.

## Architecture

Shared code owns task definitions, calibration, logging and the manipulation
vocabulary. Robot-specific code is isolated behind one adapter contract.

~~~text
task / perception / calibration
            |
       shared skills
            |
      RobotAdapter
       /        \
   Vega          Sharpa
 dexcontrol     North/Wave
~~~

See docs/ARCHITECTURE.md.

## Offline preparation

~~~bash
python onsite.py doctor
python onsite.py check-config --robot vega
python onsite.py dry-run --robot vega --part battery_size1
python -m unittest discover -s tests -v
~~~

The hardware adapters are intentionally disabled until each competition robot's
installed control path and stop behavior have been verified. A dry run never
contacts hardware and never claims physical success.

See docs/ONSITE.md and docs/ARRIVAL.md.
