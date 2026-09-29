# Vega handover cleanup

Run this from the Windows laptop checkout before giving the robot to another team.

1. Update the laptop checkout:

```powershell
git pull --ff-only origin main
```

2. Run the handover cleanup:

```powershell
.\tools\copy_from_vega.ps1 -HandoverCleanup
```

The script copies the robot's persistent calibration/config/annotation state into the laptop repo, commits and pushes any changes, verifies `origin/main`, then deletes:

- `/home/dexmate/ROCO-SteadyHand-live`
- `/home/dexmate/ROCO-SteadyHand`

If backup, commit, push, verification, or the robot-side safety check fails, repository deletion does not run.
