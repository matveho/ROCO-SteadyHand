#!/usr/bin/env bash
# In-place competition deployment. No version archives or run-directory copies.
set -euo pipefail
LIVE="$1"
BUNDLE="$2"  # '-' means the expected commit is already on the robot.
EXPECTED="$3"
SETTINGS="$4"
PREFLIGHT="${5:-0}"
[[ "$LIVE" = /* && "$LIVE" != / && "$EXPECTED" =~ ^[0-9a-f]{40}$ ]] || exit 4
mkdir -p "$(dirname "$LIVE")"
exec 9>"${LIVE}.deploy.lock"
flock -n 9 || { echo 'Another deployment is running.' >&2; exit 8; }
if pgrep -af '[p]ython[^ ]* .*tools/vega_(competition_pipeline|wrist_part_calibrate|head_target_preview|head_fallback|board_five_point_calibrate|wrist_servo|wrist_fine_center|battery_size1_(pick|center|coarse_hover))\.py'; then
  echo 'REFUSING DEPLOY: stop the robot-control program first.' >&2
  exit 8
fi
if [ ! -e "$LIVE/.git" ]; then
  mkdir -p "$LIVE"
  git -C "$LIVE" init -q
fi
if [ "$BUNDLE" != '-' ]; then
  git -C "$LIVE" fetch --quiet --force "$BUNDLE" main:refs/remotes/deploy/main
  ACTUAL="$(git -C "$LIVE" rev-parse refs/remotes/deploy/main)"
  [ "$ACTUAL" = "$EXPECTED" ] || { echo 'Incoming commit mismatch; live files unchanged.' >&2; exit 4; }
fi
git -C "$LIVE" cat-file -e "$EXPECTED^{commit}"
for name in competition_actions.json competition_plan.json competition_offsets.json; do
  [ -f "$SETTINGS/$name" ] || { echo "Missing laptop setting: $name" >&2; exit 4; }
done
# Only calibration is copied temporarily; never copy Git history or runs.
PRESERVE="$(mktemp -d "${LIVE}.calibration.XXXXXX")"
restore_calibration() {
  if [ -d "$PRESERVE/calibration" ]; then
    mkdir -p "$LIVE/calibration"
    if ! cp -a "$PRESERVE/calibration/." "$LIVE/calibration/"; then
      echo "CALIBRATION RESTORE FAILED: retained at $PRESERVE" >&2
      return 1
    fi
  fi
  rm -rf -- "$PRESERVE"
}
cleanup() {
  code=$?
  trap - EXIT
  restore_calibration || code=1
  exit "$code"
}
# Finish the copy before installing a trap that could restore a partial copy.
if [ -d "$LIVE/calibration" ]; then
  cp -a "$LIVE/calibration" "$PRESERVE/calibration"
fi
trap cleanup EXIT
git -C "$LIVE" checkout --quiet -f -B main "$EXPECTED"
restore_calibration
mkdir -p "$LIVE/configs"
cp "$SETTINGS/competition_actions.json" "$LIVE/configs/competition_actions.json"
cp "$SETTINGS/competition_plan.json" "$LIVE/configs/competition_plan.json"
cp "$SETTINGS/competition_offsets.json" "$LIVE/competition_offsets.json"
[ "$(git -C "$LIVE" rev-parse HEAD)" = "$EXPECTED" ] || exit 4
if [ "$PREFLIGHT" = 1 ]; then
  export ROBOT_NAME="${ROBOT_NAME:-dm/vgfcb66075ea-1u}"
  unset PYTHONPATH
  PYTHON_BIN='/home/dexmate/miniconda3/bin/python3'
  [ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(command -v python3)"
  (cd "$LIVE" && "$PYTHON_BIN" tools/vega_preflight.py) || {
    echo 'PREFLIGHT FAILED: code was updated in place; inspect before running.' >&2
    exit 10
  }
fi
echo "DEPLOYED_SHA=$EXPECTED"
echo "DEPLOY_DIR=$LIVE"
echo 'Robot calibration and run logs kept. Laptop competition settings installed.'
