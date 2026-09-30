#requires -Version 5.1
<#
.SYNOPSIS
Deploy committed main to the Vega over Ethernet while preserving rollback copies.

The script never deletes the previous live checkout before the incoming checkout
has been staged, hash-checked, and passed the no-motion preflight.  Use -ListVersions
or -RestoreVersion to inspect/recover a named archive.
#>
[CmdletBinding()]
param(
    [string]$HostName = "192.168.50.20",
    [string]$UserName = "dexmate",
    [string]$LiveDir = "/home/dexmate/ROCO-SteadyHand-live",
    [switch]$SkipPull,
    [switch]$SkipPreflight,
    [switch]$ListVersions,
    [string]$RestoreVersion,
    [string]$VersionName,
    [ValidateSet("replace", "new")][string]$DeployMode = "replace"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Remote = "$UserName@$HostName"
$CommonSsh = @("-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=8")

function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)][string]$Exe,
        [Parameter(Mandatory=$true)][string[]]$Arguments
    )
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}

# The robot account uses this fixed onsite password.  An environment override is
# still accepted for teams that rotate credentials, but normal deployment must
# never stop for an interactive password prompt.
$Password = if ($env:DEXMATE_PASSWORD) { $env:DEXMATE_PASSWORD } else { "hello-dex" }
$AskPassPath = $null
$OldAskPass = $env:SSH_ASKPASS
$OldAskPassRequire = $env:SSH_ASKPASS_REQUIRE
$OldDisplay = $env:DISPLAY
$OldRocoPassword = $env:ROCO_SSH_PASSWORD
$BundlePath = $null
$RemoteScriptPath = $null

try {
    foreach ($command in @("git", "ssh", "scp")) {
        if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
            throw "$command is not available in PATH"
        }
    }
    if ($LiveDir -notmatch '^/[A-Za-z0-9._/-]+$') {
        throw "LiveDir contains unsupported characters"
    }

    if ($Password) {
        $env:ROCO_SSH_PASSWORD = $Password
        $AskPassPath = Join-Path $env:TEMP "roco-ssh-askpass-$PID.cmd"
        @'
@echo off
powershell.exe -NoProfile -Command "[Console]::Out.Write($env:ROCO_SSH_PASSWORD)"
'@ | Set-Content -Path $AskPassPath -Encoding ASCII
        $env:SSH_ASKPASS = $AskPassPath
        $env:SSH_ASKPASS_REQUIRE = "force"
        $env:DISPLAY = "roco-deploy"
    }

    if ($ListVersions) {
        Write-Host "Archived Vega versions:"
        Invoke-Checked ssh ($CommonSsh + @(
            $Remote,
            "find /home/dexmate/ROCO-SteadyHand-versions -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -r"
        ))
        return
    }

    if ($RestoreVersion) {
        if ($RestoreVersion -notmatch '^[A-Za-z0-9._-]+$') {
            throw "RestoreVersion contains unsupported characters"
        }
        $restore = @"
set -euo pipefail
LIVE='$LiveDir'
ROOT='/home/dexmate/ROCO-SteadyHand-versions'
V='$RestoreVersion'
SRC="$ROOT/$V"
[ -d "$SRC" ] || { echo "Unknown archived version: $V" >&2; exit 5; }
pgrep -af 'tools/vega_competition_pipeline.py' | grep -v "[g]rep" >/dev/null && { echo 'REFUSING RESTORE: competition pipeline is running' >&2; exit 8; } || true
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BACKUP="$ROOT/${STAMP}_before_restore"
[ -d "$LIVE" ] && cp -a "$LIVE" "$BACKUP"
STAGE="${LIVE}.restore.$$"
rm -rf "$STAGE"
cp -a "$SRC" "$STAGE"
[ -d "$STAGE/.git" ] || { echo 'Archive is not a Git checkout' >&2; rm -rf "$STAGE"; exit 6; }
if [ -d "$LIVE" ]; then mv "$LIVE" "${LIVE}.swap.$$"; fi
if ! mv "$STAGE" "$LIVE"; then
  [ -d "${LIVE}.swap.$$" ] && mv "${LIVE}.swap.$$" "$LIVE"
  exit 7
fi
rm -rf "${LIVE}.swap.$$"
echo "RESTORED_VERSION=$V"
[ -n "${BACKUP:-}" ] && echo "CURRENT_LIVE_BACKUP=$BACKUP"
(cd "$LIVE" && git rev-parse HEAD)
"@
        Invoke-Checked ssh ($CommonSsh + @($Remote, $restore))
        return
    }

    if (-not $VersionName) {
        $choice = Read-Host "Deploy: archive current and replace live [O], or create a named new version [N] (O/N)"
        if ($choice -match '^[Nn]$') { $DeployMode = "new" }
        elseif ($choice -notmatch '^[Oo]$') { throw "Choose O or N" }
        $VersionName = Read-Host "Name for the archived current version (Enter = automatic label)"
    }
    if (-not $VersionName) { $VersionName = "before-$DeployMode" }
    if ($VersionName -notmatch '^[A-Za-z0-9._-]+$') {
        throw "VersionName may contain only letters, numbers, dot, underscore, and hyphen"
    }

    Push-Location $RepoRoot
    try {
        if (-not (Test-Path ".git")) { throw "Repository root not found at $RepoRoot" }
        if (-not $SkipPull) {
            Write-Host "Updating local main..."
            Invoke-Checked git @("pull", "--ff-only", "origin", "main")
        }
        $Head = (& git rev-parse main).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $Head) { throw "Could not resolve local main" }
        $BundleName = "roco-main-$PID.bundle"
        $BundlePath = Join-Path $env:TEMP $BundleName
        Write-Host "Bundling main $Head..."
        Invoke-Checked git @("bundle", "create", $BundlePath, "main")
        Invoke-Checked git @("bundle", "verify", $BundlePath)

        $RemoteScriptName = "roco-deploy-$PID.sh"
        $RemoteScriptPath = Join-Path $env:TEMP $RemoteScriptName
        $SkipPreflightValue = if ($SkipPreflight) { "1" } else { "0" }
        $RemoteTemplate = @'
#!/usr/bin/env bash
set -euo pipefail
LIVE='__LIVE__'
BUNDLE="/home/dexmate/__BUNDLE__"
EXPECTED='__EXPECTED__'
SKIP_PREFLIGHT='__SKIP_PREFLIGHT__'
MODE='__MODE__'
VERSION_LABEL='__VERSION_LABEL__'
VERSION_ROOT='/home/dexmate/ROCO-SteadyHand-versions'

mkdir -p "$VERSION_ROOT"
pgrep -af 'tools/vega_competition_pipeline.py' | grep -v '[g]rep' >/dev/null && {
  echo 'REFUSING DEPLOY: competition pipeline is running; stop it first.' >&2
  exit 8
} || true
if [ -d "$LIVE" ]; then
  DIRTY="$(git -C "$LIVE" status --porcelain --untracked-files=no 2>/dev/null || true)"
  NON_CALIBRATION="$(printf '%s\n' "$DIRTY" | awk 'NF && $2 !~ /^calibration\// {print}')"
  if [ -n "$NON_CALIBRATION" ]; then
    echo "REFUSING DEPLOY: tracked non-calibration files in $LIVE have onsite edits:" >&2
    echo "$NON_CALIBRATION" >&2
    exit 3
  fi
  if [ -n "$DIRTY" ]; then
    echo "PRESERVING onsite calibration edits while updating code:"
    echo "$DIRTY"
  fi
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE_NAME="${STAMP}_${VERSION_LABEL}_${EXPECTED:0:12}"
ARCHIVE="$VERSION_ROOT/$ARCHIVE_NAME"
while [ -e "$ARCHIVE" ]; do ARCHIVE="${VERSION_ROOT}/${ARCHIVE_NAME}_$RANDOM"; done
if [ -d "$LIVE" ]; then
  cp -a "$LIVE" "$ARCHIVE"
  echo "PRESERVED_LIVE_VERSION=$ARCHIVE"
fi
CALIBRATION_PRESERVE="/home/dexmate/roco_calibration_edits_${STAMP}_$$"
if [ -d "$LIVE" ]; then
  while IFS= read -r PATHNAME; do
    [ -n "$PATHNAME" ] || continue
    mkdir -p "$CALIBRATION_PRESERVE/$(dirname "$PATHNAME")"
    cp -a "$LIVE/$PATHNAME" "$CALIBRATION_PRESERVE/$PATHNAME"
  done < <(git -C "$LIVE" diff --name-only HEAD -- calibration)
fi
STAGE="${LIVE}.deploy.${EXPECTED:0:12}.$$"
rm -rf "$STAGE"
if [ -d "$LIVE" ]; then cp -a "$LIVE" "$STAGE"; else mkdir -p "$STAGE"; git -C "$STAGE" init; fi
git -C "$STAGE" fetch --force "$BUNDLE" main:refs/remotes/deploy/main
MIGRATION_BACKUP="/home/dexmate/roco_untracked_before_deploy_${STAMP}"
MIGRATION_COUNT=0
while IFS= read -r PATHNAME; do
  [ -n "$PATHNAME" ] || continue
  if [ -e "$STAGE/$PATHNAME" ] && ! git -C "$STAGE" ls-files --error-unmatch -- "$PATHNAME" >/dev/null 2>&1; then
    mkdir -p "$MIGRATION_BACKUP/$(dirname "$PATHNAME")"
    cp -a "$STAGE/$PATHNAME" "$MIGRATION_BACKUP/$PATHNAME"
    rm -rf -- "$STAGE/$PATHNAME"
    MIGRATION_COUNT=$((MIGRATION_COUNT + 1))
  fi
done < <(git -C "$STAGE" ls-tree -r --name-only refs/remotes/deploy/main)
if [ "$MIGRATION_COUNT" -gt 0 ]; then echo "PRESERVED $MIGRATION_COUNT old untracked paths in $MIGRATION_BACKUP"; fi
git -C "$STAGE" checkout -f -B main refs/remotes/deploy/main
if [ -d "$CALIBRATION_PRESERVE" ]; then
  while IFS= read -r PATHNAME; do
    [ -n "$PATHNAME" ] || continue
    mkdir -p "$STAGE/$(dirname "$PATHNAME")"
    cp -a "$CALIBRATION_PRESERVE/$PATHNAME" "$STAGE/$PATHNAME"
  done < <(find "$CALIBRATION_PRESERVE" -type f -path "$CALIBRATION_PRESERVE/calibration/*" -printf '%P\n' 2>/dev/null)
  echo 'RESTORED onsite calibration edits into the deployed checkout.'
fi
ACTUAL="$(git -C "$STAGE" rev-parse HEAD)"
[ "$ACTUAL" = "$EXPECTED" ] || { echo "REFUSING: staged SHA $ACTUAL != expected $EXPECTED" >&2; rm -rf "$STAGE"; exit 4; }
if [ "$SKIP_PREFLIGHT" != "1" ]; then
  echo '=== staged no-motion preflight ==='
  # Non-interactive SSH does not load the operator's conda shell setup.  Use
  # the installed runtime explicitly so the preflight checks the same SDK that
  # the competition commands use, and provide the fixed robot identity.
  ROBOT_NAME="${ROBOT_NAME:-dm/vgfcb66075ea-1u}"
  export ROBOT_NAME
  PYTHON_BIN='/home/dexmate/miniconda3/bin/python3'
  [ -x "$PYTHON_BIN" ] || PYTHON_BIN="$(command -v python3)"
  (cd "$STAGE" && PATH="$(dirname "$PYTHON_BIN"):$PATH" ROBOT_NAME="$ROBOT_NAME" "$PYTHON_BIN" tools/vega_preflight.py) || { echo 'PREFLIGHT FAILED; live checkout unchanged.' >&2; rm -rf "$STAGE"; exit 10; }
fi
SWAP="${LIVE}.swap.$$"
rm -rf "$SWAP" "$CALIBRATION_PRESERVE"
if [ -d "$LIVE" ]; then mv "$LIVE" "$SWAP"; fi
if ! mv "$STAGE" "$LIVE"; then
  [ -d "$SWAP" ] && mv "$SWAP" "$LIVE"
  exit 7
fi
rm -rf "$SWAP"
rm -f "$BUNDLE" "/home/dexmate/__REMOTE_SCRIPT__"
echo "DEPLOY_MODE=$MODE"
echo "DEPLOYED_SHA=$ACTUAL"
echo "DEPLOY_DIR=$LIVE"
'@
        $RemoteBody = $RemoteTemplate
        $RemoteBody = $RemoteBody.Replace("__LIVE__", $LiveDir)
        $RemoteBody = $RemoteBody.Replace("__BUNDLE__", $BundleName)
        $RemoteBody = $RemoteBody.Replace("__EXPECTED__", $Head)
        $RemoteBody = $RemoteBody.Replace("__SKIP_PREFLIGHT__", $SkipPreflightValue)
        $RemoteBody = $RemoteBody.Replace("__REMOTE_SCRIPT__", $RemoteScriptName)
        $RemoteBody = $RemoteBody.Replace("__MODE__", $DeployMode)
        $RemoteBody = $RemoteBody.Replace("__VERSION_LABEL__", $VersionName)
        $RemoteBody = $RemoteBody -replace "`r`n", "`n"
        [System.IO.File]::WriteAllText($RemoteScriptPath, $RemoteBody, (New-Object System.Text.UTF8Encoding($false)))
        Write-Host "Copying bundle to $Remote..."
        Invoke-Checked scp ($CommonSsh + @($BundlePath, $RemoteScriptPath, "${Remote}:/home/dexmate/"))
        Write-Host "Staging and deploying without replacing the rollback archive..."
        Invoke-Checked ssh ($CommonSsh + @($Remote, "bash /home/dexmate/$RemoteScriptName"))
        Write-Host "Vega deployment complete: $Head"
    }
    finally { Pop-Location }
}
finally {
    if ($BundlePath -and (Test-Path $BundlePath)) { Remove-Item $BundlePath -Force -ErrorAction SilentlyContinue }
    if ($RemoteScriptPath -and (Test-Path $RemoteScriptPath)) { Remove-Item $RemoteScriptPath -Force -ErrorAction SilentlyContinue }
    if ($AskPassPath -and (Test-Path $AskPassPath)) { Remove-Item $AskPassPath -Force -ErrorAction SilentlyContinue }
    $env:SSH_ASKPASS = $OldAskPass
    $env:SSH_ASKPASS_REQUIRE = $OldAskPassRequire
    $env:DISPLAY = $OldDisplay
    $env:ROCO_SSH_PASSWORD = $OldRocoPassword
}
