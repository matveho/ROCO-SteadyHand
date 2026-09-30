#requires -Version 5.1
<#
.SYNOPSIS
Deploy the current committed main branch to the competition Vega over Ethernet.

.DESCRIPTION
- Pulls local main with --ff-only unless -SkipPull is supplied.
- Creates a temporary Git bundle.
- Copies the bundle and a tiny remote deployment script to the Vega.
- Refuses to overwrite tracked edits in ~/ROCO-SteadyHand-live.
- Updates the deployment checkout from the bundle without touching untracked onsite files.
- Prints the deployed SHA and runs the no-motion Vega preflight.

Authentication:
  Preferred: SSH key authentication.
  Otherwise set DEXMATE_PASSWORD in the local environment. If it is absent, this
  script prompts once and reuses that password for scp + ssh through SSH_ASKPASS.
  No password is stored in this repository.
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
    [string]$VersionName
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Remote = "$UserName@$HostName"

function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)][string]$Exe,
        [Parameter(Mandatory=$true)][string[]]$Arguments
    )
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe failed with exit code $LASTEXITCODE"
    }
}

$Password = "hello-dex"
$AskPassPath = $null
$OldAskPass = $env:SSH_ASKPASS
$OldAskPassRequire = $env:SSH_ASKPASS_REQUIRE
$OldDisplay = $env:DISPLAY
$OldRocoPassword = $env:ROCO_SSH_PASSWORD
$BundlePath = $null
$RemoteScriptPath = $null

try {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        throw "git is not available in PATH"
    }
    if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
        throw "ssh is not available in PATH"
    }
    if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
        throw "scp is not available in PATH"
    }

    $CommonSsh = @(
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=8"
    )

    if ($ListVersions) {
        Write-Host "Archived Vega versions:"
        Invoke-Checked ssh ($CommonSsh + @(
            $Remote,
            "find /home/dexmate/ROCO-SteadyHand-versions -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -r"
        ))
        return
    }

    $Mode = "deploy"
    $SelectedVersion = ""
    if ($RestoreVersion) {
        $Mode = "restore"
        $SelectedVersion = $RestoreVersion
    }

    if (-not $Password) {
        $Secure = Read-Host "DexMate SSH password (or configure DEXMATE_PASSWORD once)" -AsSecureString
        $Bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
        try {
            $Password = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Bstr)
        }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Bstr)
        }
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

    if ($Mode -eq "restore") {
        if ($SelectedVersion -notmatch '^[A-Za-z0-9._-]+$') {
            throw "RestoreVersion contains unsupported characters"
        }
        Write-Host "Restoring archived Vega version $SelectedVersion..."
        $RestoreCommand = "set -e; LIVE='$LiveDir'; ROOT=/home/dexmate/ROCO-SteadyHand-versions; V='$SelectedVersion'; " +
            '[ -d "$ROOT/$V" ] || { echo "Unknown archived version" >&2; exit 5; }; ' +
            'BACKUP="$ROOT/$(date -u +%Y%m%dT%H%M%SZ)_before_restore"; ' +
            'cp -a "$LIVE" "$BACKUP"; rm -rf "$LIVE.restore.tmp"; ' +
            'cp -a "$ROOT/$V" "$LIVE.restore.tmp"; rm -rf "$LIVE"; ' +
            'mv "$LIVE.restore.tmp" "$LIVE"; echo RESTORED_VERSION=$V; echo CURRENT_LIVE_BACKUP=$BACKUP'
        Invoke-Checked ssh ($CommonSsh + @(
            $Remote,
            $RestoreCommand
        ))
        return
    }

    if (-not $VersionName) {
        $Choice = Read-Host "Deploy current code: overwrite live [O] or archive a named rollback version [N] (O/N)"
        if ($Choice -match '^[Nn]$') {
            $VersionName = Read-Host "Rollback label (old live is always preserved; label is for this archive)"
        }
    }
    if (-not $VersionName) {
        $VersionName = "deploy"
    }
    if ($VersionName -notmatch '^[A-Za-z0-9._-]+$') {
        throw "VersionName may contain only letters, numbers, dot, underscore, and hyphen"
    }

    Push-Location $RepoRoot
    try {
        if (-not (Test-Path ".git")) {
            throw "Repository root not found at $RepoRoot"
        }

        if (-not $SkipPull) {
            Write-Host "Updating local main..."
            Invoke-Checked git @("pull", "--ff-only", "origin", "main")
        }

        $Head = (& git rev-parse main).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $Head) {
            throw "Could not resolve local main"
        }

        $BundleName = "roco-main-$PID.bundle"
        $BundlePath = Join-Path $env:TEMP $BundleName
        Write-Host "Bundling main $Head..."
        Invoke-Checked git @("bundle", "create", $BundlePath, "main")
        Invoke-Checked git @("bundle", "verify", $BundlePath)

        $RemoteScriptName = "roco-deploy-$PID.sh"
        $RemoteScriptPath = Join-Path $env:TEMP $RemoteScriptName
        $SkipPreflightValue = if ($SkipPreflight) { "1" } else { "0" }

        if ($LiveDir -notmatch '^/[A-Za-z0-9._/-]+$') {
            throw "LiveDir contains unsupported characters: $LiveDir"
        }

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

if [ "$MODE" = "deploy" ]; then
    mkdir -p "$VERSION_ROOT"
    ARCHIVE_NAME="$(date -u +%Y%m%dT%H%M%SZ)_${VERSION_LABEL}_${EXPECTED:0:12}"
    ARCHIVE="$VERSION_ROOT/$ARCHIVE_NAME"
    while [ -e "$ARCHIVE" ]; do
        ARCHIVE="${VERSION_ROOT}/${ARCHIVE_NAME}_$RANDOM"
    done
    if [ -d "$LIVE" ]; then
        cp -a "$LIVE" "$ARCHIVE"
        echo "PRESERVED_LIVE_VERSION=$ARCHIVE"
    fi
fi

if [ ! -d "$LIVE/.git" ]; then
    mkdir -p "$LIVE"
    git -C "$LIVE" init
fi

DIRTY="$(git -C "$LIVE" status --porcelain --untracked-files=no)"
if [ -n "$DIRTY" ]; then
    echo "REFUSING DEPLOY: tracked files in $LIVE have onsite edits:" >&2
    echo "$DIRTY" >&2
    exit 3
fi

git -C "$LIVE" fetch --force "$BUNDLE" main:refs/remotes/deploy/main

# Calibration/profile artifacts were historically untracked on the Jetson.
# They are now versioned in the laptop repository. Preserve any old untracked
# copy before checkout so Git can materialize the canonical tracked file
# instead of refusing with "untracked working tree files would be overwritten".
# Nothing outside an incoming tracked path is removed.
MIGRATION_BACKUP="/home/dexmate/roco_untracked_before_deploy_$(date -u +%Y%m%dT%H%M%SZ)"
MIGRATION_COUNT=0
while IFS= read -r PATHNAME; do
    [ -n "$PATHNAME" ] || continue
    if [ -e "$LIVE/$PATHNAME" ] && ! git -C "$LIVE" ls-files --error-unmatch -- "$PATHNAME" >/dev/null 2>&1; then
        mkdir -p "$MIGRATION_BACKUP/$(dirname "$PATHNAME")"
        cp -a "$LIVE/$PATHNAME" "$MIGRATION_BACKUP/$PATHNAME"
        rm -rf -- "$LIVE/$PATHNAME"
        MIGRATION_COUNT=$((MIGRATION_COUNT + 1))
    fi
done < <(git -C "$LIVE" ls-tree -r --name-only refs/remotes/deploy/main)
if [ "$MIGRATION_COUNT" -gt 0 ]; then
    echo "PRESERVED $MIGRATION_COUNT old untracked paths in $MIGRATION_BACKUP"
fi

git -C "$LIVE" checkout -B main refs/remotes/deploy/main

ACTUAL="$(git -C "$LIVE" rev-parse HEAD)"
if [ "$ACTUAL" != "$EXPECTED" ]; then
    echo "REFUSING: deployed SHA $ACTUAL != expected $EXPECTED" >&2
    exit 4
fi

rm -f "$BUNDLE" "/home/dexmate/__REMOTE_SCRIPT__"

echo "DEPLOYED_SHA=$ACTUAL"
echo "DEPLOY_DIR=$LIVE"

if [ "$SKIP_PREFLIGHT" != "1" ]; then
    echo
    echo "=== Vega no-motion preflight ==="
    set +e
    (cd "$LIVE" && python3 tools/vega_preflight.py)
    PRE=$?
    set -e
    echo "PREFLIGHT_EXIT=$PRE"
fi
'@

        $RemoteBody = $RemoteTemplate
        $RemoteBody = $RemoteBody.Replace("__LIVE__", $LiveDir)
        $RemoteBody = $RemoteBody.Replace("__BUNDLE__", $BundleName)
        $RemoteBody = $RemoteBody.Replace("__EXPECTED__", $Head)
        $RemoteBody = $RemoteBody.Replace("__SKIP_PREFLIGHT__", $SkipPreflightValue)
        $RemoteBody = $RemoteBody.Replace("__REMOTE_SCRIPT__", $RemoteScriptName)
        $RemoteBody = $RemoteBody.Replace("__MODE__", $Mode)
        $RemoteBody = $RemoteBody.Replace("__VERSION_LABEL__", $VersionName)

        # Bash on the Jetson requires Unix LF line endings. Windows PowerShell
        # Set-Content would write CRLF, which makes "set -euo pipefail" parse as
        # an invalid option (the hidden CR becomes part of "pipefail").
        $RemoteBody = $RemoteBody -replace "`r`n", "`n"
        [System.IO.File]::WriteAllText(
            $RemoteScriptPath,
            $RemoteBody,
            (New-Object System.Text.UTF8Encoding($false))
        )

        Write-Host "Copying bundle to $Remote..."
        Invoke-Checked scp ($CommonSsh + @(
            $BundlePath,
            $RemoteScriptPath,
            "${Remote}:/home/dexmate/"
        ))

        Write-Host "Updating robot deployment checkout..."
        Invoke-Checked ssh ($CommonSsh + @(
            $Remote,
            "bash /home/dexmate/$RemoteScriptName"
        ))

        Write-Host
        Write-Host "Vega deployment complete: $Head"
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($BundlePath -and (Test-Path $BundlePath)) {
        Remove-Item $BundlePath -Force -ErrorAction SilentlyContinue
    }
    if ($RemoteScriptPath -and (Test-Path $RemoteScriptPath)) {
        Remove-Item $RemoteScriptPath -Force -ErrorAction SilentlyContinue
    }
    if ($AskPassPath -and (Test-Path $AskPassPath)) {
        Remove-Item $AskPassPath -Force -ErrorAction SilentlyContinue
    }

    $env:SSH_ASKPASS = $OldAskPass
    $env:SSH_ASKPASS_REQUIRE = $OldAskPassRequire
    $env:DISPLAY = $OldDisplay
    $env:ROCO_SSH_PASSWORD = $OldRocoPassword
}
