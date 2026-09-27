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
    [switch]$SkipPreflight
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

        # Bash on the Jetson requires Unix LF line endings. Windows PowerShell
        # Set-Content would write CRLF, which makes "set -euo pipefail" parse as
        # an invalid option (the hidden CR becomes part of "pipefail").
        $RemoteBody = $RemoteBody -replace "`r`n", "`n"
        [System.IO.File]::WriteAllText(
            $RemoteScriptPath,
            $RemoteBody,
            (New-Object System.Text.UTF8Encoding($false))
        )

        $CommonSsh = @(
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=8"
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
