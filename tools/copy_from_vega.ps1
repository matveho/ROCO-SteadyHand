#requires -Version 5.1
<#
.SYNOPSIS
Copy one file from the competition Vega, or perform the full handover cleanup.

.DESCRIPTION
Normal mode keeps the original one-file SCP behavior.

-HandoverCleanup runs from the Windows/laptop checkout and:
- pulls local main with --ff-only;
- merges persistent robot state from ~/ROCO-SteadyHand and ~/ROCO-SteadyHand-live
  (live wins) for calibration/, configs/, and tools/outputs/;
- copies that state into the laptop checkout;
- commits/pushes any state changes and verifies origin/main matches;
- refuses deletion if either robot checkout contains unexpected Git changes;
- removes both robot checkouts and known temporary deploy artifacts.

Robot deletion never runs if backup, local commit, push, or remote verification fails.
#>

[CmdletBinding()]
param(
    [Parameter(Position=0)]
    [string]$Source,
    [Parameter(Position=1)]
    [string]$Destination,
    [string]$HostName = "192.168.50.20",
    [string]$UserName = "dexmate",
    [switch]$HandoverCleanup
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Remote = "$UserName@$HostName"
$AskPassPath = $null
$ArchiveLocalPath = $null
$RemoteScriptPath = $null
$RemoteArchivePath = $null
$RemoteScriptRemotePath = $null
$OldAskPass = $env:SSH_ASKPASS
$OldAskPassRequire = $env:SSH_ASKPASS_REQUIRE
$OldDisplay = $env:DISPLAY
$OldRocoPassword = $env:ROCO_SSH_PASSWORD

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

try {
    if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
        throw "scp is not available in PATH"
    }

    if ($HandoverCleanup) {
        foreach ($Command in @("ssh", "git", "tar")) {
            if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
                throw "$Command is not available in PATH"
            }
        }
    }
    elseif (-not $Source -or -not $Destination) {
        throw "Normal copy mode requires Source and Destination, or use -HandoverCleanup"
    }

    $Password = $env:DEXMATE_PASSWORD
    if (-not $Password) {
        $Secure = Read-Host "DexMate SSH password" -AsSecureString
        $Bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
        try {
            $Password = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Bstr)
        }
        finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Bstr)
        }
    }

    $env:ROCO_SSH_PASSWORD = $Password
    $AskPassPath = Join-Path $env:TEMP "roco-ssh-askpass-$PID.cmd"
    @'
@echo off
powershell.exe -NoProfile -Command "[Console]::Out.Write($env:ROCO_SSH_PASSWORD)"
'@ | Set-Content -Path $AskPassPath -Encoding ASCII

    $env:SSH_ASKPASS = $AskPassPath
    $env:SSH_ASKPASS_REQUIRE = "force"
    $env:DISPLAY = "roco-copy"

    $CommonSsh = @(
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=8"
    )

    if (-not $HandoverCleanup) {
        Invoke-Checked scp ($CommonSsh + @(
            "$($Remote):$Source",
            $Destination
        ))
        return
    }

    Push-Location $RepoRoot
    try {
        if (-not (Test-Path ".git")) {
            throw "Repository root not found at $RepoRoot"
        }

        $InitialStatus = (& git status --porcelain)
        if ($LASTEXITCODE -ne 0) {
            throw "Could not read local Git status"
        }
        if ($InitialStatus) {
            throw "Local checkout has uncommitted changes. Commit/stash them before handover cleanup."
        }

        $Branch = (& git branch --show-current).Trim()
        if ($LASTEXITCODE -ne 0 -or $Branch -ne "main") {
            throw "Handover cleanup must run from the laptop main branch"
        }

        Write-Host "Updating laptop main..."
        Invoke-Checked git @("pull", "--ff-only", "origin", "main")

        $ArchiveName = "roco-handover-$PID.tgz"
        $RemoteScriptName = "roco-handover-$PID.sh"
        $ArchiveLocalPath = Join-Path $env:TEMP $ArchiveName
        $RemoteScriptPath = Join-Path $env:TEMP $RemoteScriptName
        $RemoteArchivePath = "/home/dexmate/$ArchiveName"
        $RemoteScriptRemotePath = "/home/dexmate/$RemoteScriptName"

        $RemoteTemplate = @'
#!/usr/bin/env bash
set -euo pipefail

LIVE="/home/dexmate/ROCO-SteadyHand-live"
MAIN="/home/dexmate/ROCO-SteadyHand"
ARCHIVE="/home/dexmate/__ARCHIVE__"
STAGE="/home/dexmate/.roco-handover-stage-__PID__"
SELF="/home/dexmate/__SCRIPT__"
MODE="$1"

copy_state() {
    repo="$1"
    [ -d "$repo" ] || return 0
    for item in calibration configs tools/outputs; do
        if [ -d "$repo/$item" ]; then
            mkdir -p "$STAGE/$item"
            cp -a "$repo/$item/." "$STAGE/$item/"
        fi
    done
}

unexpected_changes() {
    repo="$1"
    [ -d "$repo/.git" ] || return 0
    status="$(git -C "$repo" status --porcelain --untracked-files=all)"
    [ -z "$status" ] && return 0
    bad="$(printf '%s\n' "$status" | grep -Ev '^.. (calibration/|configs/|tools/outputs/)' || true)"
    if [ -n "$bad" ]; then
        echo "REFUSING DELETE: unexpected Git changes in $repo:" >&2
        echo "$bad" >&2
        return 1
    fi
}

case "$MODE" in
    backup)
        rm -rf -- "$STAGE"
        rm -f -- "$ARCHIVE"
        mkdir -p "$STAGE"

        copy_state "$MAIN"
        copy_state "$LIVE"

        if [ -z "$(find "$STAGE" -mindepth 1 -print -quit)" ]; then
            echo "No calibration/config/annotation state found on robot" >&2
            exit 3
        fi

        tar -C "$STAGE" -czf "$ARCHIVE" .
        rm -rf -- "$STAGE"
        echo "HANDOVER_ARCHIVE=$ARCHIVE"
        ;;

    delete)
        unexpected_changes "$MAIN"
        unexpected_changes "$LIVE"

        rm -rf -- "$LIVE" "$MAIN"
        rm -f -- "$ARCHIVE"
        rm -f /home/dexmate/roco-main-*.bundle /home/dexmate/roco-deploy-*.sh
        rm -rf -- "$STAGE"

        if [ -e "$LIVE" ] || [ -e "$MAIN" ]; then
            echo "Repository removal verification failed" >&2
            exit 4
        fi

        echo "ROBOT_REPOS_REMOVED"
        rm -f -- "$SELF"
        ;;

    cleanup-temp)
        rm -f -- "$ARCHIVE" "$SELF"
        rm -rf -- "$STAGE"
        ;;

    *)
        echo "Usage: $SELF backup|delete|cleanup-temp" >&2
        exit 2
        ;;
esac
'@

        $RemoteBody = $RemoteTemplate
        $RemoteBody = $RemoteBody.Replace("__ARCHIVE__", $ArchiveName)
        $RemoteBody = $RemoteBody.Replace("__SCRIPT__", $RemoteScriptName)
        $RemoteBody = $RemoteBody.Replace("__PID__", [string]$PID)
        $RemoteBody = $RemoteBody.Replace([Environment]::NewLine, [string][char]10)
        [System.IO.File]::WriteAllText(
            $RemoteScriptPath,
            $RemoteBody,
            (New-Object System.Text.UTF8Encoding($false))
        )

        Write-Host "Preparing robot state archive..."
        Invoke-Checked scp ($CommonSsh + @(
            $RemoteScriptPath,
            "$($Remote):/home/dexmate/"
        ))
        Invoke-Checked ssh ($CommonSsh + @(
            $Remote,
            "bash $RemoteScriptRemotePath backup"
        ))

        Write-Host "Copying calibration/config/annotation state to laptop..."
        Invoke-Checked scp ($CommonSsh + @(
            "$($Remote):$RemoteArchivePath",
            $ArchiveLocalPath
        ))

        $Entries = & tar -tzf $ArchiveLocalPath
        if ($LASTEXITCODE -ne 0) {
            throw "Could not inspect handover archive"
        }
        foreach ($Entry in $Entries) {
            $Normalized = ($Entry -replace '^\./', '').TrimEnd('/')
            if (-not $Normalized -or $Normalized -eq ".") {
                continue
            }
            if (
                $Normalized.StartsWith("/") -or
                $Normalized -match '(^|/)\.\.(/|$)' -or
                $Normalized -notmatch '^(calibration|configs|tools/outputs)(/|$)'
            ) {
                throw "Unexpected path in handover archive: $Entry"
            }
        }

        Invoke-Checked tar @("-xzf", $ArchiveLocalPath, "-C", $RepoRoot)

        Invoke-Checked git @(
            "add", "-A", "--",
            "calibration", "configs", "tools/outputs"
        )

        & git diff --cached --quiet -- calibration configs tools/outputs
        $DiffExit = $LASTEXITCODE
        if ($DiffExit -gt 1) {
            throw "git diff failed with exit code $DiffExit"
        }

        if ($DiffExit -eq 1) {
            $Stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
            Invoke-Checked git @("commit", "-m", "Preserve Vega handover state $Stamp")
        }
        else {
            Write-Host "No new persistent robot state to commit."
        }

        Write-Host "Pushing laptop state before robot deletion..."
        Invoke-Checked git @("push", "origin", "main")

        $LocalHead = (& git rev-parse HEAD).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $LocalHead) {
            throw "Could not resolve local HEAD after push"
        }

        $RemoteHeadLine = (& git ls-remote origin refs/heads/main)
        if ($LASTEXITCODE -ne 0 -or -not $RemoteHeadLine) {
            throw "Could not verify origin/main after push"
        }
        $RemoteHead = ($RemoteHeadLine -split '\s+')[0]
        if ($RemoteHead -ne $LocalHead) {
            throw "origin/main verification failed; robot repos were NOT deleted"
        }

        Write-Host "Push verified at $LocalHead"
        Write-Host "Removing robot repository copies..."
        Invoke-Checked ssh ($CommonSsh + @(
            $Remote,
            "bash $RemoteScriptRemotePath delete"
        ))

        Write-Host
        Write-Host "HANDOVER CLEANUP COMPLETE"
        Write-Host "Saved/pushed: calibration/, configs/, tools/outputs/"
        Write-Host "Removed: /home/dexmate/ROCO-SteadyHand-live and /home/dexmate/ROCO-SteadyHand"
    }
    finally {
        Pop-Location
    }
}
finally {
    if ($ArchiveLocalPath -and (Test-Path $ArchiveLocalPath)) {
        Remove-Item $ArchiveLocalPath -Force -ErrorAction SilentlyContinue
    }
    if ($RemoteScriptPath -and (Test-Path $RemoteScriptPath)) {
        Remove-Item $RemoteScriptPath -Force -ErrorAction SilentlyContinue
    }

    if ($HandoverCleanup -and $RemoteScriptRemotePath -and (Get-Command ssh -ErrorAction SilentlyContinue)) {
        try {
            $CleanupArgs = @(
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=4",
                $Remote,
                "rm -f '$RemoteArchivePath' '$RemoteScriptRemotePath'; rm -rf '/home/dexmate/.roco-handover-stage-$PID'"
            )
            & ssh @CleanupArgs 2>$null
        }
        catch {
        }
    }

    if ($AskPassPath -and (Test-Path $AskPassPath)) {
        Remove-Item $AskPassPath -Force -ErrorAction SilentlyContinue
    }

    $env:SSH_ASKPASS = $OldAskPass
    $env:SSH_ASKPASS_REQUIRE = $OldAskPassRequire
    $env:DISPLAY = $OldDisplay
    $env:ROCO_SSH_PASSWORD = $OldRocoPassword
}
