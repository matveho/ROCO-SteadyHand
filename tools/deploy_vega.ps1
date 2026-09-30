#requires -Version 5.1
<#
.SYNOPSIS
Fast in-place competition deployment over Ethernet. Keeps robot calibration/runs.
#>
[CmdletBinding()]
param(
    [string]$HostName = "192.168.50.20",
    [string]$UserName = "dexmate",
    [string]$LiveDir = "/home/dexmate/ROCO-SteadyHand-live",
    [switch]$SkipPull,
    [switch]$Preflight,
    [switch]$SkipPreflight,
    # Accepted for older copy/paste commands; all deployments now overwrite in place.
    [ValidateSet("replace", "new")][string]$DeployMode = "replace",
    [string]$VersionName
)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Remote = "$UserName@$HostName"
$SshOptions = @("-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=8")
$Token = [Guid]::NewGuid().ToString("N")
$TempDir = Join-Path ([IO.Path]::GetTempPath()) "roco-deploy-$Token"
$RemoteDir = "/tmp/roco-deploy-$Token"
$OldEnvironment = @{}
foreach ($Name in @("SSH_ASKPASS", "SSH_ASKPASS_REQUIRE", "DISPLAY", "ROCO_SSH_PASSWORD")) {
    $OldEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, "Process")
}
function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}
function Read-Checked([string]$Exe, [string[]]$Arguments) {
    $Output = & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
    return ($Output -join "`n").Trim()
}
try {
    foreach ($Command in @("git", "ssh", "scp")) {
        if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) { throw "$Command is not in PATH" }
    }
    if ($HostName -notmatch '^[A-Za-z0-9._-]+$' -or $UserName -notmatch '^[A-Za-z0-9._-]+$' -or
        $LiveDir -notmatch '^/[A-Za-z0-9._/-]+$' -or $LiveDir -eq '/' -or $LiveDir -match '(^|/)\.\.(/|$)') {
        throw "Invalid SSH host/user or deployment directory"
    }
    New-Item -ItemType Directory -Path $TempDir | Out-Null
    $env:ROCO_SSH_PASSWORD = if ($env:DEXMATE_PASSWORD) { $env:DEXMATE_PASSWORD } else { "hello-dex" }
    $AskPass = Join-Path $TempDir "askpass.cmd"
    @'
@echo off
powershell.exe -NoProfile -Command "[Console]::Out.Write($env:ROCO_SSH_PASSWORD)"
'@ | Set-Content -LiteralPath $AskPass -Encoding ASCII
    $env:SSH_ASKPASS = $AskPass
    $env:SSH_ASKPASS_REQUIRE = "force"
    $env:DISPLAY = "roco-deploy"
    if ($DeployMode -eq "new" -or $VersionName) { Write-Host "Archive options ignored: fast in-place deployment." }
    Push-Location $RepoRoot
    try {
        if ((Read-Checked git @("branch", "--show-current")) -ne "main") { throw "Switch the laptop to main before deploying." }
        if (-not $SkipPull) {
            Write-Host "Updating local main..."
            Invoke-Checked git @("pull", "--ff-only", "origin", "main")
        }
        $Head = Read-Checked git @("rev-parse", "main")
        $RobotHead = Read-Checked ssh ($SshOptions + @($Remote,
            "if [ -e '$LiveDir/.git' ]; then git -C '$LiveDir' rev-parse HEAD; else echo NEW; fi"))
        $Payload = Join-Path $TempDir "payload"
        New-Item -ItemType Directory -Path $Payload | Out-Null
        $Bundle = "-"
        if ($RobotHead -ne $Head) {
            $BundlePath = Join-Path $Payload "main.bundle"
            $BundleArgs = @("-c", "pack.compression=1", "bundle", "create", $BundlePath, "main")
            $History = (Read-Checked git @("rev-list", "main")) -split "`n"
            if ($RobotHead -match '^[0-9a-f]{40}$' -and $History -contains $RobotHead) {
                $BundleArgs += "^$RobotHead"
                Write-Host "Bundling only changes since $($RobotHead.Substring(0, 8))..."
            } else { Write-Host "Bundling main for initial deployment..." }
            Invoke-Checked git $BundleArgs
            $Bundle = "$RemoteDir/main.bundle"
        } else { Write-Host "Robot already has this commit; refreshing code and settings." }
        # Copy working laptop settings too, so last-minute JSON edits need no commit.
        foreach ($Relative in @("configs/competition_actions.json", "configs/competition_plan.json", "competition_offsets.json")) {
            $Source = Join-Path $RepoRoot $Relative
            $Text = [IO.File]::ReadAllText($Source)
            $null = ConvertFrom-Json -InputObject $Text
            [IO.File]::WriteAllText((Join-Path $Payload (Split-Path $Relative -Leaf)), $Text,
                (New-Object System.Text.UTF8Encoding($false)))
        }
        $Script = [IO.File]::ReadAllText((Join-Path $PSScriptRoot "deploy_vega_remote.sh")) -replace "`r`n", "`n"
        [IO.File]::WriteAllText((Join-Path $Payload "deploy.sh"), $Script,
            (New-Object System.Text.UTF8Encoding($false)))
        Write-Host "Transferring competition update..."
        Invoke-Checked scp ($SshOptions + @("-r", $Payload, "${Remote}:$RemoteDir"))
        $Check = if ($Preflight -and -not $SkipPreflight) { "1" } else { "0" }
        $Apply = "bash '$RemoteDir/deploy.sh' '$LiveDir' '$Bundle' '$Head' '$RemoteDir' '$Check'"
        # Keep transfer cleanup in the same SSH connection, preserving its exit code.
        $Apply += '; result=$?; rm -rf -- ' + "'$RemoteDir'" + '; exit "$result"'
        Invoke-Checked ssh ($SshOptions + @($Remote, $Apply))
        Write-Host "Deployment complete: $Head. Restart the robot menu."
    } finally { Pop-Location }
} finally {
    if (Test-Path -LiteralPath $TempDir) { Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue }
    foreach ($Name in $OldEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name, $OldEnvironment[$Name], "Process")
    }
}
