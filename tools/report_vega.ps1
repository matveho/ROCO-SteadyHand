#requires -Version 5.1
<#
.SYNOPSIS
Collect robot competition evidence over SCP and push a report to origin/main.
.DESCRIPTION
No deployment is needed: uploads a standalone read-only collector to /tmp.
Publishes only competition_status/latest through a temporary Git index. The
laptop branch, index, code, and calibrations are not reset or overwritten.
Run after a test finishes. Failures are evidence and do not prevent collection.
#>
[CmdletBinding()]
param(
    [string]$HostName = "192.168.50.20",
    [string]$UserName = "dexmate",
    [string]$LiveDir = "/home/dexmate/ROCO-SteadyHand-live",
    [string]$RobotPython = "/home/dexmate/miniconda3/bin/python3",
    [ValidateRange(1,100)][int]$RecentRuns = 8,
    [ValidateRange(16,2048)][int]$MaxMb = 256,
    [string[]]$IncludeRun = @(),
    [string]$Note = "",
    [string[]]$Video = @()
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Remote = "$UserName@$HostName"
$SshOptions = @("-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=8")
$Token = [Guid]::NewGuid().ToString("N")
$Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$LocalRoot = Join-Path $RepoRoot "runs\robot_report_${Stamp}_$Token"
$Payload = Join-Path $LocalRoot "payload"
$RemoteScript = "/tmp/roco-report-$Token.py"
$RemoteZip = "/tmp/roco-report-$Token.zip"
$AskPass = Join-Path ([IO.Path]::GetTempPath()) "roco-report-askpass-$Token.cmd"
$IndexPath = Join-Path $LocalRoot "publish.index"
$RemoteUploaded = $false
$OldEnvironment = @{}
foreach ($Name in @("SSH_ASKPASS", "SSH_ASKPASS_REQUIRE", "DISPLAY", "ROCO_SSH_PASSWORD", "GIT_INDEX_FILE")) {
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
    if ($HostName -notmatch '^[A-Za-z0-9._-]+$' -or $UserName -notmatch '^[A-Za-z0-9._-]+$') {
        throw "Invalid SSH host/user"
    }
    foreach ($PathValue in @($LiveDir, $RobotPython)) {
        if ($PathValue -notmatch '^/[A-Za-z0-9._/-]+$') { throw "Unsupported remote path: $PathValue" }
    }
    foreach ($Run in $IncludeRun) {
        if ($Run -notmatch '^[A-Za-z0-9._/-]+$' -or $Run -match '(^|/)\.\.(/|$)') { throw "Invalid run path: $Run" }
    }
    New-Item -ItemType Directory -Force $Payload | Out-Null
    $Password = if ($env:DEXMATE_PASSWORD) { $env:DEXMATE_PASSWORD } else { "hello-dex" }
    $env:ROCO_SSH_PASSWORD = $Password
    @'
@echo off
powershell.exe -NoProfile -Command "[Console]::Out.Write($env:ROCO_SSH_PASSWORD)"
'@ | Set-Content -LiteralPath $AskPass -Encoding ASCII
    $env:SSH_ASKPASS = $AskPass
    $env:SSH_ASKPASS_REQUIRE = "force"
    $env:DISPLAY = "roco-status"

    Write-Host "Collecting robot state and recent original test images (no motion)..."
    Invoke-Checked scp ($SshOptions + @((Join-Path $PSScriptRoot "vega_collect_status.py"), "${Remote}:$RemoteScript"))
    $RemoteUploaded = $true
    # Explicit Conda interpreter avoids the incompatible Python chosen by a
    # non-interactive SSH login. The collector itself needs only the stdlib.
    $RemoteCommand = "if [ -x '$RobotPython' ]; then REPORT_PY='$RobotPython'; else REPORT_PY=/usr/bin/python3; fi; " +
        "unset PYTHONPATH; " + '$REPORT_PY' + " '$RemoteScript' --root '$LiveDir' --output '$RemoteZip' --recent-runs $RecentRuns --max-mb $MaxMb"
    foreach ($Run in $IncludeRun) { $RemoteCommand += " --include-run '$Run'" }
    $Collected = Read-Checked ssh ($SshOptions + @($Remote, $RemoteCommand))
    Write-Host $Collected
    $HashMatch = [regex]::Match($Collected, '(?m)^BUNDLE_SHA256=([a-f0-9]{64})\s*$')
    if (-not $HashMatch.Success) { throw "Collector did not return a bundle hash" }
    $ZipPath = Join-Path $LocalRoot "robot-evidence.zip"
    Invoke-Checked scp ($SshOptions + @("${Remote}:$RemoteZip", $ZipPath))
    if ((Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $HashMatch.Groups[1].Value) {
        throw "Evidence ZIP hash mismatch; nothing will be pushed"
    }
    # Validate names and types before allowing Windows to extract anything.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $Zip = [IO.Compression.ZipFile]::OpenRead($ZipPath)
    try {
        foreach ($Entry in $Zip.Entries) {
            $Name = $Entry.FullName
            $Mode = ([int64]$Entry.ExternalAttributes -shr 16) -band 61440
            if ($Name -notmatch '^competition_status/latest/.+' -or $Name -match '(^|/)\.\.(/|$)|\\|:' -or $Mode -eq 40960) {
                throw "Unexpected archive member: $Name"
            }
        }
    }
    finally { $Zip.Dispose() }
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $Payload
    $Latest = Join-Path $Payload "competition_status\latest"
    $Manifest = Get-Content -Raw -LiteralPath (Join-Path $Latest "manifest.json") | ConvertFrom-Json
    foreach ($Property in $Manifest.PSObject.Properties) {
        $Path = Join-Path $Latest $Property.Name
        if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Property.Value.sha256) {
            throw "Extracted evidence hash mismatch: $($Property.Name)"
        }
    }
    $Report = Get-Content -Raw -LiteralPath (Join-Path $Latest "report.json") | ConvertFrom-Json
    $Operator = @{ note = $Note; captured_at_utc = (Get-Date).ToUniversalTime().ToString("o"); videos = @() }
    foreach ($Source in $Video) {
        $Item = Get-Item -LiteralPath $Source
        if ($Item.PSIsContainer -or $Item.Length -gt 40MB) { throw "Video must be a file under 40 MB: $Source" }
        if ($Item.Extension -notmatch '^\.(mp4|mov|mkv|avi|webm)$') { throw "Unsupported video type: $Source" }
        $Directory = Join-Path $Latest "operator"
        New-Item -ItemType Directory -Force $Directory | Out-Null
        $VideoName = "video_$($Operator.videos.Count)$($Item.Extension.ToLowerInvariant())"
        Copy-Item -LiteralPath $Item.FullName -Destination (Join-Path $Directory $VideoName)
        $Operator.videos += "operator/$VideoName"
    }
    $Utf8 = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText((Join-Path $Latest "operator_observations.json"), ($Operator | ConvertTo-Json -Depth 5), $Utf8)
    # Publish on the current remote main tree using a separate index/work tree.
    # This includes ONLY report files: unrelated laptop edits/staged commits
    # cannot accidentally become part of the status upload.
    $GitDir = Read-Checked git @("-C", $RepoRoot, "rev-parse", "--absolute-git-dir")
    $GitArgs = @("--git-dir=$GitDir", "--work-tree=$Payload")
    $env:GIT_INDEX_FILE = $IndexPath
    $Commit = $null
    for ($Attempt = 1; $Attempt -le 3; $Attempt++) {
        Invoke-Checked git @("-C", $RepoRoot, "fetch", "origin", "main")
        $Parent = Read-Checked git @("-C", $RepoRoot, "rev-parse", "FETCH_HEAD")
        Invoke-Checked git ($GitArgs + @("read-tree", $Parent))
        Invoke-Checked git ($GitArgs + @("rm", "--quiet", "-r", "--cached", "--ignore-unmatch", "--", "competition_status/latest"))
        Invoke-Checked git ($GitArgs + @("add", "--force", "--all", "--", "competition_status/latest"))
        $Tree = Read-Checked git ($GitArgs + @("write-tree"))
        $Commit = Read-Checked git ($GitArgs + @("commit-tree", $Tree, "-p", $Parent,
            "-m", "Report Vega competition state $Stamp (robot $($Report.git_revision))"))
        & git -C $RepoRoot push origin "${Commit}:refs/heads/main"
        if ($LASTEXITCODE -eq 0) { break }
        if ($Attempt -eq 3) { throw "Push failed; evidence remains in $LocalRoot. No laptop or robot calibration was replaced." }
        Write-Host "Push did not complete; refreshing remote main and retrying ($Attempt/3)..."
    }
    Write-Host ""
    Write-Host "PUSHED COMPETITION REPORT: $Commit"
    Write-Host "Repository evidence: competition_status/latest/report.json"
    Write-Host "Local backup: $LocalRoot"
    Write-Host "Robot code: $($Report.git_revision)"
    foreach ($Check in $Report.diagnostics.PSObject.Properties) {
        Write-Host ("  {0}: exit={1} {2}" -f $Check.Name, $Check.Value.returncode, $Check.Value.error)
    }
    Write-Host "Warnings: $(@($Report.warnings).Count); omitted files: $(@($Report.omitted_files).Count); missing references: $(@($Report.missing_references).Count)"
    Write-Host "Send the agent the PUSHED COMPETITION REPORT line and your observed result. Run again after the next test."
}
finally {
    # Only remove this invocation's two disposable /tmp artifacts on the robot.
    if ($RemoteUploaded) {
        try { & ssh @SshOptions $Remote "rm -f -- '$RemoteScript' '$RemoteZip'" 2>$null }
        catch { Write-Warning "Temporary robot report files remain in /tmp; evidence was preserved." }
    }
    if (Test-Path -LiteralPath $AskPass) { Remove-Item -LiteralPath $AskPass -Force -ErrorAction SilentlyContinue }
    if (Test-Path -LiteralPath $IndexPath) { Remove-Item -LiteralPath $IndexPath -Force -ErrorAction SilentlyContinue }
    foreach ($Name in $OldEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($Name, $OldEnvironment[$Name], "Process")
    }
}
