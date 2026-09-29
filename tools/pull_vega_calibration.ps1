#requires -Version 5.1
<#
.SYNOPSIS
Pull onsite Vega calibration artifacts into the canonical repository paths.

.DESCRIPTION
The Jetson has no Internet connection.  This script uses one SSH/SCP
connection setup and pulls the files that are created by the physical
calibration tools.  Existing local files are copied into an ignored,
timestamped runs/ backup first.  JSON is parsed before it can replace a
canonical file.

The script does not silently commit or push robot measurements.  Use -Commit
after reviewing the printed status, and add -Push when that commit should be
pushed to origin/main.
#>

[CmdletBinding()]
param(
    [string]$HostName = "192.168.50.20",
    [string]$UserName = "dexmate",
    [string]$LiveDir = "/home/dexmate/ROCO-SteadyHand-live",
    [switch]$Commit,
    [switch]$Push,
    [switch]$SkipBoard,
    [switch]$SkipWrist
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Remote = "$UserName@$HostName"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$BackupRoot = Join-Path $RepoRoot "runs\robot_pull_$Stamp"
$StageRoot = Join-Path $BackupRoot "incoming"
$AskPassPath = $null
$OldAskPass = $env:SSH_ASKPASS
$OldAskPassRequire = $env:SSH_ASKPASS_REQUIRE
$OldDisplay = $env:DISPLAY
$OldRocoPassword = $env:ROCO_SSH_PASSWORD
$Pulled = New-Object System.Collections.Generic.List[string]
$ChangedPaths = New-Object System.Collections.Generic.List[string]

function Fail([string]$Message) {
    throw $Message
}

function Invoke-ScpPull {
    param(
        [Parameter(Mandatory=$true)][string]$RemotePath,
        [Parameter(Mandatory=$true)][string]$Destination,
        [switch]$Recursive,
        [switch]$Optional
    )

    $parent = Split-Path -Parent $Destination
    if ($parent) { New-Item -ItemType Directory -Force $parent | Out-Null }
    $source = "{0}:{1}" -f $Remote, $RemotePath
    $scpArgs = @(
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=8"
    )
    if ($Recursive) { $scpArgs += "-r" }
    $scpArgs += @($source, $Destination)
    & scp @scpArgs
    if ($LASTEXITCODE -ne 0) {
        if ($Optional) {
            Write-Warning "Optional robot artifact not available: $RemotePath"
            return $false
        }
        Fail "Required robot artifact could not be copied: $RemotePath"
    }
    return $true
}

function Validate-JsonFile {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Description
    )
    try {
        $value = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    }
    catch {
        Fail "$Description is not valid JSON: $Path"
    }
    if ($null -eq $value) { Fail "$Description is empty: $Path" }
    return $value
}

function Install-JsonArtifact {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$RemotePath,
        [Parameter(Mandatory=$true)][string]$CanonicalPath,
        [switch]$Required
    )
    $stagePath = Join-Path $StageRoot $Name
    $ok = Invoke-ScpPull -RemotePath $RemotePath -Destination $stagePath -Optional:(-not $Required)
    if (-not $ok) { return }
    $value = Validate-JsonFile -Path $stagePath -Description $Name
    if ($Name -eq "head_fallback_profiles.json" -and [int]$value.schema_version -ne 1) {
        Fail "head fallback profile has unsupported schema_version"
    }
    if ($Name -eq "wrist_part_profiles.json" -and [int]$value.schema_version -ne 1) {
        Fail "wrist profile has unsupported schema_version"
    }
    if ($Name -eq "vega_board_manual.json" -and @([int]$value.schema_version) -notcontains 1 -and @([int]$value.schema_version) -notcontains 2) {
        Fail "board calibration has unsupported schema_version"
    }
    if (Test-Path -LiteralPath $CanonicalPath) {
        $backupPath = Join-Path $BackupRoot (Split-Path -Leaf $CanonicalPath)
        Copy-Item -LiteralPath $CanonicalPath -Destination $backupPath -Force
    }
    $canonicalParent = Split-Path -Parent $CanonicalPath
    New-Item -ItemType Directory -Force $canonicalParent | Out-Null
    Copy-Item -LiteralPath $stagePath -Destination $CanonicalPath -Force
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CanonicalPath).Hash
    $Pulled.Add($CanonicalPath) | Out-Null
    $ChangedPaths.Add($CanonicalPath) | Out-Null
    Write-Host ("PULLED {0}  SHA256={1}" -f $CanonicalPath, $hash)
}

function Install-FileArtifact {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$RemotePath,
        [Parameter(Mandatory=$true)][string]$CanonicalPath,
        [switch]$Optional
    )
    $stagePath = Join-Path $StageRoot $Name
    $ok = Invoke-ScpPull -RemotePath $RemotePath -Destination $stagePath -Optional:$Optional
    if (-not $ok) { return }
    if (Test-Path -LiteralPath $CanonicalPath) {
        $backupPath = Join-Path $BackupRoot (Split-Path -Leaf $CanonicalPath)
        Copy-Item -LiteralPath $CanonicalPath -Destination $backupPath -Force
    }
    $canonicalParent = Split-Path -Parent $CanonicalPath
    New-Item -ItemType Directory -Force $canonicalParent | Out-Null
    Copy-Item -LiteralPath $stagePath -Destination $CanonicalPath -Force
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $CanonicalPath).Hash
    $ChangedPaths.Add($CanonicalPath) | Out-Null
    Write-Host ("PULLED {0}  SHA256={1}" -f $CanonicalPath, $hash)
}

try {
    if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
        Fail "scp is not available in PATH. Install/use OpenSSH on the Windows laptop."
    }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Fail "git is not available in PATH. Run this from the repository's Git shell or PowerShell."
    }

    New-Item -ItemType Directory -Force $StageRoot | Out-Null

    $Password = $env:DEXMATE_PASSWORD
    if (-not $Password) {
        $Secure = Read-Host "DexMate SSH password" -AsSecureString
        $Bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
        try { $Password = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Bstr) }
        finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Bstr) }
    }
    $env:ROCO_SSH_PASSWORD = $Password
    $AskPassPath = Join-Path $env:TEMP "roco-ssh-askpass-$PID.cmd"
    @'
@echo off
powershell.exe -NoProfile -Command "[Console]::Out.Write($env:ROCO_SSH_PASSWORD)"
'@ | Set-Content -Path $AskPassPath -Encoding ASCII
    $env:SSH_ASKPASS = $AskPassPath
    $env:SSH_ASKPASS_REQUIRE = "force"
    $env:DISPLAY = "roco-pull-calibration"

    $calibrationDir = "$LiveDir/calibration"
    # The head fallback is optional; the current competition path can run
    # without it and older robots do not have this file.
    Install-JsonArtifact -Name "head_fallback_profiles.json" `
        -RemotePath "$calibrationDir/head_fallback_profiles.json" `
        -CanonicalPath (Join-Path $RepoRoot "calibration\head_fallback_profiles.json") `
        -Required:$false

    if (-not $SkipBoard) {
        Install-JsonArtifact -Name "vega_board_manual.json" `
            -RemotePath "$calibrationDir/vega_board_manual.json" `
            -CanonicalPath (Join-Path $RepoRoot "calibration\vega_board_manual.json")
        Install-JsonArtifact -Name "vega_board_manual_fallback.json" `
            -RemotePath "$calibrationDir/vega_board_manual_fallback.json" `
            -CanonicalPath (Join-Path $RepoRoot "calibration\vega_board_manual_fallback.json") `
            -Required:$false
        Install-JsonArtifact -Name "vega_board_live.json" `
            -RemotePath "$calibrationDir/vega_board_live.json" `
            -CanonicalPath (Join-Path $RepoRoot "calibration\vega_board_live.json") `
            -Required:$false
        Install-JsonArtifact -Name "task_geometry_audit.json" `
            -RemotePath "$calibrationDir/task_geometry_audit.json" `
            -CanonicalPath (Join-Path $RepoRoot "calibration\task_geometry_audit.json") `
            -Required:$false
    }

    if (-not $SkipWrist) {
        Install-JsonArtifact -Name "wrist_part_profiles.json" `
            -RemotePath "$calibrationDir/wrist_part_profiles.json" `
            -CanonicalPath (Join-Path $RepoRoot "calibration\wrist_part_profiles.json")

        $templateStage = Join-Path $StageRoot "wrist_templates"
        if (Invoke-ScpPull -RemotePath "$calibrationDir/wrist_templates" `
                -Destination $StageRoot -Recursive -Optional) {
            $remoteTemplates = Join-Path $StageRoot "wrist_templates"
            if (Test-Path -LiteralPath $remoteTemplates) {
                $localTemplates = Join-Path $RepoRoot "calibration\wrist_templates"
                New-Item -ItemType Directory -Force $localTemplates | Out-Null
                Copy-Item -Path (Join-Path $remoteTemplates "*") -Destination $localTemplates -Recurse -Force
                $ChangedPaths.Add($localTemplates) | Out-Null
                Write-Host "MERGED calibration\wrist_templates\"
            }
        }
    }

    # These are runtime inputs referenced by the robot config but often
    # generated onsite and therefore untracked on the Jetson.
    Install-FileArtifact -Name "vega_1u_competition.urdf" `
        -RemotePath "$LiveDir/configs/robots/vega_1u_competition.urdf" `
        -CanonicalPath (Join-Path $RepoRoot "configs\robots\vega_1u_competition.urdf") `
        -Optional

    $posesStage = Join-Path $StageRoot "poses"
    if (Invoke-ScpPull -RemotePath "$LiveDir/poses" `
            -Destination $StageRoot -Recursive -Optional) {
        if (Test-Path -LiteralPath $posesStage) {
            $localPoses = Join-Path $RepoRoot "poses"
            New-Item -ItemType Directory -Force $localPoses | Out-Null
            Copy-Item -Path (Join-Path $posesStage "*") -Destination $localPoses -Recurse -Force
            $ChangedPaths.Add($localPoses) | Out-Null
            Write-Host "MERGED poses\"
        }
    }

    Write-Host ""
    Write-Host "ROBOT CALIBRATION PULL COMPLETE"
    Write-Host "Backup/staging: $BackupRoot"
    if ($Pulled.Count -eq 0) {
        Write-Warning "No JSON calibration artifacts were found. Nothing was changed."
        exit 2
    }

    Push-Location $RepoRoot
    try {
        $status = @(git status --short)
        if ($status.Count -gt 0) {
            Write-Host ""
            Write-Host "REVIEW THESE CHANGES:"
            $status | ForEach-Object { Write-Host $_ }
            if ($Push) { $Commit = $true }
            if ($Commit) {
                $stagePaths = New-Object System.Collections.Generic.List[string]
                foreach ($changed in $ChangedPaths) {
                    $relative = $changed.Substring($RepoRoot.Length).TrimStart([char[]]"\\/")
                    if ($relative -and -not $stagePaths.Contains($relative)) {
                        $stagePaths.Add($relative) | Out-Null
                    }
                }
                if ($stagePaths.Count -eq 0) { Fail "No pulled paths available to stage" }
                git add -- $stagePaths.ToArray()
                if ($LASTEXITCODE -ne 0) { Fail "git add failed" }
                git commit -m "Record onsite Vega calibration artifacts"
                if ($LASTEXITCODE -ne 0) { Fail "git commit failed" }
                if ($Push) {
                    git push origin main
                    if ($LASTEXITCODE -ne 0) { Fail "git push failed" }
                    Write-Host "PUSHED origin/main"
                }
            }
            else {
                Write-Host "No commit made. Review with: git diff -- calibration"
                Write-Host "Commit later with: git add calibration; git commit -m 'Record onsite Vega calibration artifacts'"
            }
        }
        else {
            Write-Host "No tracked calibration changes detected."
        }
    }
    finally { Pop-Location }
}
finally {
    if ($AskPassPath -and (Test-Path $AskPassPath)) { Remove-Item $AskPassPath -Force -ErrorAction SilentlyContinue }
    $env:SSH_ASKPASS = $OldAskPass
    $env:SSH_ASKPASS_REQUIRE = $OldAskPassRequire
    $env:DISPLAY = $OldDisplay
    $env:ROCO_SSH_PASSWORD = $OldRocoPassword
}
