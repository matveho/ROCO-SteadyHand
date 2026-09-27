#requires -Version 5.1
<#
.SYNOPSIS
Copy one file from the competition Vega to Windows using SSH_ASKPASS.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Source,
    [Parameter(Mandatory=$true, Position=1)]
    [string]$Destination,
    [string]$HostName = "192.168.50.20",
    [string]$UserName = "dexmate"
)

$ErrorActionPreference = "Stop"
$Remote = "$UserName@$HostName"
$AskPassPath = $null
$OldAskPass = $env:SSH_ASKPASS
$OldAskPassRequire = $env:SSH_ASKPASS_REQUIRE
$OldDisplay = $env:DISPLAY
$OldRocoPassword = $env:ROCO_SSH_PASSWORD

try {
    if (-not (Get-Command scp -ErrorAction SilentlyContinue)) {
        throw "scp is not available in PATH"
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

    $Args = @(
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=8",
        "${Remote}:$Source",
        $Destination
    )
    & scp @Args
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed with exit code $LASTEXITCODE"
    }
}
finally {
    if ($AskPassPath -and (Test-Path $AskPassPath)) {
        Remove-Item $AskPassPath -Force -ErrorAction SilentlyContinue
    }
    $env:SSH_ASKPASS = $OldAskPass
    $env:SSH_ASKPASS_REQUIRE = $OldAskPassRequire
    $env:DISPLAY = $OldDisplay
    $env:ROCO_SSH_PASSWORD = $OldRocoPassword
}
