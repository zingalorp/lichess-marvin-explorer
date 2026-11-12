[CmdletBinding()]
param(
    [string]$Python,
    [string]$Weights,
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Device = 'auto',
    [switch]$Uninstall,
    [switch]$ValidateOnly,
    [switch]$SkipRegistration
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$HostName = 'com.marvin.lichess_explorer'
$ExtensionId = (Get-Content -LiteralPath (Join-Path $ProjectRoot 'extension-id.txt') -Raw).Trim()
$ManifestPath = Join-Path $ProjectRoot "$HostName.json"
$LauncherPath = Join-Path $ProjectRoot 'host/run_host.cmd'
$RegistryPaths = @(
    "HKCU:\Software\Google\Chrome\NativeMessagingHosts\$HostName",
    "HKCU:\Software\Microsoft\Edge\NativeMessagingHosts\$HostName"
)
if ($Uninstall) {
    foreach ($RegistryPath in $RegistryPaths) {
        if (Test-Path -LiteralPath $RegistryPath) { Remove-Item -LiteralPath $RegistryPath -Force }
    }
    Write-Host 'Marvin Explorer host unregistered. Project files and other extensions are unchanged.'
    return
}
if (-not $Python) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) { $Python = $PythonCommand.Source }
    else { throw 'Pass -Python with the full path to a Python interpreter with the host dependencies installed.' }
}
$Python = (Resolve-Path -LiteralPath $Python).Path
# Windows PowerShell can turn native stderr warnings into terminating errors.
# Check the interpreter's exit status, rather than treating a warning as failure.
$PreviousErrorPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $Python -c 'import chess, numpy, onnxruntime' 2>$null
    $DependencyExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $PreviousErrorPreference
}
if ($DependencyExitCode -ne 0) { throw "Install the dependencies first: & '$Python' -m pip install -r '$ProjectRoot\host\requirements.txt'" }
# -Weights is the folder holding marvin_79m.onnx, marvin_23m.onnx and their .offset.npz files
# (a path to one of the .onnx files also works).
if (-not $Weights) {
    $ExistingConfig = Join-Path $ProjectRoot 'host-config.json'
    $Weights = Join-Path $ProjectRoot 'weights'
    if (Test-Path -LiteralPath $ExistingConfig) {
        $Saved = (Get-Content -LiteralPath $ExistingConfig -Raw | ConvertFrom-Json).weightsDir
        if ($Saved) { $Weights = $Saved }
    }
}
$Weights = (Resolve-Path -LiteralPath $Weights).Path
if (Test-Path -LiteralPath $Weights -PathType Leaf) { $Weights = Split-Path -Parent $Weights }
if (-not (Test-Path -LiteralPath (Join-Path $Weights 'marvin_79m.onnx') -PathType Leaf)) {
    throw "Model weights not found: $Weights\marvin_79m.onnx"
}
if ($ValidateOnly) {
    Write-Host 'Interpreter, dependencies and model files are present. No settings were changed.'
    return
}
$Config = @{weightsDir = $Weights; device = $Device}
$ExistingConfig = Join-Path $ProjectRoot 'host-config.json'
if (Test-Path -LiteralPath $ExistingConfig) {
    $PreviousConfig = Get-Content -LiteralPath $ExistingConfig -Raw | ConvertFrom-Json
    foreach ($Property in $PreviousConfig.PSObject.Properties) {
        if ($Property.Name -notin @('weights', 'weightsDir', 'device')) { $Config[$Property.Name] = $Property.Value }
    }
}
$Utf8 = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllText($ExistingConfig, ($Config | ConvertTo-Json -Depth 4), $Utf8)
# Native framing uses stdout. Interpreter diagnostics go to stderr.
$BundledPython = Join-Path $ProjectRoot 'runtime/python.exe'
if ($Python -eq $BundledPython) {
    $Launcher = "@echo off`r`n`"%~dp0..\runtime\python.exe`" `"%~dp0native_host.py`"`r`n"
} else {
    $Launcher = "@echo off`r`n`"$Python`" `"%~dp0native_host.py`"`r`n"
}
[IO.File]::WriteAllText($LauncherPath, $Launcher, [Text.Encoding]::Default)
$Manifest = [ordered]@{
    name = $HostName
    description = 'Local Marvin model for the Lichess opening explorer'
    path = $LauncherPath
    type = 'stdio'
    allowed_origins = @("chrome-extension://$ExtensionId/")
}
[IO.File]::WriteAllText($ManifestPath, ($Manifest | ConvertTo-Json), $Utf8)
if (-not $SkipRegistration) {
    foreach ($RegistryPath in $RegistryPaths) {
        if (-not (Test-Path -LiteralPath $RegistryPath)) { New-Item -Path $RegistryPath -Force | Out-Null }
        Set-ItemProperty -LiteralPath $RegistryPath -Name '(default)' -Value $ManifestPath
    }
}
Write-Host "Ready. Load $ProjectRoot\extension at chrome://extensions or edge://extensions."
Write-Host "Expected extension ID: $ExtensionId"
Write-Host 'Open https://lichess.org/analysis#explorer/lichess and choose Marvin.'
