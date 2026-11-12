[CmdletBinding()]
param([switch]$ValidateOnly, [switch]$Uninstall)
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$Installer = Join-Path $ProjectRoot 'install.ps1'
try {
    if ($Uninstall) {
        # Only remove a registration owned by this extracted copy.
        $ExpectedManifest = Join-Path $ProjectRoot 'com.marvin.lichess_explorer.json'
        foreach ($Browser in @('Google\Chrome', 'Microsoft\Edge')) {
            $Key = "HKCU:\Software\$Browser\NativeMessagingHosts\com.marvin.lichess_explorer"
            if (Test-Path -LiteralPath $Key) {
                $Registered = (Get-Item -LiteralPath $Key).GetValue('')
                if ($Registered -eq $ExpectedManifest) {
                    Remove-Item -LiteralPath $Key -Force
                } else {
                    Write-Host 'Another Marvin installation is registered. Its registration was left in place.'
                }
            }
        }
        Write-Host 'Host unregistered for this copy. Remove Marvin Opening Explorer on the browser extensions page.'
        return
    }
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw 'This bundle requires 64-bit Windows on an Intel or AMD PC.'
    }
    $Python = Join-Path $ProjectRoot 'runtime/python.exe'
    $Weights = Join-Path $ProjectRoot 'weights'
    $NativeLibrary = Join-Path $ProjectRoot 'runtime/Lib/site-packages/onnxruntime/capi/onnxruntime_providers_shared.dll'
    if ($NativeLibrary.Length -ge 260) {
        throw 'This folder path is too long for the bundled native libraries. Move the extracted folder to a shorter location, such as Documents\Marvin, then run Setup.cmd again.'
    }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw 'The bundled runtime is missing. Use Extract All on the ZIP before running Setup.cmd.'
    }
    Write-Host 'Checking the bundled runtime and model. This can take a moment...'
    & $Python (Join-Path $ProjectRoot 'host/check_installation.py')
    if ($LASTEXITCODE -ne 0) { throw 'The bundled model check failed. See the error above.' }
    & $Installer -Python $Python -Weights $Weights -Device cpu -ValidateOnly:$ValidateOnly
    if ($ValidateOnly) { return }
    Write-Host ''
    Write-Host 'Setup complete. No Python or CUDA installation is needed.'
    Write-Host '1. Open chrome://extensions or edge://extensions.'
    Write-Host '2. Turn on Developer mode and click Load unpacked.'
    Write-Host "3. Select this folder: $ProjectRoot\extension"
    Write-Host '4. Sign in to Lichess, open an analysis board or study, open the book panel and choose Marvin.'
    Write-Host ''
    Write-Host 'Keep the extracted folder. If you move it, run Setup.cmd again.'
} catch {
    Write-Host ''
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
