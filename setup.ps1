[CmdletBinding()]
# -Unattended (for testing) changes nothing outside this folder: no browser
# registration, clipboard or browser window.
param([switch]$ValidateOnly, [switch]$Uninstall, [switch]$Unattended)
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
    # install.ps1's own hints are replaced by the friendlier steps below.
    & $Installer -Python $Python -Weights $Weights -Device cpu -ValidateOnly:$ValidateOnly -SkipRegistration:$Unattended 6>$null
    if ($ValidateOnly) { return }

    # Last step: add the extension to the browser. Chrome only allows this by
    # hand, so make it as short as possible: the folder path goes on the
    # clipboard and the extensions page is opened for the user.
    $ExtensionFolder = Join-Path $ProjectRoot 'extension'
    $Browsers = foreach ($Candidate in @(
        @{Name = 'Chrome'; Exe = 'chrome.exe'; Page = 'chrome://extensions/'},
        @{Name = 'Edge'; Exe = 'msedge.exe'; Page = 'edge://extensions/'})) {
        foreach ($Hive in @('HKCU:', 'HKLM:')) {
            $Key = "$Hive\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\$($Candidate.Exe)"
            $Path = (Get-ItemProperty -LiteralPath $Key -ErrorAction SilentlyContinue).'(default)'
            if ($Path -and (Test-Path -LiteralPath $Path)) { $Candidate.Path = $Path; $Candidate; break }
        }
    }
    $Browser = @($Browsers)[0]
    $PageName = if ($Browser) { $Browser.Page } else { 'chrome://extensions (or edge://extensions)' }
    $Copied = $false
    if (-not $Unattended) { try { Set-Clipboard -Value $ExtensionFolder; $Copied = $true } catch {} }

    Write-Host ''
    Write-Host 'Setup complete!' -ForegroundColor Green
    Write-Host ''
    Write-Host 'One last step: add the extension to your browser.' -ForegroundColor Cyan
    Write-Host "  1. On the $PageName page, turn on 'Developer mode' (switch at the top right)."
    Write-Host "  2. Click 'Load unpacked' (top left)."
    if ($Copied) {
        Write-Host '  3. In the folder window, click the address bar at the top, press Ctrl+V'
        Write-Host "     and Enter, then click 'Select Folder'. (The folder is already copied.)"
    } else {
        Write-Host "  3. Choose this folder and click 'Select Folder':"
    }
    Write-Host "     $ExtensionFolder" -ForegroundColor Yellow
    Write-Host '  4. Open https://lichess.org/analysis, click the book button under the board'
    Write-Host "     and choose the 'Marvin' tab."
    Write-Host ''
    Write-Host 'Keep this folder where it is. If you move it, run Setup.cmd again.'
    Write-Host ''
    if ($Browser -and -not $Unattended) {
        Read-Host "Press Enter to open $($Browser.Name)'s extensions page" | Out-Null
        try { Start-Process -FilePath $Browser.Path -ArgumentList $Browser.Page }
        catch { Write-Host "Could not open $($Browser.Name). Open $($Browser.Page) yourself." }
    }
} catch {
    Write-Host ''
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
