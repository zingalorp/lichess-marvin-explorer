@echo off
setlocal
title Marvin Opening Explorer - Uninstall
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -Uninstall
if errorlevel 1 (
  echo.
  echo Uninstall failed. Read the message above.
  pause
  exit /b 1
)
echo.
pause
