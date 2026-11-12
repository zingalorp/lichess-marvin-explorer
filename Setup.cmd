@echo off
setlocal
title Marvin Opening Explorer - Setup
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 (
  echo.
  echo Setup failed. Read the message above and START-HERE.txt.
  pause
  exit /b 1
)
echo.
pause
