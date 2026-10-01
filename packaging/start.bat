@echo off
rem TrainWatch launcher: runs the bundled Python, nothing to install.
rem (ASCII only on purpose: cmd.exe reads .bat files in the system code page)
cd /d "%~dp0"
set "TRAINWATCH_HOME=%~dp0"
if not exist "%~dp0runtime\pythonw.exe" (
  echo runtime\pythonw.exe not found.
  echo Please extract the whole folder, or check whether antivirus quarantined it.
  pause
  exit /b 1
)
start "" "%~dp0runtime\pythonw.exe" "%~dp0app\main.py"
