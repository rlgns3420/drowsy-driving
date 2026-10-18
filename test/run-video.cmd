@echo off
setlocal
set "ROOT=%~dp0.."
if not exist "%ROOT%\.venv\Scripts\python.exe" (
  echo Python environment missing. Install requirements using README.md first.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Drag a recorded video file onto this launcher.
  echo Or run: run-video.cmd "C:\videos\driver.mp4"
  pause
  exit /b 1
)
"%ROOT%\.venv\Scripts\python.exe" "%~dp0video.py" --video "%~1" --preview --realtime --save-video
set "RESULT=%ERRORLEVEL%"
pause
exit /b %RESULT%
