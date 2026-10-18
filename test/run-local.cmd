@echo off
chcp 65001 >nul
pushd "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Python 환경이 없습니다. README의 설치 명령을 먼저 실행하세요.
  pause
  popd
  exit /b 1
)
".venv\Scripts\python.exe" "test\local.py" %*
if errorlevel 1 pause
popd
