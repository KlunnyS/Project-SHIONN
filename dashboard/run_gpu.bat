@echo off
setlocal
cd /d "%~dp0"

set "PY=.venv-gpu\Scripts\python.exe"
if not exist "%PY%" (
    echo GPU environment not found. Run setup_gpu.bat first.
    exit /b 1
)

"%PY%" scripts\check_gpu.py
if errorlevel 1 exit /b 1
"%PY%" app.py
