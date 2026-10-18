@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "MODE=%~1"
if "%~2" neq "" (
    echo Usage: setup_gpu.bat [auto^|legacy^|modern]
    exit /b 2
)
if not defined MODE set "MODE=auto"

if /i "%MODE%"=="auto" (
    set "CC_MAJOR="
    set "CC_MINOR="
    for /f "tokens=1,2 delims=." %%A in ('nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader 2^>nul') do (
        set "CC_MAJOR=%%A"
        set "CC_MINOR=%%B"
    )
    if not defined CC_MAJOR (
        echo Could not detect GPU capability. Specify legacy or modern.
        exit /b 1
    )
    set "MODE=modern"
    if !CC_MAJOR! LSS 7 set "MODE=legacy"
    if !CC_MAJOR! EQU 7 if !CC_MINOR! LSS 5 set "MODE=legacy"
)

if /i not "%MODE%"=="legacy" if /i not "%MODE%"=="modern" (
    echo Usage: setup_gpu.bat [auto^|legacy^|modern]
    exit /b 2
)

py -3.11 -m venv .venv-gpu
if errorlevel 1 (
    echo Python 3.11 is required. Install it with the Python Launcher.
    exit /b 1
)
set "PY=.venv-gpu\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

if /i "%MODE%"=="legacy" (
    "%PY%" -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
) else (
    "%PY%" -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
)
if errorlevel 1 exit /b 1

"%PY%" -m pip install -r ..\requirements.txt
if errorlevel 1 exit /b 1

"%PY%" -m pip install -r requirements-gpu.txt
if errorlevel 1 exit /b 1
echo GPU environment ready. Start it with run_gpu.bat
