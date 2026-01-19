@echo off
echo Checking for NVIDIA GPU...
echo.

REM Check if nvidia-smi exists
where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] nvidia-smi not found!
    echo This means NVIDIA drivers are not installed or GPU is not available.
    pause
    exit /b 1
)

echo Running nvidia-smi...
echo.
nvidia-smi

pause


