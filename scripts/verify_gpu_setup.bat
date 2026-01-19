@echo off
echo ============================================================
echo Verifying GPU Setup
echo ============================================================
echo.

echo 1. Checking NVIDIA GPU...
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if errorlevel 1 (
    echo [ERROR] nvidia-smi failed!
    pause
    exit /b 1
)

echo.
echo 2. Checking PyTorch CUDA support...
python scripts\utils\check_gpu.py

echo.
pause


