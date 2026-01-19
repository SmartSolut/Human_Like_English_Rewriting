@echo off
echo ============================================================
echo Installing PyTorch with CUDA support
echo ============================================================
echo.

REM Check GPU
echo Checking GPU...
nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>nul
if errorlevel 1 (
    echo [ERROR] nvidia-smi not found! GPU drivers may not be installed.
    pause
    exit /b 1
)

echo.
echo Your GPU: Quadro RTX 3000
echo CUDA Version: 12.8
echo.
echo This will:
echo   1. Uninstall the current CPU-only PyTorch
echo   2. Install PyTorch with CUDA 12.1 support
echo   3. Verify the installation
echo.
pause

echo.
echo ============================================================
echo Step 1: Uninstalling old PyTorch...
echo ============================================================
pip uninstall torch torchvision torchaudio -y
if errorlevel 1 (
    echo [WARNING] Some packages may not have been uninstalled
)

echo.
echo ============================================================
echo Step 2: Installing PyTorch with CUDA 12.1...
echo ============================================================
echo This may take a few minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [ERROR] Installation failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Step 3: Verifying installation...
echo ============================================================
python scripts\utils\check_gpu.py
if errorlevel 1 (
    echo [ERROR] Verification failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Install other dependencies: pip install -r requirements-cuda.txt
echo   2. Run training: scripts\train_part_2.bat
echo.
pause

