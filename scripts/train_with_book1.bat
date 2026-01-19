@echo off
REM ============================================================
REM Train on combined MPC data + book1 data
REM REQUIRES: NVIDIA GPU with CUDA support
REM ============================================================

REM Keep window open on error and enable delayed expansion
setlocal enabledelayedexpansion

REM Add pause at start for debugging (optional - remove after testing)
REM pause

echo.
echo ============================================================
echo TRAINING SCRIPT - MPC + Book1 Data
echo ============================================================
echo.
echo This script will train on combined data (108,611 samples)
echo.
echo Press Ctrl+C to cancel, or wait 3 seconds to continue...
timeout /t 3 /nobreak >nul

REM Get script directory and change to project root
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Failed to change to project directory!
    echo Script directory: %SCRIPT_DIR%
    echo Expected parent: %SCRIPT_DIR%..
    echo Current directory: %CD%
    echo ============================================================
    echo.
    pause
    exit /b 1
)

echo [OK] Changed to project directory: %CD%
echo.

REM Verify Python is available
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python and add it to PATH.
    echo ============================================================
    echo.
    pause
    exit /b 1
)

python --version
echo [OK] Python found
echo.

REM Check GPU availability
echo Checking GPU availability...
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: CUDA is not available!
    echo GPU is required for training.
    echo.
    echo Please check:
    echo   1. NVIDIA GPU drivers are installed
    echo   2. CUDA is installed
    echo   3. PyTorch with CUDA support is installed
    echo.
    echo Run: python check_gpu.py
    echo ============================================================
    echo.
    pause
    exit /b 1
)

echo [OK] GPU available!
echo.

REM Set file paths (relative to project root)
set "TRAIN_FILE=data\processed\mpc_cleaned_combined_train_with_book1.json"
set "VAL_FILE=data\processed\mpc_cleaned_combined_val.json"

echo Checking training files...
echo.

REM Check training file
if not exist "%TRAIN_FILE%" (
    echo ============================================================
    echo ERROR: Training file not found!
    echo.
    echo Expected: %TRAIN_FILE%
    echo Current directory: %CD%
    echo.
    echo Full path: %CD%\%TRAIN_FILE%
    echo ============================================================
    echo.
    pause
    exit /b 1
)
echo [OK] Training file found: %TRAIN_FILE%

REM Check validation file
if not exist "%VAL_FILE%" (
    echo ============================================================
    echo ERROR: Validation file not found!
    echo.
    echo Expected: %VAL_FILE%
    echo Current directory: %CD%
    echo.
    echo Full path: %CD%\%VAL_FILE%
    echo ============================================================
    echo.
    pause
    exit /b 1
)
echo [OK] Validation file found: %VAL_FILE%
echo.

REM Check if trainer.py exists
set "TRAINER_SCRIPT=src\training\trainer.py"
if not exist "%TRAINER_SCRIPT%" (
    echo ============================================================
    echo ERROR: Trainer script not found!
    echo.
    echo Expected: %TRAINER_SCRIPT%
    echo Current directory: %CD%
    echo ============================================================
    echo.
    pause
    exit /b 1
)
echo [OK] Trainer script found: %TRAINER_SCRIPT%
echo.

REM Display summary
echo ============================================================
echo TRAINING CONFIGURATION
echo ============================================================
echo Train file: %TRAIN_FILE%
echo Validation file: %VAL_FILE%
echo Trainer script: %TRAINER_SCRIPT%
echo Project directory: %CD%
echo.
echo Training on: 108,611 samples (MPC + Book1)
echo Validation on: 13,576 samples
echo.
echo GPU-OPTIMIZED with fp16 Mixed Precision
echo ============================================================
echo.
echo Starting training in 2 seconds...
timeout /t 2 /nobreak >nul
echo.

REM Run training
python "%TRAINER_SCRIPT%" "%TRAIN_FILE%" "%VAL_FILE%"

set "TRAIN_EXIT_CODE=!errorlevel!"

if !TRAIN_EXIT_CODE! neq 0 (
    echo.
    echo ============================================================
    echo ERROR: Training failed!
    echo.
    echo Exit code: !TRAIN_EXIT_CODE!
    echo.
    echo Please check the error messages above for details.
    echo.
    echo Common issues:
    echo   - Out of GPU memory: Reduce batch size
    echo   - File not found: Check file paths
    echo   - CUDA error: Check GPU drivers
    echo ============================================================
    echo.
    pause
    exit /b !TRAIN_EXIT_CODE!
)

echo.
echo ============================================================
echo TRAINING COMPLETED SUCCESSFULLY!
echo ============================================================
echo.
echo Model saved to: models\final
echo.
echo You can now use the trained model in the API:
echo   scripts\start_api.bat
echo.
echo Or test the model:
echo   python test_model_part1.py
echo.
echo ============================================================
echo.
pause
exit /b 0