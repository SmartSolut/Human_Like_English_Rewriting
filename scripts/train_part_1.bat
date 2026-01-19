@echo off
REM Train on Part 1 of 5 (cleaned MPC data)
REM REQUIRES: NVIDIA GPU with CUDA support

REM Change to project root directory
cd /d "%~dp0.."

echo ============================================================
echo Training on Part 1/5 (Cleaned MPC Data - ~21,722 samples)
echo GPU-OPTIMIZED with fp16 Mixed Precision
echo ============================================================
echo.
echo NOTE: This script uses CLEANED version if available
echo       Falls back to non-cleaned version if cleaned not found
echo.

REM Check GPU availability
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    echo ERROR: CUDA is not available! GPU is required for training.
    echo Please check your CUDA installation and GPU drivers.
    pause
    exit /b 1
)

echo GPU check passed. Starting training...
echo.

set SPLITS_DIR=data/processed/splits_5_parts_cleaned
set VAL_FILE=data/processed/mpc_cleaned_combined_val.json
set TRAIN_FILE=%SPLITS_DIR%\train_part_1_cleaned.json

REM Check if cleaned version exists, fallback to non-cleaned if not
if not exist "%TRAIN_FILE%" (
    echo Warning: Cleaned version not found, using non-cleaned version...
    set SPLITS_DIR=data/processed/splits_5_parts
    set TRAIN_FILE=%SPLITS_DIR%\train_part_1.json
)

if not exist "%TRAIN_FILE%" (
    echo Error: Training file not found: %TRAIN_FILE%
    pause
    exit /b 1
)

echo Training on: %TRAIN_FILE%
echo Validation: %VAL_FILE%
echo.

python src/training/trainer.py "%TRAIN_FILE%" "%VAL_FILE%"

if errorlevel 1 (
    echo Error in Part 1 training!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Part 1/5 training completed!
echo ============================================================
pause


