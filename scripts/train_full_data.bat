@echo off
REM Train on FULL combined data (all parts together)
REM REQUIRES: NVIDIA GPU with CUDA support

echo.
echo Starting training script...
echo.

REM Change to project root directory
cd /d "%~dp0.."
if errorlevel 1 (
    echo ERROR: Failed to change to project directory!
    echo Current directory: %CD%
    pause
    exit /b 1
)

echo Current directory: %CD%
echo.

echo ============================================================
echo Training on FULL Combined Data (All Parts Together)
echo Train: 108,606 samples | Val: 13,576 samples
echo GPU-OPTIMIZED with fp16 Mixed Precision
echo ============================================================

REM Check GPU availability
echo Checking GPU availability...
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo ============================================================
    echo ERROR: CUDA is not available! GPU is required for training.
    echo Please check your CUDA installation and GPU drivers.
    echo ============================================================
    echo.
    pause
    exit /b 1
)
echo GPU check passed!

echo GPU check passed. Starting training...
echo.

set TRAIN_FILE=data\processed\mpc_cleaned_combined_train.json
set VAL_FILE=data\processed\mpc_cleaned_combined_val.json

echo Checking files...
if not exist "%TRAIN_FILE%" (
    echo.
    echo ============================================================
    echo ERROR: Training file not found!
    echo Expected: %TRAIN_FILE%
    echo Current directory: %CD%
    echo ============================================================
    echo.
    pause
    exit /b 1
)

if not exist "%VAL_FILE%" (
    echo.
    echo ============================================================
    echo ERROR: Validation file not found!
    echo Expected: %VAL_FILE%
    echo Current directory: %CD%
    echo ============================================================
    echo.
    pause
    exit /b 1
)

echo Files found successfully!

echo Training on: %TRAIN_FILE%
echo Validation: %VAL_FILE%
echo.
echo This will train on ALL data at once (better than sequential training)
echo.

echo Starting Python training script...
echo.

python src\training\trainer.py "%TRAIN_FILE%" "%VAL_FILE%"

set TRAIN_EXIT_CODE=%errorlevel%

if %TRAIN_EXIT_CODE% neq 0 (
    echo.
    echo ============================================================
    echo ERROR: Training failed with exit code %TRAIN_EXIT_CODE%
    echo ============================================================
    pause
    exit /b %TRAIN_EXIT_CODE%
)

echo.
echo ============================================================
echo Full data training completed!
echo ============================================================
pause
