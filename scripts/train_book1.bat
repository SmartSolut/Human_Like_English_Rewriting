@echo off
REM Train on Book1 data (AI to Human paraphrasing)
REM REQUIRES: NVIDIA GPU with CUDA support

REM Change to project root directory
cd /d "%~dp0.."

echo ============================================================
echo Training on Book1 Data (AI to Human Paraphrasing)
echo GPU-OPTIMIZED with fp16 Mixed Precision
echo ============================================================

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

set TRAIN_FILE=data/processed/book1_train.json
set VAL_FILE=data/processed/book1_val.json

if not exist "%TRAIN_FILE%" (
    echo Error: Training file not found: %TRAIN_FILE%
    pause
    exit /b 1
)

if not exist "%VAL_FILE%" (
    echo Error: Validation file not found: %VAL_FILE%
    pause
    exit /b 1
)

echo Training on: %TRAIN_FILE%
echo Validation: %VAL_FILE%
echo.
echo Note: This will continue training from models\final
echo.

python src/training/trainer.py "%TRAIN_FILE%" "%VAL_FILE%"

if errorlevel 1 (
    echo Error in Book1 training!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Book1 training completed!
echo ============================================================
pause

