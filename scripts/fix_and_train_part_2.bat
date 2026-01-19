@echo off
REM Fix adapter_config.json and then train Part 2
REM This ensures the config is fixed before training

echo ============================================================
echo Step 1: Fixing adapter_config.json
echo ============================================================
echo.

REM Change to project root directory
cd /d "%~dp0.."

REM Run fix script
if exist "fix_adapter_config_anywhere.py" (
    python fix_adapter_config_anywhere.py
    if errorlevel 1 (
        echo [ERROR] Failed to fix adapter_config.json!
        pause
        exit /b 1
    )
) else (
    echo [WARNING] fix_adapter_config_anywhere.py not found
    echo Skipping fix step...
)

echo.
echo ============================================================
echo Step 2: Starting training on Part 2
echo ============================================================
echo.

REM Run training
call "%~dp0train_part_2.bat"
