@echo off
REM Fix adapter_config.json to remove unsupported parameters
REM This script can be run from anywhere

echo ============================================================
echo Fixing adapter_config.json
echo ============================================================
echo.

REM Change to project root directory
cd /d "%~dp0.."

REM Check if fix script exists
if not exist "fix_adapter_config_anywhere.py" (
    echo [ERROR] fix_adapter_config_anywhere.py not found!
    echo Please make sure you're in the project root directory.
    pause
    exit /b 1
)

REM Check if model directory exists
if not exist "models\final" (
    echo [ERROR] models\final directory not found!
    echo Please make sure the model directory exists.
    pause
    exit /b 1
)

echo Running fix script...
echo.

python fix_adapter_config_anywhere.py

if errorlevel 1 (
    echo.
    echo [ERROR] Fix failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Fix complete! You can now run training.
echo ============================================================
pause
