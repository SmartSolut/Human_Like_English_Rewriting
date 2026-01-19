@echo off
REM ============================================================
REM START TRAINING - Use cmd /k to keep window open ALWAYS
REM Double-click this file to run training
REM ============================================================

echo.
echo ============================================================
echo Starting Training...
echo ============================================================
echo.
echo Note: Window will stay open even if there's an error
echo.

REM Change to scripts directory
cd /d "%~dp0"

REM Run training script with cmd /k to keep window open
cmd /k "train_with_book1.bat"

REM Should never reach here, but just in case
pause
