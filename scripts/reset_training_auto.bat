@echo off
REM Reset training environment - Delete all trained models and checkpoints
REM This allows training from scratch (AUTOMATIC - NO CONFIRMATION)

echo ============================================================
echo RESET TRAINING ENVIRONMENT (AUTOMATIC)
echo ============================================================
echo.
echo Deleting all trained models and checkpoints...
echo.

REM Change to project root directory
cd /d "%~dp0"
if errorlevel 1 (
    echo ERROR: Failed to change to project directory!
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Delete models/final
if exist "models\final" (
    echo Deleting models\final\...
    rmdir /s /q "models\final"
    if exist "models\final" (
        echo [ERROR] Failed to delete models\final\
        pause
        exit /b 1
    ) else (
        echo [OK] Deleted models\final\
    )
) else (
    echo [INFO] models\final\ does not exist
)

REM Delete models/checkpoints
if exist "models\checkpoints" (
    echo Deleting models\checkpoints\...
    rmdir /s /q "models\checkpoints"
    if exist "models\checkpoints" (
        echo [ERROR] Failed to delete models\checkpoints\
        pause
        exit /b 1
    ) else (
        echo [OK] Deleted models\checkpoints\
    )
) else (
    echo [INFO] models\checkpoints\ does not exist
)

REM Delete data/cache
if exist "data\cache" (
    echo Deleting data\cache\...
    rmdir /s /q "data\cache"
    if exist "data\cache" (
        echo [ERROR] Failed to delete data\cache\
        pause
        exit /b 1
    ) else (
        echo [OK] Deleted data\cache\
    )
) else (
    echo [INFO] data\cache\ does not exist
)

echo.
echo ============================================================
echo Reset Complete!
echo ============================================================
echo.
echo All trained models and checkpoints have been deleted.
echo You can now start training from scratch.
echo.
pause
