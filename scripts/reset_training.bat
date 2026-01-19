@echo off
REM Reset training environment - Delete all trained models and checkpoints
REM This allows training from scratch

echo ============================================================
echo RESET TRAINING ENVIRONMENT
echo ============================================================
echo.
echo WARNING: This will delete all trained models and checkpoints!
echo This will allow you to train from scratch.
echo.
echo Files that will be deleted:
echo   - models\final\  (Trained model)
echo   - models\checkpoints\  (All training checkpoints)
echo   - data\cache\  (Tokenized data cache)
echo.

set /p confirm="Are you sure? Type YES to continue: "
if not "%confirm%"=="YES" (
    echo.
    echo Cancelled. No files were deleted.
    pause
    exit /b 0
)

echo.
echo ============================================================
echo Deleting files...
echo ============================================================
echo.

REM Delete models/final
if exist "models\final" (
    echo Deleting models\final\...
    rmdir /s /q "models\final"
    if exist "models\final" (
        echo [ERROR] Failed to delete models\final\
    ) else (
        echo [OK] Deleted models\final\
    )
) else (
    echo [INFO] models\final\ does not exist (already deleted)
)

echo.

REM Delete models/checkpoints
if exist "models\checkpoints" (
    echo Deleting models\checkpoints\...
    rmdir /s /q "models\checkpoints"
    if exist "models\checkpoints" (
        echo [ERROR] Failed to delete models\checkpoints\
    ) else (
        echo [OK] Deleted models\checkpoints\
    )
) else (
    echo [INFO] models\checkpoints\ does not exist (already deleted)
)

echo.

REM Delete data/cache
if exist "data\cache" (
    echo Deleting data\cache\...
    rmdir /s /q "data\cache"
    if exist "data\cache" (
        echo [ERROR] Failed to delete data\cache\
    ) else (
        echo [OK] Deleted data\cache\
    )
) else (
    echo [INFO] data\cache\ does not exist (already deleted)
)

echo.
echo ============================================================
echo Reset Complete!
echo ============================================================
echo.
echo You can now start training from scratch using:
echo   train_with_book1.bat
echo   OR
echo   scripts\train_with_book1.bat
echo.
pause
