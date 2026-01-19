@echo off
echo ============================================================
echo Backing up current model (128 tokens)
echo ============================================================
echo.

if not exist "models\backup_128_tokens" (
    mkdir "models\backup_128_tokens"
    echo Created backup directory: models\backup_128_tokens
)

if exist "models\final" (
    echo Copying models\final to models\backup_128_tokens...
    xcopy "models\final" "models\backup_128_tokens\final" /E /I /Y
    echo.
    echo Backup completed!
    echo Old model saved to: models\backup_128_tokens\final
) else (
    echo Warning: models\final not found!
    echo No model to backup.
)

echo.
pause

