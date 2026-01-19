@echo off
REM Clear tokenization cache
REM This forces regeneration of tokenized datasets

echo ============================================================
echo Clearing Tokenization Cache
echo ============================================================

cd /d "%~dp0.."

set CACHE_DIR=data\cache

if exist "%CACHE_DIR%\train_tokenized" (
    echo Removing train_tokenized cache...
    rmdir /s /q "%CACHE_DIR%\train_tokenized"
    echo ✅ Train cache cleared
) else (
    echo ℹ️  No train cache found
)

if exist "%CACHE_DIR%\val_tokenized" (
    echo Removing val_tokenized cache...
    rmdir /s /q "%CACHE_DIR%\val_tokenized"
    echo ✅ Validation cache cleared
) else (
    echo ℹ️  No validation cache found
)

if exist "%CACHE_DIR%\cache_marker.txt" (
    del "%CACHE_DIR%\cache_marker.txt"
    echo ✅ Cache marker removed
)

echo.
echo ============================================================
echo Cache cleared successfully!
echo Next training will regenerate tokenized datasets
echo ============================================================
pause
