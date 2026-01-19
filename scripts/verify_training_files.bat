@echo off
REM ============================================================
REM Verify Training Files - Check which files exist and are used
REM ============================================================

cd /d "%~dp0.."

echo ============================================================
echo Training Files Verification
echo ============================================================
echo.

echo [1] Checking main training files...
echo.

if exist "data\processed\mpc_cleaned_combined_train.json" (
    echo [OK] mpc_cleaned_combined_train.json
) else (
    echo [MISSING] mpc_cleaned_combined_train.json
)

if exist "data\processed\mpc_cleaned_combined_train_with_book1.json" (
    echo [OK] mpc_cleaned_combined_train_with_book1.json
) else (
    echo [MISSING] mpc_cleaned_combined_train_with_book1.json
)

if exist "data\processed\mpc_cleaned_combined_val.json" (
    echo [OK] mpc_cleaned_combined_val.json
) else (
    echo [MISSING] mpc_cleaned_combined_val.json
)

echo.
echo [2] Checking split files (non-cleaned)...
echo.

for /L %%i in (1,1,5) do (
    if exist "data\processed\splits_5_parts\train_part_%%i.json" (
        echo [OK] splits_5_parts\train_part_%%i.json
    ) else (
        echo [MISSING] splits_5_parts\train_part_%%i.json
    )
)

echo.
echo [3] Checking split files (cleaned)...
echo.

for /L %%i in (1,1,5) do (
    if exist "data\processed\splits_5_parts_cleaned\train_part_%%i_cleaned.json" (
        echo [OK] splits_5_parts_cleaned\train_part_%%i_cleaned.json
    ) else (
        echo [MISSING] splits_5_parts_cleaned\train_part_%%i_cleaned.json
    )
)

echo.
echo [4] Scripts Configuration:
echo.

echo TRAIN_NOW.bat / START_TRAINING.bat:
echo   -> train_with_book1.bat
echo   -> Uses: mpc_cleaned_combined_train_with_book1.json
echo.

echo train_full_data.bat:
echo   -> Uses: mpc_cleaned_combined_train.json
echo.

echo train_part_1.bat to train_part_5.bat:
echo   -> Uses: splits_5_parts_cleaned\train_part_X_cleaned.json (with fallback)
echo.

echo ============================================================
echo Verification Complete
echo ============================================================
pause
