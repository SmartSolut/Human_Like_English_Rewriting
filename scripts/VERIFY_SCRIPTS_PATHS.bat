@echo off
REM ============================================================
REM VERIFY SCRIPTS PATHS - Check all scripts use correct paths
REM ============================================================

cd /d "%~dp0.."

echo ============================================================
echo VERIFYING SCRIPTS PATHS
echo ============================================================
echo.
echo Checking if scripts point to CLEANED data files...
echo.

set ERRORS=0

REM Check train_part_X.bat scripts
echo [1] Checking train_part_X.bat scripts...
echo ============================================================
for /L %%i in (1,1,5) do (
    echo.
    echo Checking train_part_%%i.bat:
    findstr /C:"SPLITS_DIR" "scripts\train_part_%%i.bat" | findstr /V "REM" >nul
    if errorlevel 1 (
        echo   [ERROR] SPLITS_DIR not found!
        set /a ERRORS+=1
    ) else (
        findstr /C:"SPLITS_DIR" "scripts\train_part_%%i.bat" | findstr /C:"splits_5_parts_cleaned" >nul
        if errorlevel 1 (
            echo   [ERROR] Does NOT use splits_5_parts_cleaned!
            echo   Current path:
            findstr /C:"SPLITS_DIR" "scripts\train_part_%%i.bat" | findstr /V "REM"
            set /a ERRORS+=1
        ) else (
            echo   [OK] Uses splits_5_parts_cleaned
            findstr /C:"SPLITS_DIR" "scripts\train_part_%%i.bat" | findstr /V "REM"
        )
    )
    
    findstr /C:"TRAIN_FILE" "scripts\train_part_%%i.bat" | findstr /C:"_cleaned.json" >nul
    if errorlevel 1 (
        echo   [WARNING] TRAIN_FILE may not use _cleaned.json
        findstr /C:"TRAIN_FILE" "scripts\train_part_%%i.bat" | findstr /V "REM"
    ) else (
        echo   [OK] Uses _cleaned.json file
    )
)

echo.
echo ============================================================
echo [2] Checking main training scripts...
echo ============================================================
echo.

echo train_with_book1.bat:
findstr /C:"TRAIN_FILE" "scripts\train_with_book1.bat" | findstr /C:"mpc_cleaned" >nul
if errorlevel 1 (
    echo   [ERROR] Does NOT use mpc_cleaned file!
    set /a ERRORS+=1
) else (
    echo   [OK] Uses mpc_cleaned file
    findstr /C:"TRAIN_FILE" "scripts\train_with_book1.bat" | findstr /V "REM"
)

echo.
echo train_full_data.bat:
findstr /C:"TRAIN_FILE" "scripts\train_full_data.bat" | findstr /C:"mpc_cleaned" >nul
if errorlevel 1 (
    echo   [ERROR] Does NOT use mpc_cleaned file!
    set /a ERRORS+=1
) else (
    echo   [OK] Uses mpc_cleaned file
    findstr /C:"TRAIN_FILE" "scripts\train_full_data.bat" | findstr /V "REM"
)

echo.
echo ============================================================
echo [3] Checking prepare_clean_data.bat...
echo ============================================================
echo.

findstr /C:"python" "scripts\prepare_clean_data.bat" | findstr /C:"scripts\\utils\\clean_all_data_parts.py" >nul
if errorlevel 1 (
    echo   [ERROR] Python path is incorrect!
    echo   Current path:
    findstr /C:"python" "scripts\prepare_clean_data.bat"
    set /a ERRORS+=1
) else (
    echo   [OK] Python path is correct
    findstr /C:"python" "scripts\prepare_clean_data.bat"
)

echo.
echo ============================================================
echo VERIFICATION SUMMARY
echo ============================================================
echo.

if %ERRORS% EQU 0 (
    echo ✅ ALL SCRIPTS ARE CORRECTLY CONFIGURED
    echo    All scripts point to CLEANED data files
) else (
    echo ❌ FOUND %ERRORS% ERROR(S)
    echo    Some scripts need to be fixed
)

echo.
echo ============================================================
pause
