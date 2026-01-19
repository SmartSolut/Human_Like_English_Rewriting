@echo off
echo ============================================================
echo Cleaning Data for Training (IN PLACE)
echo ============================================================
echo.
echo Files will be cleaned directly in splits_5_parts
echo No backup copies will be created
echo.

cd /d "%~dp0.."

echo Cleaning all data parts (1-5)...
python scripts\utils\clean_all_data_parts.py
if errorlevel 1 (
    echo Error: Failed to clean data!
    pause
    exit /b 1
)

echo.
echo Data cleaned successfully!
echo Training will use the cleaned files directly.
echo.
pause

