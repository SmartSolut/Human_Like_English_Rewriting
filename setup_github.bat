@echo off
echo ========================================
echo Setup GitHub Repository
echo ========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed!
    echo Please install Git from https://git-scm.com/
    pause
    exit /b 1
)

echo Step 1: Initializing Git repository...
if exist .git (
    echo Git repository already exists.
) else (
    git init
    echo Git repository initialized.
)

echo.
echo Step 2: Adding all files...
git add .

echo.
echo Step 3: Creating initial commit...
git commit -m "Initial commit: Human-Like English Rewriting System"

echo.
echo Step 4: Setting up remote repository...
echo Repository: https://github.com/SmartSolut/Human_Like_English_Rewriting.git
git remote remove origin 2>nul
git remote add origin https://github.com/SmartSolut/Human_Like_English_Rewriting.git

echo.
echo Step 5: Setting branch to main...
git branch -M main

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To push to GitHub, run:
echo   git push -u origin main
echo.
echo You will be prompted for credentials:
echo   Username: SmartSolut
echo   Password: 123smartsoulation123??
echo.
pause
