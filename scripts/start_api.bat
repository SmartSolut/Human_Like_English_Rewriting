@echo off
REM Start FastAPI server for testing the model
REM This will load the trained model from models/final

echo ============================================================
echo Starting Human-Like English Rewriting API
echo ============================================================
echo.
echo API will be available at: http://localhost:8000
echo Test page: http://localhost:8000 (built-in) or test_rewrite.html
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

cd /d "%~dp0.."

REM Check if model exists
if not exist "models\final\adapter_config.json" (
    echo WARNING: Trained model not found at models\final
    echo API will use base model (not fine-tuned)
    echo.
)

python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

pause

