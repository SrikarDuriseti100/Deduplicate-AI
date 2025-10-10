@echo off
echo Starting Article Deduplication System with Gradio...
echo =====================================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Python found!

REM Install requirements if needed
echo Installing/updating requirements...
pip install -r requirements.txt

REM Create necessary directories
if not exist "data" mkdir data
if not exist "logs" mkdir logs

echo Launching Gradio application...
echo.
echo The application will open in your web browser at http://localhost:7860
echo Press Ctrl+C to stop the application.
echo.

python gradio_app.py