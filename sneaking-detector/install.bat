@echo off
REM Quick-install script for Sneaking Detector (Windows)

cd /d "%~dp0"
echo === Sneaking Detector installer ===

python --version >nul 2>&1 || (
    echo ERROR: Python not found. Install Python 3.9+ from https://python.org
    pause & exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt

echo.
echo === Installation complete ===
echo.
echo To run the app:
echo   .venv\Scripts\activate.bat
echo   python run.py
echo.
echo For faster detection, also install YOLOv8:
echo   pip install ultralytics
pause
