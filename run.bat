@echo off
setlocal

if not exist venv (
    echo [ERROR] No virtual environment found in this folder.
    echo Please run install.bat first.
    echo Press any key to continue...
    pause >nul
    exit /b 1
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate the virtual environment.
    echo Press any key to continue...
    pause >nul
    exit /b 1
)

echo Starting sneakyReport (TM) — Visual Dataset Intelligence...
echo (The first launch can take 1-2 minutes - that's normal.)
echo.

streamlit run app.py

echo Press any key to continue...
pause >nul
