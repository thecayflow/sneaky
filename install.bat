@echo off
setlocal

echo ============================================
echo   sneaky (TM) Semantic Report - Installer
echo ============================================
echo.

REM --- Check Python is available -----------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on your PATH.
    echo Please install Python 3.11 from https://www.python.org/downloads/
    echo and make sure to check "Add python.exe to PATH" during setup.
    echo Then run this installer again.
    echo Press any key to continue...
    pause >nul
    exit /b 1
)

echo Found Python:
python --version
echo.

REM --- Create the virtual environment if it doesn't exist yet ------------
if exist venv (
    echo A "venv" folder already exists - skipping its creation.
    echo (Delete the "venv" folder first if you want a completely clean install.)
) else (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment. See the messages above.
        echo Press any key to continue...
        pause >nul
        exit /b 1
    )
)
echo.

REM --- Activate it ----------------------------------------------------------
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Could not activate the virtual environment.
    echo Press any key to continue...
    pause >nul
    exit /b 1
)

REM --- Install PyTorch with CUDA support first (needs a special index) ---
echo Installing PyTorch with CUDA support...
echo (This is a large download - several GB - please be patient.)
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed. See the messages above.
    echo Press any key to continue...
    pause >nul
    exit /b 1
)
echo.

REM --- Install the rest of the dependencies -------------------------------
echo Installing the remaining dependencies from requirements.txt...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. See the messages above.
    echo Press any key to continue...
    pause >nul
    exit /b 1
)
echo.

REM --- Download the spaCy language model ----------------------------------
echo Downloading the spaCy language model...
echo.
python -m spacy download en_core_web_sm
if errorlevel 1 (
    echo [ERROR] spaCy language model download failed. See the messages above.
    echo Press any key to continue...
    pause >nul
    exit /b 1
)
echo.

REM --- Final check: verify everything actually loads ----------------------
echo ============================================
echo   Running installation check...
echo ============================================
echo.
python check_install.py
if errorlevel 1 (
    echo [ERROR] Installation check failed - see the messages above.
    echo Press any key to continue...
    pause >nul
    exit /b 1
)

echo.
echo ============================================
echo   Setup verified - use run.bat to start the app.
echo ============================================
echo Press any key to continue...
pause >nul
