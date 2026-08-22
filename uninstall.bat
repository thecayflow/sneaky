@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   sneakyReport (TM) — Visual Dataset Intelligence - Uninstaller
echo ============================================
echo.
echo Before continuing, make sure the app is NOT currently running
echo (close its browser tab and the terminal window it's running in).
echo.
echo Press any key to continue...
pause >nul
echo.

REM --- Remove the virtual environment -------------------------------------
if exist venv (
    echo Removing the virtual environment ^(venv folder^)...
    rmdir /s /q venv
    if exist venv (
        echo [WARNING] Could not fully remove the venv folder.
        echo It may still be in use - close the app and try again,
        echo or delete the "venv" folder by hand in File Explorer.
    ) else (
        echo Done - the venv folder has been removed.
    )
) else (
    echo No venv folder found - nothing to remove here.
)
echo.

REM --- Optionally remove the dataset analysis cache -----------------------
if exist cache (
    set /p REMOVE_CACHE="Also remove the 'cache' folder (dataset analysis cache)? [y/N] "
    if /i "!REMOVE_CACHE!"=="y" (
        rmdir /s /q cache
        echo Cache folder removed.
    ) else (
        echo Keeping the cache folder.
    )
    echo.
)

REM --- What's left, and what a script shouldn't do automatically ---------
echo ============================================
echo   Almost done - two things to finish by hand:
echo ============================================
echo.
echo 1. This project folder itself is still here. To remove it
echo    completely, close this window and delete the folder in
echo    File Explorer:
echo      %CD%
echo.
echo 2. Downloaded AI model weights (a few GB) are cached OUTSIDE
echo    this folder entirely, at:
echo      %USERPROFILE%\.cache\huggingface
echo    That cache may be shared with OTHER AI/ML tools on your
echo    computer (any other project using Hugging Face models),
echo    so it is NOT removed automatically here.
echo.
echo    Not sure if you should delete it? It's just a cache, not
echo    something irreplaceable: if you delete it and another tool
echo    needs those same model files again later, it will simply
echo    re-download them automatically - nothing breaks permanently
echo    either way. If in doubt, it's safe to just leave it alone.
echo.
echo Press any key to continue...
pause >nul
