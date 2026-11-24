@echo off
title GateSentinel - Live GUI Demo
cd /d "%~dp0"
echo ======================================================
echo    GateSentinel ALPR - Live Camera Detection
echo ======================================================
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo TIP: Close any apps using your camera first (Teams, Zoom, browser tabs)
echo Camera: External USB (index 0)
echo Mode: SINGLE-STAGE - plate detector runs on full frame
echo       (no vehicle needed - point camera at any plate image)
echo.
echo Controls in the video window:
echo   Q / Esc  = Quit
echo   M        = Toggle single-stage / two-stage mode
echo   S        = Save snapshot
echo   P        = Pause
echo.
python scripts\demo_live.py --source 0 --single-stage
pause
