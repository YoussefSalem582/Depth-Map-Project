@echo off
echo 🎯 Depth Map Project - Flask Web Application
echo ================================================
echo Starting Flask server...
echo.

REM Set PYTHONPATH
set PYTHONPATH=%~dp0src

REM Run the Flask app
python app.py

pause 