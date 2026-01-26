@echo off
title FPL Assistant
cd /d "%~dp0"
echo Starting FPL Assistant...
echo.
call .venv\Scripts\activate 2>nul || call venv\Scripts\activate 2>nul || echo No venv found, using system Python
streamlit run src/fpl_assistant/app.py
pause
