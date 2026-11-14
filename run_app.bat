@echo off
rem Starts the Engine Health Monitor. Double-click this file; the app opens in your browser.
rem Keep this window open while you use the app. Close it (or press Ctrl+C) to stop the app.
cd /d "%~dp0improved_implementation"
python -m streamlit run app/streamlit_app.py
pause
