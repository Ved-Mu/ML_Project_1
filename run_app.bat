@echo off
rem Starts the Engine Health Monitor. Double-click this file; the app opens in your browser
rem at http://localhost:8501 after a few seconds.
rem Keep this window open while you use the app. Close it (or press Ctrl+C) to stop the app.
cd /d "%~dp0improved_implementation"
rem "headless" skips Streamlit's first-run email question, so open the browser ourselves.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 6; Start-Process 'http://localhost:8501'"
python -m streamlit run app/streamlit_app.py --server.headless true --server.port 8501
pause
