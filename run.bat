@echo off
echo ==============================================
echo Starting Customer Churn Prediction System...
echo ==============================================
call venv\Scripts\activate.bat
streamlit run app.py
pause
