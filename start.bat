@echo off
title WHSPR - Starting Servers

echo ==========================================
echo  WHSPR - Starting MySQL...
echo ==========================================

net start MySQL80 2>nul
if %errorlevel% neq 0 (
    net start MySQL 2>nul
    if %errorlevel% neq 0 (
        echo MySQL already running or service name differs. Continuing...
    )
)

echo.
echo ==========================================
echo  WHSPR - Installing dependencies...
echo ==========================================

echo.
echo [1/2] Installing backend dependencies...
cd /d "%~dp0whspr-be"
pip install -r requirements.txt --quiet

echo.
echo [2/2] Installing frontend dependencies...
cd /d "%~dp0whspr-fe"
npm install --silent

echo.
echo ==========================================
echo  Starting servers...
echo ==========================================

echo.
echo Starting backend on http://localhost:8000
start "WHSPR Backend" cmd /k "cd /d "%~dp0whspr-be" && uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo Starting frontend on http://localhost:3000
start "WHSPR Frontend" cmd /k "cd /d "%~dp0whspr-fe" && npm run dev"

echo.
echo Both servers are running.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Waiting for frontend to start...
timeout /t 5 /nobreak >nul
start chrome http://localhost:3000
echo.
pause
