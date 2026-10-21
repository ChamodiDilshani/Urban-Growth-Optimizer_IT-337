@echo off
title Urbanscapes-LK Multi-Tier Starter
echo ========================================================
echo   Starting Urbanscapes-LK MERN + Python Application
echo ========================================================

echo [1/3] Starting Python Model Engine on Port 5001...
start "Python Model Engine" cmd /k "python python_service/main.py"

timeout /t 3 /nobreak >nul

echo [2/3] Starting Node Express Backend on Port 5000...
start "Express Backend" cmd /k "node server/server.js"

timeout /t 2 /nobreak >nul

echo [3/3] Starting React Frontend on Port 5173...
cd client
start "React Frontend" cmd /k "npm run dev"

timeout /t 2 /nobreak >nul
start http://localhost:5173

echo All services launched!
