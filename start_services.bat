@echo off
title Urbanscapes-LK Full Stack Launcher
echo ===================================================
echo Starting Urbanscapes-LK Full Stack Application
echo ===================================================

echo [1/3] Starting Python Model Engine on port 5001...
start "Python Model Engine (Port 5001)" cmd /k "python python_service/main.py"

echo [2/3] Starting Express Backend on port 5000...
start "Express Backend (Port 5000)" cmd /k "cd backend && npm start"

echo [3/3] Starting React Frontend on port 5173...
start "React Frontend (Port 5173)" cmd /k "cd frontend && npm run dev"

echo ===================================================
echo All services launched in their respective windows!
echo Frontend: http://localhost:5173
echo Backend:  http://localhost:5000
echo Python:   http://localhost:5001
echo ===================================================
