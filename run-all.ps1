# Urbanscapes-LK Multi-Tier Starter Script
# Starts:
# 1. Python FastAPI Model Engine (Port 5001)
# 2. Express.js Backend Server (Port 5000)
# 3. React Vite Frontend (Port 5173)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  🏙️ Starting Urbanscapes-LK MERN + Python Application   " -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Start Python Service
Write-Host "[1/3] Launching Python FastAPI Model Engine (Port 5001)..." -ForegroundColor Green
$pythonProc = Start-Process python -ArgumentList "python_service/main.py" -PassThru -NoNewWindow

Start-Sleep -Seconds 3

# 2. Start Express Backend
Write-Host "[2/3] Launching Node.js Express Backend (Port 5000)..." -ForegroundColor Green
$serverProc = Start-Process node -ArgumentList "server/server.js" -PassThru -NoNewWindow

Start-Sleep -Seconds 2

# 3. Start React Client
Write-Host "[3/3] Launching React Vite Frontend (Port 5173)..." -ForegroundColor Green
Write-Host "Access the application in your browser at: http://localhost:5173" -ForegroundColor Yellow
Start-Process "http://localhost:5173"

Set-Location client
npm run dev

# Cleanup when React Vite terminates
if ($pythonProc -and -not $pythonProc.HasExited) { Stop-Process -Id $pythonProc.Id -Force }
if ($serverProc -and -not $serverProc.HasExited) { Stop-Process -Id $serverProc.Id -Force }
