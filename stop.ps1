# stop.ps1
# Cleanly stops the backend server, MediaMTX, and Docker containers

Write-Host "[+] Stopping Vision Assistant services..." -ForegroundColor Cyan

# Stop Python and MediaMTX background processes if running
Stop-Process -Name "python", "mediamtx" -Force -ErrorAction SilentlyContinue
Write-Host "    [OK] Stopped local Python and MediaMTX processes." -ForegroundColor Green

# Stop Docker containers
docker compose down
Write-Host "    [OK] Stopped Docker containers." -ForegroundColor Green

Write-Host "[+] All systems shut down cleanly." -ForegroundColor Green