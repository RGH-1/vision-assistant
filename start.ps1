# start.ps1
# One-command startup script for the Vision Assistant local infrastructure & backend

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "       Starting Vision Assistant Environment      " -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

# 1. Automatically detect and display the laptop's LAN IPv4 Address
$lanIp = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceIndex $(Get-NetConnectionProfile | Select-Object -ExpandProperty InterfaceIndex) | Where-Object {$_.IPAddress -match '^(192\.168\.|10\.|172\.(1[6-9]|2[0-9]|3[0-1]))'}).IPAddress | Select-Object -First 1

if (-not $lanIp) {
    # Fallback to standard active IPv4 lookup if primary profile match fails
    $lanIp = (Test-Connection -ComputerName (hostname) -Count 1).IPV4Address.IPAddressToString
}

Write-Host "[*] Detected Laptop LAN IP: " -NoNewline; Write-Host "$lanIp" -ForegroundColor Yellow
Write-Host "[*] Update your Flutter app config or SettingsModel serverIp to: $lanIp" -ForegroundColor DarkGray
Write-Host ""

# 2. Check and start PostgreSQL Docker Container
Write-Host "[+] Checking PostgreSQL Docker container..." -ForegroundColor Cyan
$containerRunning = (docker inspect -f '{{.State.Running}}' vision-assistant-db-1 2>&1)

if ($containerRunning -ne 'true') {
    Write-Host "[+] Starting PostgreSQL Docker container..." -ForegroundColor Cyan
    docker compose -f infra/docker-compose.yml --env-file .env up -d db
    Write-Host "    [OK] PostgreSQL Docker container started successfully." -ForegroundColor Green
} else {
    Write-Host "    [OK] PostgreSQL Docker container is already running." -ForegroundColor Green
}

# 3. Wait a moment for PostgreSQL to warm up
Write-Host "[*] Waiting for PostgreSQL database to warm up..." -ForegroundColor DarkGray
Start-Sleep -Seconds 3

# 4. Start MediaMTX streaming server in a separate background window
Write-Host "[+] Starting MediaMTX server..." -ForegroundColor Cyan
if (Test-Path "infra/mediamtx.exe") {
    Start-Process -FilePath "infra/mediamtx.exe" -ArgumentList "infra/mediamtx.yml" -WindowStyle Normal
} else {
    Write-Host "[-] Warning: infra/mediamtx.exe not found. Make sure MediaMTX is downloaded in the infra folder." -ForegroundColor Yellow
}

# 5. Start Python backend server using uv
Write-Host "[+] Starting Python Backend Server..." -ForegroundColor Cyan
Start-Process -FilePath "uv" -ArgumentList "run python -m server" -WindowStyle Normal

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "          All Services Started Successfully!      " -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host " -> Backend API Doc / Dashboard: http://$($lanIp):8000" -ForegroundColor Yellow
Write-Host " -> MediaMTX WHIP Endpoint:    http://$($lanIp):8889/mystream/whip" -ForegroundColor Yellow
Write-Host "==================================================" -ForegroundColor Cyan