# get_mediamtx.ps1 - Downloads and extracts MediaMTX pinned release for Windows
$ErrorActionPreference = "Stop"

$version = "v1.11.3"
$zipName = "mediamtx_${version}_windows_amd64.zip"
$downloadUrl = "https://github.com/bluenviron/mediamtx/releases/download/$version/$zipName"
$infraDir = Join-Path $PSScriptRoot ".."
$zipPath = Join-Path $infraDir $zipName

Write-Host "Downloading MediaMTX $version..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath

Write-Host "Extracting mediamtx.exe..." -ForegroundColor Cyan
Expand-Archive -Path $zipPath -DestinationPath $infraDir -Force

Write-Host "Cleaning up zip file..." -ForegroundColor Cyan
Remove-Item $zipPath

Write-Host "MediaMTX successfully downloaded to infra/ directory!" -ForegroundColor Green