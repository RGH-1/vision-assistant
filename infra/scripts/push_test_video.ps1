param (
    [string]$VideoPath = "data/videos/classroom.mp4"
)

if (-not (Test-Path $VideoPath)) {
    Write-Error "Video file not found at: $VideoPath."
    exit 1
}

Write-Host "Streaming $VideoPath to rtsp://localhost:8554/phone..." -ForegroundColor Green

# FFmpeg loops the video indefinitely and pushes H264 RTSP feed to MediaMTX
ffmpeg -re -stream_loop -1 -i "$VideoPath" -c:v libx264 -preset veryfast -tune zerolatency -an -f rtsp rtsp://localhost:8554/phone