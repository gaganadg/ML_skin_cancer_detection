param(
  [int]$Port = 8001
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Write-Host "Restarting ML service on port $Port..." -ForegroundColor Cyan

# Kill python processes bound to the ML port
try {
  $conns = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($pid in $conns) { try { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } catch {} }
} catch {}

# Activate venv if exists
$activate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activate) { . $activate }

# Start uvicorn in background
Start-Process -WindowStyle Hidden cmd.exe -ArgumentList '/c',"uvicorn ml_service.main:app --host 0.0.0.0 --port $Port" | Out-Null

# Wait for health
$deadline = (Get-Date).AddSeconds(10)
while ((Get-Date) -lt $deadline) {
  try {
    $resp = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/health").Content
    if ($resp -like '*ok*') { Write-Host "ML service is up on $Port" -ForegroundColor Green; exit 0 }
  } catch { Start-Sleep -Milliseconds 500 }
}

throw "ML service failed to start on port $Port"
