param(
  [int]$MlPort = 8001,
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Write-Host "Restarting all services..." -ForegroundColor Cyan

# ML
& .\scripts\restart_ml.ps1 -Port $MlPort

# Backend
try {
  $conns = Get-NetTCPConnection -LocalPort $BackendPort -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($pid in $conns) { try { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } catch {} }
} catch {}
Start-Process -WindowStyle Hidden -WorkingDirectory (Resolve-Path 'backend').Path -FilePath cmd.exe -ArgumentList '/c','npm run dev' | Out-Null

# Frontend
try {
  $conns = Get-NetTCPConnection -LocalPort $FrontendPort -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
  foreach ($pid in $conns) { try { Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue } catch {} }
} catch {}
Start-Process -WindowStyle Hidden -WorkingDirectory (Resolve-Path 'frontend').Path -FilePath cmd.exe -ArgumentList '/c','npm run dev' | Out-Null

# Verify health
Start-Sleep -Seconds 2
$backendOk = $false
$frontendOk = $false
try { $b = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$BackendPort/health").Content; if ($b -like '*ok*') { $backendOk = $true } } catch {}
try { $f = (Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$FrontendPort/").StatusCode; if ($f -eq 200) { $frontendOk = $true } } catch {}

if (-not $backendOk) { Write-Warning "Backend might not be ready on :$BackendPort" }
if (-not $frontendOk) { Write-Warning "Frontend might not be ready on :$FrontendPort" }

# Open browser
Start-Process "http://localhost:$FrontendPort/"

Write-Host "All restart commands issued." -ForegroundColor Green

