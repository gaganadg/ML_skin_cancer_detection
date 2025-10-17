param(
  [string]$DataDir = "data",
  [int]$MLEpochs = 5,
  [int]$BatchSize = 16,
  [double]$Lr = 0.001
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Test-PortOpen([int]$Port){
  try {
    $listener = New-Object System.Net.Sockets.TcpClient
    $async = $listener.BeginConnect('127.0.0.1', $Port, $null, $null)
    $wait = $async.AsyncWaitHandle.WaitOne(200)
    if ($wait -and $listener.Connected) { $listener.Close(); return $true } else { return $false }
  } catch { return $false }
}

Write-Host "[1/6] Python venv + ML deps" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { py -m venv .venv }
& .\.venv\Scripts\Activate.ps1
pip install -q -r ml_service/requirements.txt

Write-Host "[2/6] Dataset audit (if exists)" -ForegroundColor Cyan
if (Test-Path $DataDir) {
  .\.venv\Scripts\python.exe ml_service\train.py --data-dir $DataDir --audit-only
} else {
  Write-Host "Dataset not found at '$DataDir' (skipping audit)" -ForegroundColor Yellow
}

Write-Host "[3/6] Train model (if dataset present)" -ForegroundColor Cyan
if (Test-Path $DataDir) {
  .\.venv\Scripts\python.exe ml_service\train.py --data-dir $DataDir --epochs $MLEpochs --batch-size $BatchSize --lr $Lr
} else {
  Write-Host "Dataset not found at '$DataDir' (skipping training)" -ForegroundColor Yellow
}

Write-Host "[4/6] Start ML service (8001)" -ForegroundColor Cyan
if (-not (Test-PortOpen 8001)) {
  Start-Process -WindowStyle Hidden -FilePath powershell -ArgumentList @('-NoProfile','-Command','cd "'+$PWD.Path+'"; .\.venv\Scripts\Activate.ps1; py ml_service\main.py') | Out-Null
  Start-Sleep -Seconds 2
} else { Write-Host "Port 8001 already in use (ML likely running)" -ForegroundColor Yellow }

Write-Host "[5/6] Start backend (8000)" -ForegroundColor Cyan
if (-not (Test-PortOpen 8000)) {
  if (-not (Test-Path "backend\node_modules")) { npm --prefix backend install }
  Start-Process -WindowStyle Hidden -FilePath powershell -ArgumentList @('-NoProfile','-Command','cd backend; npm start') | Out-Null
  Start-Sleep -Seconds 2
} else { Write-Host "Port 8000 already in use (backend likely running)" -ForegroundColor Yellow }

Write-Host "[6/6] Start frontend (5173)" -ForegroundColor Cyan
if (-not (Test-PortOpen 5173)) {
  if (-not (Test-Path "frontend\node_modules")) { npm --prefix frontend install }
  Start-Process -WindowStyle Hidden -FilePath powershell -ArgumentList @('-NoProfile','-Command','cd frontend; npm run dev -- --host') | Out-Null
  Start-Sleep -Seconds 2
} else { Write-Host "Port 5173 already in use (frontend likely running)" -ForegroundColor Yellow }

Write-Host "\nHealth checks:" -ForegroundColor Green
try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/health | Out-Null; Write-Host "ML   : OK -> http://127.0.0.1:8001/health" -ForegroundColor Green } catch { Write-Host "ML   : FAIL" -ForegroundColor Red }
try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health | Out-Null; Write-Host "BE   : OK -> http://127.0.0.1:8000/health" -ForegroundColor Green } catch { Write-Host "BE   : FAIL" -ForegroundColor Red }
try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5173 | Out-Null; Write-Host "FE   : OK -> http://127.0.0.1:5173" -ForegroundColor Green } catch { Write-Host "FE   : starting -> http://127.0.0.1:5173" -ForegroundColor Yellow }

Write-Host "\nDone. Open http://localhost:5173 to use the app." -ForegroundColor Green












