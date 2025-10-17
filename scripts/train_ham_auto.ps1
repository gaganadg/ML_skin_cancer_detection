param(
  [string]$CsvPath = "data/train/HAM10000_metadata.csv",
  [string]$ImagesDir = "",
  [int]$Epochs = 8,
  [int]$BatchSize = 16,
  [double]$Lr = 0.001,
  [string]$Out = "ml_service/model_weights.pth"
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not (Test-Path $CsvPath)) { throw "Metadata CSV not found: $CsvPath" }

function Find-HamImagesDir {
  param([string]$CsvDir)
  $candidates = @(
    Join-Path $CsvDir '..', '..', 'ham_images', '','' | Resolve-Path -ErrorAction SilentlyContinue,
    (Join-Path $CsvDir '..' | Resolve-Path -ErrorAction SilentlyContinue),
    (Resolve-Path 'data/ham_images' -ErrorAction SilentlyContinue)
  ) | Where-Object { $_ -ne $null } | ForEach-Object { $_.Path }
  foreach ($d in $candidates) {
    if (Test-Path $d) {
      $hasJpg = Get-ChildItem -Path $d -Filter '*.jpg' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
      if ($hasJpg) { return $d }
    }
  }
  return $null
}

& .\.venv\Scripts\Activate.ps1

if (-not $ImagesDir) {
  $CsvFull = Resolve-Path $CsvPath
  $CsvDir = Split-Path -Parent $CsvFull
  $ImagesDir = Find-HamImagesDir -CsvDir $CsvDir
  if (-not $ImagesDir) { throw "Could not auto-detect HAM images directory. Place images under data/ham_images or specify --ImagesDir." }
}

Write-Host "Using ImagesDir=$ImagesDir" -ForegroundColor Cyan

.
\.venv\Scripts\python.exe ml_service\train_ham.py --metadata-csv $CsvPath --images-dir $ImagesDir --epochs $Epochs --batch-size $BatchSize --lr $Lr --out $Out

# Restart ML service safely
Write-Host "Restarting ML service..." -ForegroundColor Cyan
Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like '*python*' } | ForEach-Object { try { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } catch {} }
Start-Sleep -Seconds 1
Start-Process -WindowStyle Hidden -FilePath powershell -ArgumentList @('-NoProfile','-Command','cd "'+$PWD.Path+'"; .\.venv\Scripts\Activate.ps1; py ml_service\main.py') | Out-Null
Start-Sleep -Seconds 2
try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/health | Out-Null; Write-Host "ML: OK" -ForegroundColor Green } catch { Write-Host "ML: FAIL" -ForegroundColor Red }




