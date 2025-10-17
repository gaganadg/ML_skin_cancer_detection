param(
  [string]$OutDir = "data/ham_images"
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Ensure-KaggleCli {
  try {
    kaggle --version | Out-Null
    return $true
  } catch {
    return $false
  }
}

function Ensure-KaggleCreds {
  $credPath = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
  return (Test-Path $credPath)
}

if (-not (Ensure-KaggleCli)) {
  Write-Host "Kaggle CLI not found. Installing via pip..." -ForegroundColor Yellow
  python -m pip install --upgrade kaggle | Out-Null
}

if (-not (Ensure-KaggleCreds)) {
  throw "Kaggle credentials not found. Place kaggle.json at %USERPROFILE%\.kaggle\kaggle.json"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "Downloading HAM10000 images (may take a while)..." -ForegroundColor Cyan
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p $OutDir -f HAM10000_images_part_1.zip -q
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000 -p $OutDir -f HAM10000_images_part_2.zip -q

Write-Host "Extracting..." -ForegroundColor Cyan
Get-ChildItem $OutDir -Filter "HAM10000_images_part_*.zip" | ForEach-Object {
  Expand-Archive -Path $_.FullName -DestinationPath $OutDir -Force
}

Write-Host "Cleaning archives..." -ForegroundColor Cyan
Get-ChildItem $OutDir -Filter "HAM10000_images_part_*.zip" | Remove-Item -Force

Write-Host "Done. Images at $OutDir" -ForegroundColor Green

