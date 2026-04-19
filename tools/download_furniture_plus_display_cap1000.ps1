param(
    [string]$Manifest = "artifacts/shapenet_subsets/furniture_plus_display_cap1000_manifest.json",
    [string]$EnvFile = ".env",
    [string]$OutputDir = "data/shapenet_subsets/furniture_plus_display_cap1000",
    [int]$Workers = 8,
    [int]$Limit = 0,
    [switch]$ForceDownload
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found in PATH."
}

$cmd = @(
    "run",
    "--python", ".venv/Scripts/python.exe",
    "python",
    "tools/download_shapenet_subset.py",
    "--manifest", $Manifest,
    "--env-file", $EnvFile,
    "--output-dir", $OutputDir,
    "--workers", "$Workers"
)

if ($Limit -gt 0) {
    $cmd += @("--limit", "$Limit")
}

if ($ForceDownload) {
    $cmd += "--force-download"
}

Write-Host "Running: uv $($cmd -join ' ')"
& uv @cmd
if ($LASTEXITCODE -ne 0) {
    throw "Subset download command failed."
}

Write-Host "Subset download complete."
