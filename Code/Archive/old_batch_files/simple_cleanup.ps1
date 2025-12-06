# Simple Project Cleanup
# Moves old diagnostic/training scripts to Archive

$ProjectRoot = "C:\CS_4280_Project\Code"
$ArchiveScripts = "$ProjectRoot\Archive\scripts"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " PROJECT CLEANUP" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

Set-Location $ProjectRoot

# Ensure Archive/scripts exists
if (-Not (Test-Path $ArchiveScripts)) {
    New-Item -ItemType Directory -Force -Path $ArchiveScripts | Out-Null
}

Write-Host "`nMoving old diagnostic/development scripts to Archive..." -ForegroundColor Yellow

# List of scripts to archive (keep working scripts in main directory)
$scriptsToArchive = @(
    "check_gpu.py",
    "diagnose_data.py",
    "investigate_data.py"
)

$moved = 0
foreach ($script in $scriptsToArchive) {
    if (Test-Path $script) {
        Write-Host "  Moving: $script" -ForegroundColor Gray
        Move-Item -Path $script -Destination $ArchiveScripts -Force
        $moved++
    }
}

if ($moved -eq 0) {
    Write-Host "  No files to move (already clean)" -ForegroundColor Green
}

Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host " ACTIVE WORKING FILES" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

Write-Host "`nCore Pipeline Scripts:" -ForegroundColor Yellow
Write-Host "  ✓ train_bilstm_cluster.py         - Training with clustering" -ForegroundColor Green
Write-Host "  ✓ build_windows_parallel_v6.py    - Build training windows" -ForegroundColor Green
Write-Host "  ✓ build_simple_windows.py         - Build test windows" -ForegroundColor Green
Write-Host "  ✓ inference_cluster_model.py      - Run predictions" -ForegroundColor Green

Write-Host "`nData Download & Processing:" -ForegroundColor Yellow
Write-Host "  ✓ download_tess_lightcurves.py    - Download TESS data" -ForegroundColor Green
Write-Host "  ✓ process_tess_for_testing.py     - Process light curves" -ForegroundColor Green
Write-Host "  ✓ convert_npy_to_csv.py            - Format converter" -ForegroundColor Green

Write-Host "`nTrained Models:" -ForegroundColor Yellow
Write-Host "  ✓ runs/bilstm_cluster/best.pt     - Best model (AUC 0.69)" -ForegroundColor Green
Write-Host "  ✓ runs/bilstm_cluster_v2/best.pt  - 8 clusters (AUC 0.68)" -ForegroundColor Green

Write-Host "`nData:" -ForegroundColor Yellow
Write-Host "  ✓ data/windows_train/             - Training windows" -ForegroundColor Green
Write-Host "  ✓ data/windows_test/              - Test windows (TESS)" -ForegroundColor Green

Write-Host "`nResults:" -ForegroundColor Yellow
Write-Host "  ✓ reports/test_predictions.csv    - TESS predictions" -ForegroundColor Green

Write-Host "`n" + "="*70 -ForegroundColor Cyan
Write-Host " CLEANUP COMPLETE" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""
