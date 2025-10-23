# Final Project Cleanup Script
# Organizes all files before GitHub push

$ProjectRoot = "C:\CS_4280_Project\Code"
$ArchiveDir = "$ProjectRoot\Archive"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " FINAL PROJECT CLEANUP" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

Set-Location $ProjectRoot

# Create archive subdirectories
New-Item -ItemType Directory -Force -Path "$ArchiveDir\scripts" | Out-Null
New-Item -ItemType Directory -Force -Path "$ArchiveDir\runs" | Out-Null

Write-Host "`nArchiving old files..." -ForegroundColor Yellow

# Archive old training scripts (keep only the working ones)
$OldScripts = @(
    "train_lstm_fixed.py",
    "train_bilstm.py",
    "diagnose_data.py",
    "investigate_data.py",
    "check_gpu.py"
)

foreach ($script in $OldScripts) {
    if (Test-Path $script) {
        Write-Host "  Archiving: $script" -ForegroundColor Gray
        Move-Item -Path $script -Destination "$ArchiveDir\scripts\" -Force
    }
}

# Archive old runs (keep only bilstm_cluster)
if (Test-Path "runs") {
    $runDirs = Get-ChildItem "runs" -Directory
    foreach ($runDir in $runDirs) {
        if ($runDir.Name -ne "bilstm_cluster") {
            Write-Host "  Archiving run: $($runDir.Name)" -ForegroundColor Gray
            if (-Not (Test-Path "$ArchiveDir\runs")) {
                New-Item -ItemType Directory -Path "$ArchiveDir\runs" | Out-Null
            }
            Move-Item -Path $runDir.FullName -Destination "$ArchiveDir\runs\" -Force
        }
    }
}

Write-Host "`n✓ Cleanup complete!" -ForegroundColor Green

Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host " ACTIVE PROJECT FILES" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

Write-Host "`nCore Scripts:" -ForegroundColor Yellow
$coreScripts = @(
    "train_bilstm_cluster.py",
    "build_windows_parallel_v6.py",
    "build_simple_windows.py",
    "inference_cluster_model.py",
    "download_tess_lightcurves.py",
    "process_tess_for_testing.py",
    "convert_npy_to_csv.py"
)
foreach ($script in $coreScripts) {
    if (Test-Path $script) {
        Write-Host "  ✓ $script" -ForegroundColor Green
    }
}

Write-Host "`nData:" -ForegroundColor Yellow
Write-Host "  ✓ data/windows_train/" -ForegroundColor Green
Write-Host "  ✓ data/windows_test/" -ForegroundColor Green

Write-Host "`nModel:" -ForegroundColor Yellow
Write-Host "  ✓ runs/bilstm_cluster/best.pt" -ForegroundColor Green

Write-Host "`nResults:" -ForegroundColor Yellow
Write-Host "  ✓ reports/test_predictions.csv" -ForegroundColor Green

Write-Host ""
