# PowerShell script to clean up CS_4280_Project structure
# Moves old/unused files to Archive folder

$ProjectRoot = "C:\CS_4280_Project\Code"
$ArchiveDir = "$ProjectRoot\Archive"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " CS_4280_Project Cleanup Script" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

# Ensure we're in the right directory
if (-Not (Test-Path $ProjectRoot)) {
    Write-Host "Error: Project directory not found: $ProjectRoot" -ForegroundColor Red
    exit 1
}

Set-Location $ProjectRoot

# Create Archive subdirectories if they don't exist
$ArchiveScripts = "$ArchiveDir\scripts"

Write-Host "Creating archive structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $ArchiveScripts | Out-Null

Write-Host "Archive structure ready.`n" -ForegroundColor Green

# Old training scripts to archive
$OldTrainingScripts = @(
    "train_lstm_retrain.py",
    "train_lstm_retrain_v2.py",
    "train_lstm_cluster_v3.py"
)

# Old window building scripts to archive
$OldWindowScripts = @(
    "build_windows_v4.py",
    "build_windows_parallel_v5.py",
    "build_windows_infer_v1.py"
)

# Old postfilter scripts to archive
$OldPostfilterScripts = @(
    "postfilter_inference.py",
    "postfilter_inference_v2.py",
    "postfilter_sweep.py"
)

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Archiving Old Scripts" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

$movedCount = 0

# Archive old training scripts
Write-Host "Archiving old training scripts..." -ForegroundColor Cyan
foreach ($script in $OldTrainingScripts) {
    if (Test-Path $script) {
        Write-Host "  Moving: $script" -ForegroundColor Yellow
        Move-Item -Path $script -Destination $ArchiveScripts -Force
        $movedCount++
    }
}

# Archive old window building scripts
Write-Host "`nArchiving old window building scripts..." -ForegroundColor Cyan
foreach ($script in $OldWindowScripts) {
    if (Test-Path $script) {
        Write-Host "  Moving: $script" -ForegroundColor Yellow
        Move-Item -Path $script -Destination $ArchiveScripts -Force
        $movedCount++
    }
}

# Archive old postfilter scripts
Write-Host "`nArchiving old postfilter scripts..." -ForegroundColor Cyan
foreach ($script in $OldPostfilterScripts) {
    if (Test-Path $script) {
        Write-Host "  Moving: $script" -ForegroundColor Yellow
        Move-Item -Path $script -Destination $ArchiveScripts -Force
        $movedCount++
    }
}

Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host " Current Active Files" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Python scripts in Code directory:" -ForegroundColor Yellow
Get-ChildItem "*.py" | ForEach-Object {
    Write-Host "  $($_.Name)" -ForegroundColor Green
}

Write-Host "`n=====================================================================" -ForegroundColor Cyan
Write-Host " Summary" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Files moved to archive: $movedCount" -ForegroundColor Yellow
Write-Host ""
Write-Host "Cleanup complete!" -ForegroundColor Green
Write-Host ""
