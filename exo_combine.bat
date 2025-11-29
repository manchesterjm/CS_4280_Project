@echo off
REM Combine TESS Sector 1 + Kepler datasets for cross-mission training
REM Usage: exo_combine.bat

echo ============================================================
echo   COMBINING TESS + KEPLER DATASETS
echo ============================================================
echo.

REM Initialize conda
call C:\Users\manch\miniconda3\condabin\conda.bat activate exo-lstm-gpu

cd /d C:\CS_4280_Project\Code

python combine_datasets.py ^
    --tess_dir data\windows_sector1_full\train ^
    --kepler_dir data\windows_kepler ^
    --output_dir data\windows_combined

echo.
echo ============================================================
echo   COMBINATION COMPLETE
echo ============================================================
echo.
echo Next step: Run exo_train_combined.bat to train on combined data
echo.
pause
