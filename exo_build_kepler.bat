@echo off
REM Build training windows from downloaded Kepler light curves
REM Usage: exo_build_kepler.bat

echo ============================================================
echo   BUILDING WINDOWS FROM KEPLER LIGHT CURVES
echo ============================================================
echo.

REM Initialize conda
call C:\Users\manch\miniconda3\condabin\conda.bat activate exo-lstm-gpu

cd /d C:\CS_4280_Project\Code

python build_windows_from_kepler_csv.py ^
    --data_dir E:\kepler_lightcurves ^
    --output_dir data\windows_kepler ^
    --seq_len 2048 ^
    --n_windows 3

echo.
echo ============================================================
echo   BUILD COMPLETE
echo ============================================================
echo.
echo Next step: Run exo_combine.bat to merge with TESS data
echo.
pause
