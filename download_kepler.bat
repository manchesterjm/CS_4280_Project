@echo off
REM Download Kepler confirmed planet light curves for cross-mission training
REM Usage: download_kepler.bat [n_targets]
REM   n_targets: Number of Kepler planets to download (default: 100)

setlocal EnableDelayedExpansion

REM Default to 100 planets if not specified
set N_TARGETS=%1
if "%N_TARGETS%"=="" set N_TARGETS=100

set OUTPUT_DIR=E:\kepler_lightcurves

echo ============================================================
echo   DOWNLOADING KEPLER CONFIRMED PLANET LIGHT CURVES
echo ============================================================
echo.
echo Output directory: %OUTPUT_DIR%
echo Number of targets: %N_TARGETS%
echo.
echo This will download actual Kepler observations using lightkurve.
echo Estimated time: 10-30 minutes depending on connection.
echo.
pause

REM Initialize conda
call C:\Users\manch\miniconda3\condabin\conda.bat activate exo-lstm-gpu

cd /d C:\CS_4280_Project\Code

python download_kepler_planets.py --output_dir "%OUTPUT_DIR%" --n_targets %N_TARGETS% --save_list

echo.
echo ============================================================
echo   DOWNLOAD COMPLETE
echo ============================================================
echo.
echo Next step: Run exo_build_kepler.bat to process the data
echo.
pause
