@echo off
REM Simple exoplanet training wrapper
REM Usage: exo_train 1          (train 1 epoch, batch=32, fastest on 8GB GPU)
REM        exo_train 5 large    (train 5 epochs, batch=128, slower but stable gradients)
REM        exo_train 25         (overnight run - 25 epochs, ~2 hours)

setlocal EnableDelayedExpansion

REM Default to 1 epoch if not specified
set EPOCHS=%1
if "%EPOCHS%"=="" set EPOCHS=1

REM Batch size: default=32 (fastest on 8GB GPU), "large"=128
set BATCH_SIZE=32
if /i "%2"=="large" (
    set BATCH_SIZE=128
    echo [exo_train] Large batch mode: batch_size=128 ^(slower on 8GB GPU^)
) else if /i "%2"=="medium" (
    set BATCH_SIZE=64
    echo [exo_train] Medium mode: batch_size=64
)

REM Standard training directory
set SAVE_DIR=C:\CS_4280_Project\Code\runs\sector1_incremental
set CHECKPOINT=%SAVE_DIR%\last.pt

REM Check if we should resume
set RESUME_ARG=
if exist "%CHECKPOINT%" (
    echo [exo_train] Found checkpoint, will resume from %CHECKPOINT%
    set "RESUME_ARG=--resume "%CHECKPOINT%""
)

echo [exo_train] Training for %EPOCHS% epoch(s), batch_size=%BATCH_SIZE%
echo.

REM Initialize conda
call C:\Users\manch\miniconda3\condabin\conda.bat activate exo-lstm-gpu

cd /d C:\CS_4280_Project\Code

python train_bilstm_cluster.py ^
    --windows_dir "data\windows_sector1_full\train" ^
    --n_clusters 5 ^
    --epochs_to_train %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --lr 0.0001 ^
    --hidden 256 ^
    --layers 4 ^
    --dropout 0.311 ^
    --save_dir "%SAVE_DIR%" ^
    --amp_dtype fp16 ^
    --pos_weight 7.41 ^
    --num_workers 0 ^
    --seed 42 ^
    %RESUME_ARG%

echo.
echo [exo_train] Done!
