@echo off
REM Simple BiLSTM training (no clustering - more generalizable)
REM Usage: exo_simple 5         (train 5 epochs)
REM        exo_simple 10 large  (train 10 epochs, batch=128)

setlocal EnableDelayedExpansion

REM Default to 5 epochs if not specified
set EPOCHS=%1
if "%EPOCHS%"=="" set EPOCHS=5

REM Batch size: default=32 (fastest), "large"=128
set BATCH_SIZE=32
if /i "%2"=="large" (
    set BATCH_SIZE=128
    echo [exo_simple] Large batch mode: batch_size=128
)

set SAVE_DIR=C:\CS_4280_Project\Code\runs\simple_bilstm

echo [exo_simple] Training Simple BiLSTM (no clustering) for %EPOCHS% epoch(s), batch_size=%BATCH_SIZE%
echo.

REM Initialize conda
call C:\Users\manch\miniconda3\condabin\conda.bat activate exo-lstm-gpu

cd /d C:\CS_4280_Project\Code

python train_simple_bilstm.py ^
    --windows_dir "data\windows_sector1_full\train" ^
    --epochs %EPOCHS% ^
    --batch_size %BATCH_SIZE% ^
    --lr 0.0001 ^
    --hidden 256 ^
    --layers 4 ^
    --dropout 0.3 ^
    --pos_weight 7.41 ^
    --save_dir "%SAVE_DIR%" ^
    --amp_dtype fp16 ^
    --num_workers 0 ^
    --seed 42

echo.
echo [exo_simple] Done!
