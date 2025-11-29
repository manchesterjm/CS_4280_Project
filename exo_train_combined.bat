@echo off
REM Train BiLSTM on combined TESS + Kepler dataset
REM Usage: exo_train_combined.bat [epochs]
REM   epochs: Number of training epochs (default: 60)

setlocal EnableDelayedExpansion

set EPOCHS=%1
if "%EPOCHS%"=="" set EPOCHS=60

echo ============================================================
echo   TRAINING ON COMBINED TESS + KEPLER DATASET
echo ============================================================
echo.
echo Epochs: %EPOCHS%
echo Dataset: data\windows_combined
echo.

REM Initialize conda
call C:\Users\manch\miniconda3\condabin\conda.bat activate exo-lstm-gpu

cd /d C:\CS_4280_Project\Code

REM Note: The combine script outputs the recommended pos_weight
REM You may need to adjust this based on the actual class balance
set POS_WEIGHT=5.0

python train_simple_bilstm.py ^
    --windows_dir data\windows_combined ^
    --epochs %EPOCHS% ^
    --batch_size 32 ^
    --lr 0.0001 ^
    --hidden 256 ^
    --layers 4 ^
    --dropout 0.3 ^
    --pos_weight %POS_WEIGHT% ^
    --save_dir runs\combined_model ^
    --amp_dtype fp16 ^
    --num_workers 0 ^
    --seed 42

echo.
echo ============================================================
echo   TRAINING COMPLETE
echo ============================================================
echo.
echo Model saved to: runs\combined_model\best.pt
echo.
echo To evaluate on different datasets:
echo   python evaluate_simple.py --model_path runs\combined_model\best.pt --data_dir [DATA_DIR]
echo.
pause
