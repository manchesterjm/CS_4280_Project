@echo off
REM Quick-start script for balancing dataset and training
REM Based on professor's recommendation: "SMOTE, down sample then up sample"

echo ====================================================================
echo Class Imbalance Solution: Down-sample + Up-sample
echo ====================================================================
echo.
echo Current dataset: 150 planets (23%%) + 505 non-planets (77%%)
echo Target dataset:  300 planets (50%%) + 300 non-planets (50%%)
echo.
echo This will:
echo   1. Down-sample non-planets: 505 -^> 300
echo   2. Up-sample planets: 150 -^> 300
echo   3. Result: 600 balanced windows (50/50)
echo.
pause

echo.
echo Step 1: Installing imbalanced-learn (if needed)...
echo ====================================================================
pip install imbalanced-learn

echo.
echo Step 2: Balancing dataset...
echo ====================================================================
python balance_dataset_smote.py ^
  --windows_dir "data/windows_train" ^
  --output_dir "data/windows_balanced" ^
  --method downsample_upsample ^
  --target_size 300 ^
  --seed 42

if errorlevel 1 (
    echo ERROR: Balancing failed!
    pause
    exit /b 1
)

echo.
echo Step 3: Training model on balanced dataset...
echo ====================================================================
echo NOTE: pos_weight changed from 3.367 to 1.0 since data is now balanced
echo.

python train_bilstm_cluster.py ^
  --windows_dir "data/windows_balanced" ^
  --n_clusters 5 ^
  --epochs 80 ^
  --batch_size 128 ^
  --lr 0.000225 ^
  --hidden 256 ^
  --layers 4 ^
  --dropout 0.311 ^
  --save_dir "runs/bilstm_cluster_balanced" ^
  --amp_dtype fp16 ^
  --pos_weight 1.0 ^
  --num_workers 0

if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo COMPLETE! Model trained on balanced dataset
echo ====================================================================
echo.
echo Results saved to: runs/bilstm_cluster_balanced/
echo.
echo Next steps:
echo   1. Compare AUC with unbalanced model (baseline: 0.7572)
echo   2. Check precision improvement
echo   3. Test on 100 real planets
echo.
pause
