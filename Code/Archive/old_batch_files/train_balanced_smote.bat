@echo off
echo ====================================================================
echo TRUE SMOTE: Creating Balanced Dataset via Interpolation
echo ====================================================================
echo.
echo This uses SMOTE to create NEW synthetic planets by interpolating
echo between real ones, NOT simple duplication.
echo.
echo Strategy:
echo   1. SMOTE: Create 150 NEW planets by blending original 150
echo   2. Down-sample: Reduce non-planets from 505 to 300
echo   3. Result: 300 planets (150 real + 150 synthetic) + 300 non-planets
echo.
pause

echo.
echo Step 1: Creating balanced dataset with TRUE SMOTE...
echo ====================================================================
python balance_with_true_smote.py ^
  --windows_dir "data/windows_train" ^
  --output_dir "data/windows_smote_true" ^
  --method hybrid ^
  --target_size 300 ^
  --seed 42

if errorlevel 1 (
    echo ERROR: SMOTE balancing failed!
    pause
    exit /b 1
)

echo.
echo Step 2: Training model on SMOTE-balanced dataset...
echo ====================================================================
echo This may take 25-30 minutes...
echo.

python train_bilstm_cluster.py ^
  --windows_dir "data/windows_smote_true" ^
  --n_clusters 5 ^
  --epochs 80 ^
  --batch_size 128 ^
  --lr 0.000225 ^
  --hidden 256 ^
  --layers 4 ^
  --dropout 0.311 ^
  --save_dir "runs/bilstm_cluster_smote_true" ^
  --amp_dtype fp16 ^
  --pos_weight 1.0 ^
  --num_workers 0

if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)

echo.
echo Step 3: Testing on 100 real planets...
echo ====================================================================

python inference_cluster_model.py ^
  --model_path "runs/bilstm_cluster_smote_true/best.pt" ^
  --windows_dir "data/windows_planet_test" ^
  --output_file "reports/smote_true_planet_predictions.csv"

if errorlevel 1 (
    echo ERROR: Inference failed!
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo COMPLETE!
echo ====================================================================
echo.
echo Results:
echo   Model: runs/bilstm_cluster_smote_true/best.pt
echo   Predictions: reports/smote_true_planet_predictions.csv
echo.
echo Compare with:
echo   Unbalanced model: reports/optimized_planet_predictions.csv (16/300)
echo   Simple duplication: reports/balanced_planet_predictions.csv (0/300)
echo   SMOTE interpolation: reports/smote_true_planet_predictions.csv (???)
echo.
echo Expected: Better than duplication (0/300), may approach unbalanced (16/300)
echo.
pause
