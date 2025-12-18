@echo off
REM Train both hybrid models sequentially
REM Run time: ~50-60 minutes total

echo ================================================================
echo Training Hybrid Models
echo ================================================================
echo.
echo This will train two models:
echo   1. Hybrid 90 (90%% real, 10%% synthetic) - 25 minutes
echo   2. Hybrid 75 (75%% real, 25%% synthetic) - 28 minutes
echo.
echo Total estimated time: 50-60 minutes
echo.
pause

echo.
echo ================================================================
echo Training Hybrid 90%% Model
echo ================================================================
echo.

call conda activate exo-lstm-gpu
python train_bilstm_cluster.py ^
  --windows_dir data/windows_hybrid_90 ^
  --n_clusters 5 ^
  --epochs 80 ^
  --batch_size 128 ^
  --lr 0.000225 ^
  --hidden 256 ^
  --layers 4 ^
  --dropout 0.311 ^
  --save_dir runs/bilstm_cluster_hybrid_90 ^
  --amp_dtype fp16 ^
  --pos_weight 3.0 ^
  --num_workers 0

if errorlevel 1 (
    echo.
    echo ERROR: Hybrid 90 training failed!
    pause
    exit /b 1
)

echo.
echo Hybrid 90 training complete!
echo.
echo ================================================================
echo Training Hybrid 75%% Model
echo ================================================================
echo.

python train_bilstm_cluster.py ^
  --windows_dir data/windows_hybrid_75 ^
  --n_clusters 5 ^
  --epochs 80 ^
  --batch_size 128 ^
  --lr 0.000225 ^
  --hidden 256 ^
  --layers 4 ^
  --dropout 0.311 ^
  --save_dir runs/bilstm_cluster_hybrid_75 ^
  --amp_dtype fp16 ^
  --pos_weight 3.0 ^
  --num_workers 0

if errorlevel 1 (
    echo.
    echo ERROR: Hybrid 75 training failed!
    pause
    exit /b 1
)

echo.
echo ================================================================
echo Both models trained successfully!
echo ================================================================
echo.
echo Next steps:
echo   1. Benchmark models: python benchmark_model.py --model_path runs/bilstm_cluster_hybrid_90/best.pt
echo   2. Test on planets: python inference_cluster_model.py --model_path runs/bilstm_cluster_hybrid_90/best.pt --windows_dir data/windows_planet_test
echo   3. Compare results: See HYBRID_APPROACH_SUMMARY.md
echo.
pause
