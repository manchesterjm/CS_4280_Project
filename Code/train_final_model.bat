@echo off
REM Final Model Training Script - December 6, 2025
REM Uses best hyperparameters from Optuna overnight run (Trial 21, AUC 0.9182)
REM
REM Best parameters:
REM   hidden_size: 192
REM   batch_size: 128
REM   num_layers: 4
REM   dropout: 0.334
REM   lr: 0.0000497
REM   n_clusters: 7
REM   weight_decay: 0.0000154
REM   cluster_embed_dim: 64
REM
REM Expected training time: ~70 minutes (60 epochs x ~1.1 min/epoch)

echo ============================================================
echo   FINAL MODEL TRAINING - Best Optuna Hyperparameters
echo ============================================================
echo.
echo Best Trial: 21
echo Best AUC: 0.9182
echo.
echo Hyperparameters:
echo   hidden_size: 192
echo   batch_size: 128
echo   num_layers: 4
echo   dropout: 0.334
echo   lr: 0.0000497
echo   n_clusters: 7
echo   cluster_embed_dim: 64
echo.
echo Expected time: ~70 minutes for 60 epochs
echo.
echo Press Ctrl+C to cancel, or any key to start...
pause > nul

C:\Users\manch\miniconda3\envs\exo-lstm-gpu\python.exe -u ^
  D:\CS_4280_Project\Code\train_bilstm_cluster.py ^
  --windows_dir "D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train" ^
  --n_clusters 7 ^
  --cluster_embed_dim 64 ^
  --epochs 60 ^
  --batch_size 128 ^
  --lr 0.0000497 ^
  --hidden 192 ^
  --layers 4 ^
  --dropout 0.334 ^
  --weight_decay 0.0000154 ^
  --save_dir "D:\CS_4280_Project\Code\runs\sector1_final_0918" ^
  --amp_dtype fp16 ^
  --pos_weight 7.41 ^
  --num_workers 0 ^
  --seed 42

echo.
echo ============================================================
echo   Training Complete!
echo ============================================================
echo.
echo Model saved to: runs\sector1_final_0918\
echo.
pause
