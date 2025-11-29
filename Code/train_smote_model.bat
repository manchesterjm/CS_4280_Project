@echo off
echo Training model on SMOTE-balanced dataset...
echo.

call conda activate exo-lstm-gpu

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

echo.
echo Training complete!
