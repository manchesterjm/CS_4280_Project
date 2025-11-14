@echo off
echo Training Simple BiLSTM on TESS Sector 1 dataset...
echo.

call conda activate exo-lstm-gpu

python train_simple_bilstm.py ^
  --windows_dir "data/windows_sector1_full" ^
  --hidden 256 ^
  --layers 4 ^
  --dropout 0.311 ^
  --epochs 80 ^
  --batch_size 128 ^
  --lr 0.000225 ^
  --weight_decay 1e-5 ^
  --pos_weight 3.367 ^
  --gradient_clip 1.0 ^
  --save_dir "runs/bilstm_sector1" ^
  --amp_dtype fp16 ^
  --val_split 0.15 ^
  --num_workers 0 ^
  --seed 42

echo.
echo Training complete!
