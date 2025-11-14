@echo off
echo Building training dataset from TESS Sector 1 ground truth...
echo.

call conda activate exo-lstm-gpu

REM Full dataset processing
python build_windows_from_groundtruth.py ^
  --data_dir "E:\lilith4_sector-1_groundtruth\sector-1\ground-truth" ^
  --output_dir "data/windows_sector1_full" ^
  --seq_len 2048 ^
  --n_windows 3 ^
  --seed 42

echo.
echo Dataset building complete!
echo Output saved to: data/windows_sector1_full
