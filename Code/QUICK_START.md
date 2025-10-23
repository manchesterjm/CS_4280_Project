# Quick Start Guide - Fixed Exoplanet Training

## What Was Wrong

1. **Windows Multiprocessing Bug**: PyTorch DataLoader with `num_workers > 0` crashes on Windows
2. **Class Imbalance**: 150 planets vs 505 non-planets (23% positive) - model couldn't learn properly
3. **Training on Wrong Data**: Original model trained only on clean planet signals, never saw flares/noise

## What's Fixed

1. ✅ `num_workers=0` - No more multiprocessing crashes
2. ✅ `pos_weight=3.37` - Weighted loss handles class imbalance
3. ✅ Lower learning rate (`1e-4`) - More stable training
4. ✅ Gradient clipping - Prevents exploding gradients
5. ✅ Better monitoring - Clear metrics at each epoch

## Files You Got

1. **diagnose_data.py** - Check your data quality before training
2. **train_lstm_fixed.py** - Complete fixed training script
3. **run_training.ps1** - Automated workflow script

## How to Use

### Option A: Automated (Easiest)

```powershell
# Make sure you're in your environment
conda activate exo-lstm-gpu

# Navigate to where you downloaded the files
cd C:\Users\manch\Downloads  # or wherever you saved them

# Run the automated script
.\run_training.ps1
```

### Option B: Manual (Step by Step)

```powershell
# 1. Copy files to your Code directory
Copy-Item "diagnose_data.py" "C:\CS_4280_Project\Code\"
Copy-Item "train_lstm_fixed.py" "C:\CS_4280_Project\Code\"

# 2. Navigate to Code directory  
cd C:\CS_4280_Project\Code

# 3. Check your data first
python diagnose_data.py

# 4. If data looks good, train
python train_lstm_fixed.py `
  --windows_dir "C:\CS_4280_Project\Code\data\windows_train" `
  --epochs 40 `
  --batch_size 256 `
  --lr 1e-4 `
  --hidden 128 `
  --layers 2 `
  --dropout 0.3 `
  --save_dir "C:\CS_4280_Project\Code\runs\lstm_fixed" `
  --amp_dtype fp16 `
  --pos_weight 3.37 `
  --num_workers 0
```

## What to Expect

### During Diagnostics:
```
✓ Found all required files
  Positive (planets):     150 (22.9%)
  Negative (non-planets): 505 (77.1%)
  Recommended pos_weight: 3.367
✅ Data looks good! Ready for training.
```

### During Training:
```
[epoch  1/40] loss=0.6234 val={'auc': 0.65, 'f1': 0.42} dt=25.1s
[epoch  2/40] loss=0.5891 val={'auc': 0.71, 'f1': 0.48} dt=24.8s
[epoch  5/40] loss=0.4523 val={'auc': 0.78, 'f1': 0.58} dt=24.9s
[best] improved to 0.78; saved
```

### Success Indicators:
- ✅ Loss decreases steadily (1.1 → 0.4)
- ✅ AUC improves (0.5 → 0.75+)
- ✅ F1 score increases (0.35 → 0.6+)
- ✅ No crashes, training completes

### Warning Signs:
- ❌ Loss stays flat or increases
- ❌ AUC stuck around 0.5
- ❌ NaN/Inf in metrics
- ❌ Crashes during training

## Troubleshooting

### If diagnostics shows NaN/Inf values:
```powershell
# Your data has problems - need to rebuild windows
# Check the build_windows script for bad data
```

### If training crashes:
```powershell
# Make sure num_workers=0
# Check GPU memory with: nvidia-smi
# Try smaller batch_size (128 instead of 256)
```

### If AUC doesn't improve:
```powershell
# Try higher pos_weight (5.0 or 10.0)
# Try longer training (60-80 epochs)
# Check if labels are correct in manifest.csv
```

### If out of GPU memory:
```powershell
# Reduce batch_size to 128 or 64
# Reduce hidden size to 64
# Use fp32 instead of fp16
```

## Key Parameters Explained

- `--windows_dir`: Where your training data (X.npy, y.npy) is
- `--pos_weight 3.37`: How much more to weight positive examples (505/150)
- `--num_workers 0`: CRITICAL for Windows - prevents multiprocessing crash
- `--lr 1e-4`: Learning rate (lower = more stable)
- `--amp_dtype fp16`: Mixed precision for faster training on GPU

## After Training

Your model will be saved in:
- `C:\CS_4280_Project\Code\runs\lstm_fixed\best.pt` - Best model (highest AUC)
- `C:\CS_4280_Project\Code\runs\lstm_fixed\last.pt` - Last epoch
- `C:\CS_4280_Project\Code\runs\lstm_fixed\config.json` - All settings used

## Next Steps

Once training completes successfully:
1. Test on your inference dataset
2. Analyze false positives/negatives
3. Fine-tune hyperparameters
4. Consider ensemble models

Good luck! 🚀
