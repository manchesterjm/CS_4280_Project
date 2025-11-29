# Next Session Quick Start Guide

## FIRST: Check Current Time
Run `Get-Date` to see how much time has passed since the overnight run started.

**Training started**: November 28, 2025 at ~10:00 PM MST (user will start manually)
**Expected completion**: ~8 hours later (25 epochs x ~20 min/epoch)

---

## Last Session: November 28, 2025 (5:39 AM MST)

### Ground Truth Clustering - MAJOR IMPROVEMENT

| Metric | Before (Flux Stats) | After (Ground Truth) | Change |
|--------|---------------------|----------------------|--------|
| AUC | 0.9051 | **0.9159** | +1.2% |
| F1 | 0.5891 | **0.5673** | -3.7% |
| Clustering | 99.96% cluster 0 | **Balanced 5 clusters** | FIXED |

**Key Fix**: Now using actual injected transit parameters (Period, Depth, Duration, Teff) from LILITH-4 ground truth files instead of noisy flux statistics.

### What Was Done
1. Created `merge_groundtruth_params.py` to parse ground truth files from `E:\lilith4_sector-1_groundtruth\sector-1\ground-truth\`
2. Merged 53.6% of training windows with actual transit parameters
3. Updated `train_bilstm_cluster.py` to use ground truth features for clustering
4. Fixed learning rate: **0.0001** (was 0.000225 which caused NaN crash overnight)

---

## Tonight's Overnight Run

### Command (copy-paste into PowerShell):

```powershell
cd C:\CS_4280_Project\Code && conda activate exo-lstm-gpu && python train_bilstm_cluster.py --windows_dir "data\windows_sector1_full\train" --n_clusters 5 --epochs 25 --batch_size 128 --lr 0.0001 --hidden 256 --layers 4 --dropout 0.311 --save_dir "runs\sector1_groundtruth_overnight" --amp_dtype fp16 --pos_weight 7.41 --num_workers 0 --seed 42
```

**Key Parameters:**
- `lr 0.0001` - Lower learning rate prevents NaN crash
- `epochs 25` - ~8 hours overnight
- `pos_weight 7.41` - Correct class imbalance ratio (23325/3147)
- Ground truth clustering automatically detected from metadata

---

## After Training Completes

### Test on held-out test set:
```powershell
cd C:\CS_4280_Project\Code
conda activate exo-lstm-gpu

python inference_cluster_model.py `
  --model_path "runs\sector1_groundtruth_overnight\best.pt" `
  --windows_dir "data\windows_sector1_full\test" `
  --output_file "reports\sector1_groundtruth_predictions.csv"
```

---

## Key Files

| File | Purpose |
|------|---------|
| `Code/train_bilstm_cluster.py` | Training script (ground truth clustering added) |
| `Code/merge_groundtruth_params.py` | Merges LILITH-4 ground truth into metadata |
| `Code/data/windows_sector1_full/train/meta.csv` | Training metadata with ground truth params |

---

## Important: Learning Rate Fix

Previous overnight run crashed with NaN loss because `lr=0.000225` was too high.

**Use `lr=0.0001` for stable training.**

---

**Last updated**: November 28, 2025
