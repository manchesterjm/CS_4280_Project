# Progress Log - November 27, 2025

## Session Summary
**Date/Time**: 2025-11-27 ~3:00 PM (ongoing)
**Status**: Training in progress (background process)
**Duration**: ~1 hour so far

---

## What We Did Today

### 1. Dataset Build (Completed)
Successfully built the TESS Sector 1 training dataset:

**Command used:**
```powershell
cd C:\CS_4280_Project\Code
conda activate exo-lstm-gpu
python build_sector1_dataset_v2.py `
  --data_dir "E:\lilith4_sector-1_groundtruth\sector-1\ground-truth" `
  --output_dir "data\windows_sector1_full" `
  --seq_len 2048 `
  --n_windows 3 `
  --seed 42
```

**Results:**
- Files processed: 11,017
- Processing time: ~27 seconds
- Parallel workers: 11 (left 1 core free to prevent crash)

**Training Set** (`data/windows_sector1_full/train`):
- Windows: 26,472
- Positive (planets): 3,147 (11.9%)
- Negative (non-planets): 23,325 (88.1%)

**Test Set** (`data/windows_sector1_full/test`):
- Windows: 6,579
- Positive (planets): 732 (11.1%)
- Negative (non-planets): 5,847 (88.9%)

**Labels**: Verified using `tsop301_sector1_groundtruth_master.csv` (supervised training)

---

### 2. CPU Crash Prevention (Completed)
Added CPU thread limits to prevent system overload:

**Files modified:**
1. `build_sector1_dataset_v2.py`
   - Added: `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, etc.
   - Changed: `n_jobs = cpu_count() - 1` (leaves 1 core free)

2. `train_bilstm_cluster.py`
   - Added: Same CPU thread limits at top of file

3. `autonomous_training_pipeline.py`
   - Added: Same CPU thread limits

**Why this works:**
- Before: 12 workers × 12 threads = 144 threads for 12 cores → CRASH
- After: 11 workers × 2 threads = 22 threads for 12 cores → STABLE

---

### 3. Clustering Fix (Completed but not yet applied)
Discovered and fixed a clustering issue:

**Problem:**
- 99.96% of windows assigned to cluster 0
- Clusters 1-4 had only 1-7 windows each (outliers)
- Caused by extreme variance values (std up to 99,690)

**Root cause:**
- StandardScaler is sensitive to outliers
- A few windows have extremely high variance values
- K-means clusters these outliers separately

**Fix applied to `train_bilstm_cluster.py`:**
```python
# Robust preprocessing: clip outliers to 1st-99th percentile
for i in range(features.shape[1]):
    col = features[:, i]
    p1, p99 = np.percentile(col, [1, 99])
    features[:, i] = np.clip(col, p1, p99)
```

**Note:** This fix will take effect in the NEXT training run. Current training uses old clustering.

---

### 4. Training Started (In Progress)

**Background Process ID:** c34105

**Command:**
```powershell
python train_bilstm_cluster.py `
  --windows_dir "data\windows_sector1_full\train" `
  --n_clusters 5 `
  --epochs 60 `
  --batch_size 128 `
  --lr 0.000225 `
  --hidden 256 `
  --layers 4 `
  --dropout 0.311 `
  --pos_weight 7.41 `
  --save_dir "runs\sector1_experiments\baseline_corrected_posweight" `
  --amp_dtype fp16 `
  --num_workers 0 `
  --seed 42
```

**Key hyperparameters:**
- pos_weight: 7.41 (corrected from 3.367, based on actual class ratio 23325/3147)
- Epochs: 60
- Batch size: 128
- Learning rate: 0.000225 (from Optuna optimization)
- Hidden: 256, Layers: 4 (from Optuna)

**Estimated time:** ~20 hours for 60 epochs

**Progress as of last check:**
- ~31% of epoch 1 complete
- Loss fluctuating around 1.1-1.4 (normal)
- ~7 seconds per batch, 176 batches per epoch

---

## Current Training Output Structure

```
runs/sector1_experiments/baseline_corrected_posweight/
├── best.pt          # Best checkpoint (saved when val AUC improves)
├── last.pt          # Last epoch checkpoint
├── config.json      # Training configuration
└── cluster_ids.npy  # Cluster assignments
```

---

## Issues to Watch

### 1. Clustering Not Effective (Current Run)
- 99.96% in cluster 0 - essentially no clustering benefit
- Model will still learn, but won't benefit from cluster specialization
- Fixed for future runs

### 2. Long Training Time
- 60 epochs × 20 min/epoch = 20 hours
- Consider reducing to 15-20 epochs for faster experiments
- Early stopping with patience=15 may help

### 3. Class Imbalance (Addressed)
- 11.9% positive rate is heavily imbalanced
- Using pos_weight=7.41 to compensate
- Watch for precision vs recall tradeoff

---

## Next Steps

### Immediate (When Training Completes or Checkpointed)
1. Check validation AUC from training output
2. If AUC > 0.65: approach is promising
3. If AUC < 0.55: need to reconsider approach

### Short-term (Next Session)
1. Start experiment with fixed clustering (15 epochs)
   ```powershell
   python train_bilstm_cluster.py `
     --windows_dir "data\windows_sector1_full\train" `
     --n_clusters 5 `
     --epochs 15 `
     --batch_size 128 `
     --lr 0.000225 `
     --hidden 256 `
     --layers 4 `
     --dropout 0.311 `
     --pos_weight 7.41 `
     --save_dir "runs\sector1_experiments\fixed_clustering" `
     --amp_dtype fp16 `
     --num_workers 0 `
     --seed 42
   ```

2. Compare results with and without proper clustering

### Long-term (If Results Promising)
1. Run Optuna hyperparameter optimization overnight
   ```powershell
   python optuna_optimize.py `
     --windows_dir "data\windows_sector1_full\train" `
     --n_trials 30 `
     --epochs_per_trial 20
   ```
   Expected time: 8-12 hours

2. Test best model on test set (6,579 windows)

3. Generate figures for RNN paper/presentation

---

## Performance Targets

| Metric | Target | Previous Best | Notes |
|--------|--------|---------------|-------|
| AUC | >0.70 | 0.7572 | Primary metric |
| F1 | >0.40 | 0.455 | Balance precision/recall |
| Recall | >0.80 | 0.86 | Detect most planets |
| Precision | >0.15 | 0.31 | Acceptable FP rate |

---

## Crash Recovery Instructions

### If Claude Code session crashes:

1. **Check if training is still running:**
   ```powershell
   Get-Process python
   # or
   nvidia-smi  # Check GPU utilization
   ```

2. **Check training output:**
   ```powershell
   dir C:\CS_4280_Project\Code\runs\sector1_experiments\baseline_corrected_posweight\
   type C:\CS_4280_Project\Code\runs\sector1_experiments\baseline_corrected_posweight\config.json
   ```

3. **Resume from checkpoint (if training stopped):**
   - Check `last.pt` for latest checkpoint
   - Currently no resume functionality - would need to restart from scratch
   - Best to let training complete before starting new experiments

4. **Start new training (if needed):**
   - Use commands from "Next Steps" section above
   - Clustering fix is already in `train_bilstm_cluster.py`

---

## Files Created/Modified Today

### Created
- `Code/autonomous_experiment_runner.py` - Batch experiment runner (not used due to conda issues)
- `Code/check_features.py` - Quick script to check feature distribution
- `Code/TRAINING_STATUS.md` - Quick status reference
- `PROGRESS_LOG_NOV_27_2025.md` - This file

### Modified
- `Code/build_sector1_dataset_v2.py` - Added CPU limits
- `Code/train_bilstm_cluster.py` - Added CPU limits + clustering fix
- `Code/autonomous_training_pipeline.py` - Added CPU limits

### Generated Data
- `Code/data/windows_sector1_full/train/` - 26,472 training windows
- `Code/data/windows_sector1_full/test/` - 6,579 test windows

---

## Environment Details

- **OS**: Windows 11
- **GPU**: CUDA-enabled (check with `nvidia-smi`)
- **Conda environment**: `exo-lstm-gpu`
- **Python**: Via miniconda3
- **CPU**: 6 cores, 12 logical processors

---

## Key Reference Files

- **Project instructions**: `CLAUDE.md`
- **Previous progress**: `PROGRESS_LOG_NOV_14_2025.md`
- **Quick reference**: `SATURDAY_QUICKSTART.md`
- **Training script**: `Code/train_bilstm_cluster.py`
- **Dataset builder**: `Code/build_sector1_dataset_v2.py`

---

**Last updated**: 2025-11-27 ~3:15 PM
**Training status**: In progress (epoch 1, ~31% complete)
