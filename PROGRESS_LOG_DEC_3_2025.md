# Progress Log - December 3, 2025

## Session Overview

**Date**: December 3, 2025
**Session**: Overnight Optuna run + morning check-in
**GPU**: RTX 5070 Ti (16 GB VRAM)

---

## What Happened

### 1. Updated Optuna Batch Size Search Space

**Problem**: The original Optuna search space had batch sizes `[64, 128, 192, 256]`, but our benchmark showed:
- Batch 256 causes **memory pressure** (8.67 sec/batch vs 0.54 sec for batch 136)
- Optimal batch size is **136** (1.75 min/epoch)

**Fix Applied**: Updated `optuna_optimize.py` line 238-239:
```python
# Before:
batch_size = trial.suggest_categorical('batch_size', [64, 128, 192, 256])

# After:
batch_size = trial.suggest_categorical('batch_size', [112, 128, 136, 144])
```

### 2. Started Optuna Optimization (~11:01 PM Dec 2)

**Command**:
```powershell
python optuna_optimize.py `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train `
  --n_trials 30 `
  --epochs_per_trial 30 `
  --output_dir optuna_results_sector1_5070ti
```

**Configuration**:
- 30 trials × 30 epochs per trial
- Batch sizes: [112, 128, 136, 144]
- Hidden sizes: [128, 256, 512]
- Learning rate: 1e-5 to 1e-3 (log scale)
- Dropout: 0.2 to 0.5
- Layers: 2-4
- Clusters: [3, 5, 7, 10]

### 3. Morning Status Check (~7:00 AM Dec 3)

**Results**:
- **Trial 0 completed**: AUC = **0.9160** (excellent!)
- **Trial 1 in progress** (at ~7 hours elapsed)
- GPU utilization: 100%
- GPU memory: 15.6 GB / 16.3 GB (96% used)

**Problem Identified**: Each trial takes ~52 minutes (30 epochs × 1.75 min/epoch). At this rate:
- 30 trials = ~26 hours total
- Only ~8 trials would complete overnight

### 4. Stopped Optuna Run (~7:00 AM Dec 3)

**Reason**: User needed GPU for other work.

**What Was Lost**:
- Optuna results were not saved (saves only at end of optimization)
- Trial 0's hyperparameters were not captured
- However, we know Trial 0 achieved **AUC 0.916**

---

## Key Finding

**AUC 0.916 on Trial 0** is a major improvement over previous results:
- Previous best (old dataset): AUC 0.7572
- Previous Sector 1 test: AUC 0.893
- **New Optuna Trial 0: AUC 0.916** (+2.3% over previous Sector 1 result)

This suggests the Sector 1 dataset with proper hyperparameters can achieve very strong results.

---

## Current State

### Files Modified
| File | Change |
|------|--------|
| `Code/optuna_optimize.py` | Updated batch_size search to [112, 128, 136, 144] |

### Results Folder
- `Code/optuna_results_sector1_5070ti/` - **Empty** (no results saved)

### GPU Status
- **Available**: GPU freed after stopping Optuna
- Utilization: ~1%
- Memory: ~1.2 GB (system overhead only)

---

## What Still Needs To Be Done

### Option A: Resume Optuna (Faster Settings)
Run with fewer trials and epochs to complete faster:

```powershell
cd D:\CS_4280_Project\Code
conda activate exo-lstm-gpu
python optuna_optimize.py `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train `
  --n_trials 15 `
  --epochs_per_trial 20 `
  --output_dir optuna_results_sector1_5070ti
```

**Estimated time**: 15 trials × ~35 min = **~9 hours**

### Option B: Skip Optuna, Train Final Model
Since Trial 0 already achieved AUC 0.916, we could train a final model with reasonable hyperparameters:

```powershell
cd D:\CS_4280_Project\Code
conda activate exo-lstm-gpu
python train_bilstm_cluster.py `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train `
  --n_clusters 5 `
  --epochs 60 `
  --batch_size 136 `
  --lr 0.0001 `
  --hidden 256 `
  --layers 4 `
  --dropout 0.311 `
  --pos_weight 7.41 `
  --save_dir runs\sector1_final `
  --amp_dtype fp16 `
  --num_workers 0 `
  --seed 42
```

**Estimated time**: ~1.75 hours (60 epochs × 1.75 min/epoch)

### After Training
1. Evaluate on test set
2. Generate figures (ROC curve, confusion matrix)
3. Update paper with final results
4. Record 20-second demo video
5. Prepare presentation slides

---

## Timeline

| Date | Task | Status |
|------|------|--------|
| Dec 2 (PM) | New PC setup + benchmark | ✅ Done |
| Dec 2-3 (overnight) | Optuna run | ⚠️ Interrupted (1/30 trials) |
| **Dec 3** | Resume Optuna OR train final model | **Pending** |
| Dec 4-5 | Generate figures + update paper | Pending |
| Dec 5-6 | Create slides + record demo | Pending |
| Dec 7-8 | Practice presentation | Buffer |
| **Dec 9-11** | **Presentations** | |
| **Dec 18** | **Final submission deadline** | |

---

## Benchmark Reference (RTX 5070 Ti)

| Batch Size | Est. Epoch Time | Notes |
|------------|-----------------|-------|
| 112 | 2.00 min | |
| 128 | 1.78 min | |
| **136** | **1.75 min** | **OPTIMAL** |
| 144 | 1.78 min | |
| 256 | 15.02 min | Memory pressure - AVOID |

---

**Last Updated**: December 3, 2025, ~7:00 AM
**Status**: Optuna stopped, GPU available, ready to resume or train final model
