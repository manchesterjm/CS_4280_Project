# Next Session Quick Start Guide

## Last Session: December 3, 2025 (Morning)

### Status: OPTUNA INTERRUPTED - READY TO RESUME

**What Happened**:
- Started Optuna overnight (30 trials × 30 epochs)
- **Trial 0 achieved AUC 0.916** (excellent result!)
- Stopped after ~7 hours (only 1 trial completed) - needed GPU for other work
- Results NOT saved (Optuna saves at end)

**Key Finding**: AUC 0.916 is achievable on Sector 1 dataset!

---

## Quick Resume Options

### Option A: Fast Optuna Run (~9 hours)

```powershell
# Activate environment
conda activate exo-lstm-gpu

# Navigate to code
cd D:\CS_4280_Project\Code

# Run Optuna with fewer trials
python optuna_optimize.py `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train `
  --n_trials 15 `
  --epochs_per_trial 20 `
  --output_dir optuna_results_sector1_5070ti
```

**Time**: ~15 trials × 35 min = ~9 hours

### Option B: Skip Optuna, Train Final Model (~1.75 hours)

Since we already saw AUC 0.916 is achievable, train with known-good hyperparameters:

```powershell
conda activate exo-lstm-gpu
cd D:\CS_4280_Project\Code

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

**Time**: ~1.75 hours (60 epochs × 1.75 min/epoch)

---

## Key Parameters (VERIFIED Dec 2-3, 2025)

| Parameter | Value | Source |
|-----------|-------|--------|
| **batch_size** | **136** | Benchmarked (optimal for RTX 5070 Ti) |
| lr | 0.0001 | Stable (0.000225 caused NaN crash) |
| pos_weight | 7.41 | Calculated (23325/3147) |
| hidden | 256 | Previous Optuna |
| layers | 4 | Previous Optuna |
| dropout | 0.311 | Previous Optuna |
| epochs | 60 | Standard |

---

## Data Locations

| Data | Location |
|------|----------|
| Training windows | `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train\` |
| Test windows | `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\test\` |
| Code | `D:\CS_4280_Project\Code\` |

**Training set**: 26,472 windows (3,147 planets = 11.9%)
**Test set**: 6,579 windows

---

## After Training: Evaluate & Generate Figures

### Step 1: Evaluate on Test Set

```powershell
python inference_cluster_model.py `
  --model_path runs\sector1_final\best.pt `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\test `
  --output_file reports\sector1_final_predictions.csv
```

### Step 2: Generate Figures

```powershell
python generate_sector1_figures.py
```

This creates ROC curve and confusion matrix for the paper.

---

## Benchmark Reference (RTX 5070 Ti)

| Batch Size | Est. Epoch Time | Notes |
|------------|-----------------|-------|
| 112 | 2.00 min | |
| 128 | 1.78 min | |
| **136** | **1.75 min** | **OPTIMAL** |
| 144 | 1.78 min | |
| 256 | 15.02 min | Memory cliff - AVOID |

---

## Timeline to Final Presentation

| Date | Task | Status |
|------|------|--------|
| ~~Dec 2~~ | System setup + benchmark | ✅ Done |
| ~~Dec 2-3~~ | Optuna overnight | ⚠️ Interrupted |
| **Dec 3-4** | Train final model OR resume Optuna | **NEXT** |
| Dec 4-5 | Generate figures + update paper | Pending |
| Dec 5-6 | Create slides + record 20s demo | Pending |
| Dec 7-8 | Practice presentation | Buffer |
| **Dec 9-11** | **Presentations** | |
| **Dec 18** | **Final submission** | |

---

## Files Changed Since Last Full Session

| File | Change |
|------|--------|
| `Code/optuna_optimize.py` | Batch sizes: [64,128,192,256] → [112,128,136,144] |
| `PROGRESS_LOG_DEC_3_2025.md` | Created - documents overnight run |

---

## Best Results So Far

| Dataset | Model | AUC | Notes |
|---------|-------|-----|-------|
| Old (655 windows) | BiLSTM+Cluster | 0.7572 | Previous best |
| Sector 1 (trained Nov) | BiLSTM+Cluster | 0.893 | Test set eval |
| **Sector 1 (Optuna Trial 0)** | BiLSTM+Cluster | **0.916** | New best (not saved) |

---

**Last Updated**: December 3, 2025, morning
**Status**: GPU available, ready to train
