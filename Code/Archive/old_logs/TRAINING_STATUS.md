# Training Status Report - December 3, 2025

## Current Status: PAPER COMPLETE, AWAITING NEXT TRAINING RUN

### Best Results Achieved

| Dataset | Model | AUC | Status |
|---------|-------|-----|--------|
| Old (655 windows) | BiLSTM+Cluster | 0.7572 | Archived |
| Sector 1 (Nov training) | BiLSTM+Cluster | 0.893 | Current best saved |
| **Sector 1 (Optuna Trial 0)** | BiLSTM+Cluster | **0.916** | Not saved (interrupted) |

**Key Finding**: AUC 0.916 is achievable on the Sector 1 dataset with the right hyperparameters!

---

## Dataset: TESS Sector 1 Ground Truth

**Location**: `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\`

| Split | Windows | Planets | Non-Planets | % Positive |
|-------|---------|---------|-------------|------------|
| Train | 26,472 | 3,147 | 23,325 | 11.9% |
| Test | 6,579 | 771 | 5,808 | 11.7% |
| **Total** | **33,051** | **3,918** | **29,133** | **11.8%** |

---

## Optimal Hyperparameters (RTX 5070 Ti)

Based on benchmarking and previous Optuna runs:

| Parameter | Value | Notes |
|-----------|-------|-------|
| **batch_size** | **136** | Optimal for RTX 5070 Ti (1.75 min/epoch) |
| lr | 0.0001 | Stable (0.000225 caused NaN crash) |
| **pos_weight** | **7.41** | Calculated from 23325/3147 |
| hidden | 256 | From previous Optuna |
| layers | 4 | From previous Optuna |
| dropout | 0.311 | From previous Optuna |
| epochs | 60 | Standard |
| n_clusters | 5 | Default |

**WARNING**: Do NOT use batch_size 256 (causes 15+ min/epoch due to memory pressure)

---

## Next Training Run Options

### Option A: Resume Optuna Optimization (~9 hours)

```powershell
cd D:\CS_4280_Project\Code
conda activate exo-lstm-gpu

python optuna_optimize.py `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train `
  --n_trials 15 `
  --epochs_per_trial 20 `
  --output_dir optuna_results_sector1_5070ti
```

### Option B: Train Final Model (~1.75 hours)

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

---

## After Training: Evaluation Pipeline

### 1. Run Inference on Test Set

```powershell
python inference_cluster_model.py `
  --model_path runs\sector1_final\best.pt `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\test `
  --output_file reports\sector1_final_predictions.csv
```

### 2. Generate Figures for Paper

```powershell
python generate_sector1_figures.py
```

Output: `term_project_files/Images/RNN/`
- `bilstm_sector1_roc_curve.png`
- `confusion_matrix.png`

---

## Issues Fixed (Historical)

### Clustering Problem (Fixed Nov 27)
- **Issue**: 99.96% of windows assigned to cluster 0
- **Cause**: Extreme outliers in variance features
- **Fix**: Added percentile clipping (1-99%) before StandardScaler

### Class Imbalance (Fixed Nov 27)
- **Issue**: Original pos_weight was 3.367
- **Fix**: Corrected to 7.41 (actual ratio 23325/3147)

### Batch Size Optimization (Fixed Dec 2)
- **Issue**: Batch 256 caused memory pressure (15+ min/epoch)
- **Fix**: Optimal batch_size = 136 (1.75 min/epoch)

---

## Timeline

| Date | Task | Status |
|------|------|--------|
| Nov 14 | Pivoted to Sector 1 dataset | ✅ Done |
| Nov 27 | Dataset built (33,051 windows) | ✅ Done |
| Nov 29 | Batch size benchmark | ✅ Done |
| Dec 2 | New PC setup (RTX 5070 Ti) | ✅ Done |
| Dec 2-3 | Optuna overnight | ⚠️ Interrupted (Trial 0 = AUC 0.916) |
| Dec 3 | Paper finalization | ✅ Done |
| **Dec 4-5** | **Final training** | **NEXT** |
| Dec 9-11 | Presentations | |
| Dec 18 | Final submission | |

---

**Last Updated**: December 3, 2025, afternoon
**Status**: Ready for next training run
