# Progress Log - December 5, 2025 (Evening Session)

## Session Overview

**Date**: December 5, 2025 (9:30 PM - overnight)
**GPU**: RTX 5070 Ti (16 GB VRAM)
**Goal**: Run Optuna with optimized search space [128, 192] to find hyperparameters matching/beating Trial 0's AUC 0.9142

---

## Starting State

**Best result from morning session**:
- Trial 0: **AUC 0.9142** with hidden_size=256, batch_size=112, num_layers=4, dropout=0.38, lr=0.00026

**Optimized search space** (from morning benchmarking):
```python
hidden_size = [128, 192]  # Excluded 256+ for speed
batch_size = [96, 112, 128]
```

**Expected trial times**:
- hidden_size=128: ~7 min/trial
- hidden_size=192: ~29 min/trial
- Average: ~18 min/trial

**Overnight window**: 9:30 PM → 5:00 AM = 7.5 hours = 450 minutes
**Expected trials**: 15-25 trials

---

## Timeline of Events

### 9:30 PM - Session Start
- Read CLAUDE.md, NEXT_SESSION_QUICKSTART.md, PROGRESS_LOG_DEC_5_2025.md
- Confirmed search space is [128, 192] (already updated)
- Starting Optuna run

### 9:35 PM - Optuna Started
- Command: `python optuna_optimize.py --windows_dir "D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train" --n_trials 30 --epochs_per_trial 20 --output_dir "D:\CS_4280_Project\Code\optuna_overnight_dec5"`
- Running in background
- Results saved to: `optuna_overnight_dec5/intermediate_results.json`

---

## Live Updates

| Time | Trial | AUC | hidden_size | batch_size | layers | dropout | n_clusters | Trial Time |
|------|-------|-----|-------------|------------|--------|---------|------------|------------|
| 9:43 PM | 0 | 0.9154 | 192 | 96 | 2 | 0.217 | 10 | 14.6 min |
| 9:54 PM | 1 | 0.9141 | 192 | 128 | 2 | 0.288 | 5 | 11.3 min |
| 10:09 PM | 2 | 0.9137 | 192 | 96 | 2 | 0.405 | 10 | 14.6 min |
| 10:13 PM | 3 | 0.9132 | 128 | 96 | 4 | 0.379 | 10 | 4.0 min |
| 10:39 PM | **4** | **0.9164** | 192 | 112 | 4 | 0.432 | 5 | 25.8 min |
| 10:41 PM | 5 | 0.9144 | 128 | 96 | 2 | 0.419 | 3 | 2.1 min |
| 10:46 PM | 6 | 0.9151 | 128 | 112 | 3 | 0.294 | 3 | 4.8 min |
| 10:49 PM | 7 | 0.9158 | 128 | 112 | 2 | 0.468 | 5 | 3.2 min |
| 11:11 PM | 8 | 0.9158 | 192 | 96 | 3 | 0.483 | 10 | 22.1 min |
| 11:13 PM | 9 | 0.8943 | 128 | 96 | 2 | 0.472 | 7 | PRUNED |
| 11:27 PM | 10 | 0.8796 | 192 | 112 | 4 | 0.324 | 5 | PRUNED |
| 11:37 PM | 11 | 0.8818 | 192 | 128 | 3 | 0.489 | 7 | PRUNED |
| 12:02 AM | 12 | 0.9161 | 192 | 112 | 4 | 0.428 | 5 | 25.9 min |
| 12:17 AM | 13 | 0.8805 | 192 | 112 | 4 | 0.414 | 5 | PRUNED (low lr) |
| 12:18 AM | 14 | 0.0000 | 192 | 112 | 4 | 0.368 | 5 | NaN (aborted) |
| 12:32 AM | 15 | 0.9106 | - | - | - | - | - | PRUNED |
| 12:43 AM | 16 | 0.8813 | - | - | - | - | - | PRUNED |
| 1:08 AM | **17** | **0.9171** | 192 | 112 | 4 | 0.349 | 5 | 25.8 min |
| 1:31 AM | 18 | 0.9162 | 192 | 128 | 4 | 0.334 | 7 | 23.0 min |
| 1:51 AM | 19 | 0.9153 | 192 | 112 | 3 | 0.258 | 3 | 19.4 min |
| 2:08 AM | 20 | 0.9136 | - | - | - | - | - | PRUNED |
| 2:30 AM | **21** | **0.9182** | 192 | 128 | 4 | 0.334 | 7 | 22.9 min |
| 2:46 AM | 22 | 0.9154 | 192 | 128 | 4 | 0.341 | 7 | 16.0 min |
| 2:49 AM | 23 | 0.0000 | 192 | 128 | 4 | 0.308 | 7 | NaN (aborted) |
| 2:58 AM | 24 | 0.9110 | - | - | - | - | - | PRUNED |
| 3:13 AM | 25 | 0.9131 | - | - | - | - | - | PRUNED |
| 3:19 AM | 26 | 0.9153 | 128 | 112 | 4 | 0.393 | 7 | 6.2 min |
| 3:32 AM | 27 | 0.8488 | - | - | - | - | - | PRUNED |
| 3:43 AM | 28 | 0.9132 | - | - | - | - | - | PRUNED |
| 4:06 AM | 29 | 0.9173 | 192 | 128 | 4 | 0.217 | 10 | 22.9 min |

---

## OPTIMIZATION COMPLETE!

**Finished**: December 6, 2025, 4:06 AM
**Total Time**: 397.1 minutes (~6.6 hours)
**Total Trials**: 30 (17 complete, 13 pruned)

---

## FINAL RESULTS

| Metric | Value |
|--------|-------|
| **Best Trial** | **21** |
| **Best AUC** | **0.9182** (91.82%) |
| Improvement vs morning baseline (0.9142) | **+0.40%** |
| Improvement vs original model (0.7572) | **+16.1%** |

---

## BEST HYPERPARAMETERS (Trial 21)

```python
hidden_size = 192
batch_size = 128
num_layers = 4
dropout = 0.334
lr = 0.0000497
n_clusters = 7
weight_decay = 0.0000154
cluster_embed_dim = 64
```

**Key findings**:
- More clusters (7) is better than fewer (5)
- Larger cluster embeddings (64) improve performance
- Moderate dropout (0.334) works well
- Lower learning rate (~5e-5) with 4 layers converges best

---

## Files Saved

| File | Description |
|------|-------------|
| `optuna_overnight_dec5/best_params_20251206_040603.json` | Best hyperparameters |
| `optuna_overnight_dec5/trials_20251206_040603.csv` | All trial data |
| `optuna_overnight_dec5/optuna_study_20251206_040603.pkl` | Full Optuna study |
| `optuna_overnight_dec5/intermediate_results.json` | Real-time progress |

---

## Next Steps (Morning of Dec 6)

1. **Train final model** with Trial 21 hyperparameters (60-80 epochs)
2. **Evaluate on test set**
3. **Generate figures** for paper/presentation
4. **Update CLAUDE.md** with new best results

**Training command**:
```powershell
python train_bilstm_cluster.py --windows_dir "D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train" --n_clusters 7 --epochs 60 --batch_size 128 --lr 0.0000497 --hidden 192 --layers 4 --dropout 0.334 --save_dir "runs\sector1_final_0918" --amp_dtype fp16 --pos_weight 7.41 --num_workers 0 --seed 42
```

---

**Last Updated**: December 6, 2025, 4:06 AM
**Status**: COMPLETE - Ready for final training

---

## Monitoring Commands

Check progress:
```powershell
# View intermediate results
type D:\CS_4280_Project\Code\optuna_overnight_dec5\intermediate_results.json

# Check if still running
tasklist | findstr python
```

---

## Key Files

| File | Purpose |
|------|---------|
| `Code/optuna_overnight_dec5/intermediate_results.json` | Real-time progress (updates after each trial) |
| `Code/optuna_overnight_dec5/best_params_*.json` | Final best hyperparameters |
| `Code/optuna_overnight_dec5/trials_*.csv` | All trial data |

---

## Morning Checklist (5:00 AM)

1. Check `intermediate_results.json` for completed trials
2. Compare best AUC to Trial 0 baseline (0.9142)
3. If better hyperparameters found, update NEXT_SESSION_QUICKSTART.md
4. Start final training with best hyperparameters

---

**Last Updated**: December 5, 2025, 9:35 PM
**Status**: Optuna running overnight
