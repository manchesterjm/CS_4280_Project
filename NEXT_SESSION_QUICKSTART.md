# Next Session Quick Start Guide

## Last Session: December 2, 2025 (Afternoon)

### Status: NEW SYSTEM SETUP COMPLETE

**New System**: iBuyPower RDY Y70 B01
- **GPU**: RTX 5070 Ti (16 GB VRAM) - **VERIFIED WORKING**
- **PyTorch**: 2.10.0.dev (nightly with CUDA 12.8 for Blackwell architecture)
- **Conda env**: `exo-lstm-gpu` - **INSTALLED**

---

## What Was Completed

1. **Miniconda installed** and ToS accepted
2. **exo-lstm-gpu environment created** with Python 3.11
3. **PyTorch nightly installed** (required for RTX 5070 Ti / Blackwell sm_120)
4. **All dependencies installed**: numpy, pandas, scikit-learn, optuna, matplotlib, astropy, scipy, tqdm
5. **GPU verified working**: CUDA available, 16 GB VRAM detected
6. **Data located**: `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\`
7. **Batch size benchmark completed**:
   - **Optimal batch size: 136** (1.75 min/epoch)
   - 1.8× faster than old system
8. **optuna_optimize.py updated**: `HIGH_VRAM_GPU = True`

---

## What To Do Next

### Step 1: Run Optuna Optimization (~10-15 hours)

```powershell
# Activate environment
C:\Users\manch\miniconda3\Scripts\conda.exe activate exo-lstm-gpu

# Navigate to code directory
cd D:\CS_4280_Project\Code

# Run Optuna (can run overnight)
C:\Users\manch\miniconda3\envs\exo-lstm-gpu\python.exe optuna_optimize.py `
  --windows_dir "D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train" `
  --n_trials 30 `
  --epochs_per_trial 30 `
  --output_dir "optuna_results_sector1_5070ti"
```

**Expected time**: 10-15 hours (run overnight)

### Step 2: Train Final Model with Best Hyperparameters

After Optuna completes, check `optuna_results_sector1_5070ti/best_params_*.json` for optimal hyperparameters, then:

```powershell
C:\Users\manch\miniconda3\envs\exo-lstm-gpu\python.exe train_bilstm_cluster.py `
  --windows_dir "D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train" `
  --n_clusters 5 `
  --epochs 60 `
  --batch_size 136 `
  --lr 0.0001 `
  --hidden 256 `
  --layers 4 `
  --dropout 0.311 `
  --pos_weight 7.41 `
  --save_dir "runs\sector1_5070ti_final" `
  --amp_dtype fp16 `
  --num_workers 0 `
  --seed 42
```

**Expected time**: ~1.75 hours (60 epochs × 1.75 min/epoch)

### Step 3: Evaluate on Test Set

```powershell
C:\Users\manch\miniconda3\envs\exo-lstm-gpu\python.exe inference_cluster_model.py `
  --model_path "runs\sector1_5070ti_final\best.pt" `
  --windows_dir "D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\test" `
  --output_file "reports\sector1_5070ti_predictions.csv"
```

---

## Key Parameters (VERIFIED Dec 2, 2025)

| Parameter | Value | Source |
|-----------|-------|--------|
| **batch_size** | **136** | Benchmarked (optimal for RTX 5070 Ti) |
| lr | 0.0001 | Stable (0.000225 caused NaN crash) |
| pos_weight | 7.41 | Calculated (23325/3147) |
| hidden | 256 | From previous Optuna |
| layers | 4 | From previous Optuna |
| dropout | 0.311 | From previous Optuna |
| epochs | 60 | Standard |

---

## Data Locations (New System)

| Data | Location |
|------|----------|
| Training windows | `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train\` |
| Test windows | `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\test\` |
| Ground truth source | `D:\lilith4_sector-1_groundtruth\sector-1\ground-truth` |
| Code | `D:\CS_4280_Project\Code\` |

---

## Benchmark Results Summary

### RTX 5070 Ti (16 GB VRAM)

| Batch Size | Est. Epoch Time | Notes |
|------------|-----------------|-------|
| 64 | 3.05 min | |
| 128 | 1.78 min | |
| **136** | **1.75 min** | **OPTIMAL** |
| 144 | 1.78 min | |
| 256 | 15.02 min | Memory pressure |
| 512 | OOM | Crash |

**60 epochs = ~1.75 hours** (vs 3.14 hours on RTX 3060 Ti)

---

## Timeline to Final Presentation (Dec 9-11)

| Date | Task | Est. Time |
|------|------|-----------|
| ~~Dec 2~~ | ~~System setup + benchmark~~ | ✅ DONE |
| **Dec 2-3** | Run Optuna overnight | 10-15 hrs |
| **Dec 3-4** | Final model training | 1.75 hrs |
| **Dec 4-5** | Generate figures + update paper | 4-6 hrs |
| **Dec 5-6** | Create slides + record 20s demo | 3-4 hrs |
| **Dec 7-8** | Practice presentation | Buffer |
| **Dec 9-11** | **Presentations** | |
| **Dec 18** | **Final submission deadline** | |

---

## Key Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project instructions |
| `NEXT_SESSION_QUICKSTART.md` | This file - quick start guide |
| `PROGRESS_LOG_DEC_2_2025.md` | Today's session log |
| `Code/BATCH_SIZE_BENCHMARK_RESULTS.md` | Benchmark results (both GPUs) |
| `Code/optuna_optimize.py` | Hyperparameter optimization (HIGH_VRAM_GPU=True) |

---

## Python Executable Path (Important!)

Since conda isn't in PATH, use full path:
```
C:\Users\manch\miniconda3\envs\exo-lstm-gpu\python.exe
```

Or activate via:
```powershell
C:\Users\manch\miniconda3\Scripts\conda.exe activate exo-lstm-gpu
```

---

**Last updated**: December 2, 2025 (afternoon)
