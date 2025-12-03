# Progress Log - November 29, 2025

## Session Summary
**Date/Time**: 2025-11-29
**Status**: Batch size benchmarking completed
**Key Finding**: Batch size 64 is optimal (3.14 min/epoch), not 128

---

## What We Did Today

### 1. Batch Size Benchmark (Completed)

Created and ran `benchmark_batch_sizes.py` to test batch sizes 8, 16, 32, 64, and 128.

**Test Configuration:**
- GPU: NVIDIA GeForce RTX 3060 Ti
- Dataset: TESS Sector 1 Ground Truth (26,472 training samples)
- Test Duration: 30 seconds per batch size
- Model: BiLSTM + Clustering (256 hidden, 4 layers)

**Results:**

| Batch Size | Batches/Epoch | Time/Batch (sec) | Est. Epoch Time (min) |
|------------|---------------|------------------|----------------------|
| 8 | 3309 | 0.356 | 19.61 |
| 16 | 1655 | 0.304 | 8.38 |
| 32 | 828 | 0.362 | 5.00 |
| **64** | **414** | **0.455** | **3.14** |
| 128 | 207 | 7.089 | 24.46 |

**Key Findings:**
1. **Batch size 64 is optimal** - 3.14 min/epoch (fastest)
2. **Batch size 128 is slowest** - 24.46 min/epoch (GPU memory bottleneck)
3. **Batch size 32** - 5.00 min/epoch (previously thought optimal)
4. Smaller batches (8, 16) have too many iterations despite fast per-batch time

**Impact on Training Time:**
- With batch size 128: 60 epochs × 24.46 min = **24.5 hours**
- With batch size 64: 60 epochs × 3.14 min = **3.14 hours**
- **Speedup: 7.8× faster with batch size 64**

---

## Current State

### Dataset
- **Location**: `E:\CS_4280_Project_Backup\Code\data\windows_sector1_full\`
- **Training**: 26,472 windows (3,147 planets / 23,325 non-planets)
- **Test**: 6,579 windows
- **Source**: TESS Sector 1 Ground Truth

### Best Model So Far
- **Location**: `runs\sector1_groundtruth_overnight\` (if training completed)
- **AUC**: 0.9159 (from Nov 28 ground truth clustering)
- **Configuration**: batch_size=128, lr=0.0001, epochs=25

### Files Created/Modified Today
- `Code/benchmark_batch_sizes.py` - Batch size benchmark script (created)
- `Code/BATCH_SIZE_BENCHMARK_RESULTS.md` - Benchmark results (created)
- `Code/optuna_optimize.py` - Added HIGH_VRAM_GPU toggle for new system (modified)
- `PROGRESS_LOG_NOV_29_2025.md` - This file (created)
- `NEXT_SESSION_QUICKSTART.md` - Added Dec 5 setup instructions (modified)
- `DOCUMENTATION_STANDARD.md` - Documentation requirements (created)
- `CLAUDE.md` - Added batch size benchmark results (modified)

---

## Future Plans

### Immediate (This Session)
1. ~~Run batch size benchmark~~ ✅ DONE
2. ~~Update all documentation with findings~~ ✅ DONE
3. ~~Create documentation standard template~~ ✅ DONE

### December 5, 2025 - New System Arrives
**Hardware**: iBuyPower RDY Y70 B01
- Intel i9-14900KF (24 cores)
- **RTX 5070 Ti (16 GB VRAM)** - 2× current VRAM
- 32 GB DDR5-6000
- 2TB NVMe + 4TB SSD

**TODO when new system arrives**:
1. Set up conda environment (`exo-lstm-gpu`)
2. Copy data from `E:\CS_4280_Project_Backup\` to new system
3. **Run `benchmark_batch_sizes.py`** to find new optimal batch size
   - Expected optimal: batch 128-192 (vs 64 on current system)
   - Expected epoch time: ~1-1.5 min (vs 3.15 min currently)
4. **Run Optuna hyperparameter optimization** on Sector 1 dataset
   - Previous Optuna was on old 655-window dataset
   - New search should include batch sizes 64-256
   - With faster GPU: 30 trials × 20 epochs = ~2-3 hours (vs 8-12 hours on current system)
   ```powershell
   python optuna_optimize.py `
     --windows_dir "data\windows_sector1_full\train" `
     --n_trials 30 `
     --epochs_per_trial 20 `
     --output_dir "optuna_results_sector1"
   ```
5. Retrain final model with Optuna-optimized hyperparameters

### Short-term (Next Session)
1. **Retrain with optimal batch size 64**
   ```powershell
   cd C:\CS_4280_Project\Code
   conda activate exo-lstm-gpu
   python train_bilstm_cluster.py `
     --windows_dir "E:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train" `
     --n_clusters 5 `
     --epochs 60 `
     --batch_size 64 `
     --lr 0.0001 `
     --hidden 256 `
     --layers 4 `
     --dropout 0.311 `
     --pos_weight 7.41 `
     --save_dir "runs\sector1_batch64" `
     --amp_dtype fp16 `
     --num_workers 0 `
     --seed 42
   ```
   **Expected time**: ~3.14 hours (vs 24+ hours with batch 128)

2. Test best model on held-out test set (6,579 windows)

3. Generate figures for RNN paper/presentation

### Long-term
1. Run Optuna hyperparameter optimization with batch size 64
2. Compare cross-mission generalization (TESS vs Kepler)
3. Finalize term paper results

---

## Environment Details

- **OS**: Windows 11
- **GPU**: NVIDIA GeForce RTX 3060 Ti (8 GB VRAM)
- **Conda environment**: `exo-lstm-gpu`
- **Python**: Via miniconda3
- **Data backup**: `E:\CS_4280_Project_Backup\`

---

## Key Reference Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Project instructions and commands |
| `PROGRESS_LOG_NOV_29_2025.md` | This file |
| `NEXT_SESSION_QUICKSTART.md` | Quick start guide |
| `Code/train_bilstm_cluster.py` | Training script |
| `Code/benchmark_batch_sizes.py` | Batch size benchmark |
| `DOCUMENTATION_STANDARD.md` | Documentation requirements |

---

**Last updated**: 2025-11-29
