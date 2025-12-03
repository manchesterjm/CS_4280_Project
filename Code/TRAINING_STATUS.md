# Training Status Report - November 27, 2025

## Current Training (Background Process c34105)

**Model**: BiLSTM + K-means Clustering (5 clusters)
**Dataset**: TESS Sector 1 Ground Truth
- Training windows: 26,472 (11.9% planets, 88.1% non-planets)
- Validation windows: 3,970

**Configuration**:
- Epochs: 60
- Batch size: 128
- Learning rate: 0.000225
- Hidden: 256, Layers: 4
- Dropout: 0.311
- pos_weight: 7.41 (corrected for class imbalance)
- Mixed precision: FP16

**Progress**: ~25% of epoch 1 (as of last check)
**Estimated completion**: ~20 hours total

## Issues Identified & Fixed

### Clustering Problem
- **Issue**: 99.96% of windows assigned to cluster 0 (outliers dominated clustering)
- **Root cause**: A few windows have extreme variance values (std up to 99,690)
- **Fix applied**: Added percentile clipping (1-99%) before StandardScaler
- **Note**: Fix will take effect in NEXT training run (current run uses old clustering)

### Class Imbalance
- **Issue**: Original pos_weight was 3.367, but actual ratio is 7.41 (23,325/3,147)
- **Fix applied**: Using corrected pos_weight=7.41 in current training

## Files Modified

1. `train_bilstm_cluster.py` - Added:
   - CPU thread limits (OMP_NUM_THREADS=2, etc.)
   - Robust clustering with percentile clipping

2. `build_sector1_dataset_v2.py` - Added:
   - CPU thread limits to prevent crash
   - n_jobs uses cpu_count()-1 (leaves 1 core free)

## When You Return

### Check Training Progress
```powershell
# The training is running in background
# Check the runs directory for checkpoints
dir C:\CS_4280_Project\Code\runs\sector1_experiments\baseline_corrected_posweight\
```

### View Training Log
```powershell
# If training completed, check config and results
type C:\CS_4280_Project\Code\runs\sector1_experiments\baseline_corrected_posweight\config.json
```

### Start Faster Experiment with Fixed Clustering
```powershell
cd C:\CS_4280_Project\Code
conda activate exo-lstm-gpu

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
This will use the fixed clustering (percentile clipping) and run for ~5 hours.

## Expected Results

### Validation Metrics to Watch
- **AUC** > 0.70: Model is learning useful patterns
- **Recall** > 0.80: Detecting most planets
- **Precision** > 0.15: Acceptable false positive rate
- **F1** > 0.25: Balanced performance

### Comparison Baselines
| Model | AUC | F1 | Notes |
|-------|-----|----|----|
| Previous best | 0.7572 | 0.455 | 655 windows, Optuna optimized |
| Current target | >0.70 | >0.25 | 26,472 windows |

## Optuna Optimization

**Not yet started** - will require overnight run.
If current results are promising (AUC > 0.65), recommend running Optuna:
```powershell
python optuna_optimize.py --n_trials 30 --epochs_per_trial 20 --windows_dir "data\windows_sector1_full\train"
```
Estimated time: 8-12 hours overnight.
