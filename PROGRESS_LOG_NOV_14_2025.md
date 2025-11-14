# Progress Log - November 14, 2025 (Thursday 5:28 AM)

## Session Summary
**Date/Time**: 2025-11-14 05:28:37
**Status**: Work paused - resuming Saturday
**Duration**: ~2 hours

---

## Major Pivot: Transitioning to TESS Sector 1 Ground Truth Dataset

### Background
- Previously working with small synthetic/simulated datasets (655 windows)
- Team partners are using TESS Sector 1 ground truth dataset
- **REQUIREMENT**: Must align with team to use the same dataset

### New Dataset Details
**Location**: `E:\lilith4_sector-1_groundtruth\sector-1\ground-truth`

**Structure**:
- **Total files**: 13,541 light curves across 4 categories
- **planets/**: 3,146 files (confirmed planetary transits) → **LABEL 1**
- **Stars/**: 8,624 files (stellar variability, no planets) → **LABEL 0**
- **EBs/**: 900 files (eclipsing binaries) → **LABEL 0**
- **BackEBs/**: 871 files (background eclipsing binaries) → **LABEL 0**

**Class Balance**:
- Positive (planets): 9,438 windows (23.2%)
- Negative (non-planets): 31,185 windows (76.8%)
- **62× larger than previous dataset!**

**Expected Output**:
- **~40,623 training windows** (3 windows per light curve × 13,541 files)
- Window size: 2048 timesteps
- Features: Robust z-score normalized flux

---

## Work Completed

### 1. False Positive Analysis (SMOTE Model)
**Status**: ✅ Completed

Created `analyze_false_positives.py` to analyze precision of SMOTE-balanced model:
- **Input**: 19 detections from SMOTE model on 300 planet test windows
- **Results**:
  - True Positives (phase=0.0): **3 windows** (15.8%)
  - False Positives (phase≠0.0): **16 windows** (84.2%)
  - **Precision: 15.8%** (extremely poor)

**Conclusion**: SMOTE model has very high false positive rate, detecting transits in non-transit regions.

---

### 2. New Dataset Processing Pipeline
**Status**: ⚠️ IN PROGRESS (needs rebuild)

#### Created `build_windows_from_groundtruth.py`
**Purpose**: Process Sector 1 ground truth data into training windows

**Key Features**:
- Reads gzipped light curve files (`.txt.gz`)
- Extracts 3 windows per light curve at random positions
- Normalizes using **robust z-score** (median + MAD for outlier resistance)
- Computes **statistical features for clustering**:
  - `mean`, `std`, `var` (variance)
  - `skew` (interquartile range approximation)
  - `range` (max - min)
  - `median`, `mad` (Median Absolute Deviation)
  - `peak_to_peak` (ptp)

**Output Structure**:
```
data/windows_sector1_full/
  X.npy          # (40623, 2048) normalized flux windows
  y.npy          # (40623,) labels (1=planet, 0=non-planet)
  meta.csv       # Metadata with statistical features
```

**Critical Fix Applied**:
- **PROBLEM**: Initial version only saved `tic_id, category, window_idx, label`
- **FIX**: Added 8 statistical features for K-means clustering
- **WHY**: Prevents data leakage (using 'category' would reveal labels)

---

### 3. Training Script Updates
**Status**: ✅ Completed

#### Modified `train_bilstm_cluster.py`
**Changes**:
1. Added detection for statistical features vs BLS features
2. Fallback logic:
   - **Preferred**: BLS features (period, duration, depth, bls_power)
   - **Alternative**: Statistical features (mean, std, var, skew, range, median, mad, peak_to_peak)
   - **Fallback**: All samples → cluster 0 (no clustering)

**Code Location**: `train_bilstm_cluster.py:123-162`

---

### 4. Simple BiLSTM Script (Backup)
**Status**: ✅ Created (not used)

Created `train_simple_bilstm.py` for training without clustering:
- Pure BiLSTM without K-means clustering
- **User rejected**: Must use BiLSTM+clustering per presentation requirement
- Kept as reference for comparison

---

## Critical Issue Discovered

### Dataset Rebuild Required
**Problem**:
- First dataset build (40,623 windows) completed successfully
- **BUT**: Built with OLD script version (before statistical features added)
- Metadata only contains: `['tic_id', 'category', 'window_idx', 'label']`
- **MISSING**: Statistical features needed for proper clustering

**Impact**:
- Cannot train BiLSTM+clustering model without statistical features
- Using 'category' for clustering causes **100% data leakage**
  - Cluster 0 = 100% planets
  - Clusters 1-3 = 100% non-planets
  - Model memorizes cluster→label mapping → AUC=1.0 then crashes with NaN

**Solution Required**:
```bash
# Delete old dataset
Remove-Item C:\CS_4280_Project\Code\data\windows_sector1_full\* -Force

# Rebuild with updated script (includes statistical features)
python build_windows_from_groundtruth.py \
  --data_dir E:\lilith4_sector-1_groundtruth\sector-1\ground-truth \
  --output_dir data\windows_sector1_full \
  --seq_len 2048 \
  --n_windows 3 \
  --seed 42
```

**Expected Time**: ~5-10 minutes (13,541 files to process)

---

## Next Steps (Resume Saturday)

### Priority 1: Rebuild Dataset ✅
```bash
cd C:\CS_4280_Project\Code
conda activate exo-lstm-gpu

# Rebuild dataset with statistical features
python build_windows_from_groundtruth.py \
  --data_dir E:\lilith4_sector-1_groundtruth\sector-1\ground-truth \
  --output_dir data\windows_sector1_full \
  --seq_len 2048 \
  --n_windows 3 \
  --seed 42
```

**Verify statistical features are present**:
```python
import pandas as pd
meta = pd.read_csv('data/windows_sector1_full/meta.csv')
print(meta.columns)  # Should show: mean, std, var, skew, range, median, mad, peak_to_peak
```

### Priority 2: Train BiLSTM+Clustering Model 🎯
```bash
python train_bilstm_cluster.py \
  --windows_dir data\windows_sector1_full \
  --n_clusters 5 \
  --epochs 80 \
  --batch_size 128 \
  --lr 0.000225 \
  --hidden 256 \
  --layers 4 \
  --dropout 0.311 \
  --save_dir runs\bilstm_cluster_sector1 \
  --amp_dtype fp16 \
  --pos_weight 3.367 \
  --num_workers 0 \
  --seed 42
```

**Expected Training Time**: ~45-60 minutes (80 epochs, 40K windows, GPU FP16)

**Expected Performance**:
- **Baseline**: AUC 0.70-0.75 (larger dataset, better diversity)
- **Improved**: Better recall than previous 655-window model
- **Class balance**: 23.2% positive (vs 22.9% previously)

### Priority 3: Evaluate and Test 📊
1. **Validation metrics**: Track AUC, F1, precision, recall during training
2. **Test on real planets**: Run inference on 100 confirmed exoplanet systems
3. **Compare with SMOTE model**: Quantify improvement over SMOTE approach
4. **Analyze clustering**: Examine K-means cluster distributions

### Priority 4: Documentation Updates 📝
1. Update `CLAUDE.md` with Sector 1 dataset details
2. Document training results in new summary file
3. Create comparison tables (Sector 1 vs previous datasets)

---

## Technical Architecture (No Changes)

### BiLSTM+Clustering Model
```
Input: (batch, 2048, 1) flux window + cluster_id
  ↓
Cluster Embedding (n_clusters=5 → 32-dim)
  ↓
BiLSTM (4 layers, 256 hidden, bidirectional)
  ↓
Concatenate [hidden_fwd, hidden_bwd, cluster_embed]
  ↓
FC1 (512+32 → 256) + BatchNorm + ReLU + Dropout
  ↓
FC2 (256 → 128) + BatchNorm + ReLU + Dropout
  ↓
FC3 (128 → 1) → Sigmoid → Probability
```

**Key Hyperparameters** (from Optuna optimization):
- Hidden: 256
- Layers: 4
- Dropout: 0.311
- Learning Rate: 0.000225
- Batch Size: 128
- Pos Weight: 3.367 (class imbalance correction)

---

## Files Modified/Created This Session

### Created
- `Code/analyze_false_positives.py` - Analyze FP rate in SMOTE predictions
- `Code/build_windows_from_groundtruth.py` - Process Sector 1 dataset
- `Code/build_sector1_dataset.bat` - Batch script for dataset building
- `Code/train_simple_bilstm.py` - Simple BiLSTM without clustering (backup)
- `Code/train_sector1.bat` - Training batch script for Sector 1
- `SECTOR1_DATASET_SUMMARY.md` - Sector 1 dataset documentation
- `SECTOR1_QUICKSTART.md` - Quick reference guide
- `PROGRESS_LOG_NOV_14_2025.md` - This file

### Modified
- `Code/train_bilstm_cluster.py` - Added statistical feature support for clustering

### Partially Created (needs rebuild)
- `Code/data/windows_sector1_full/` - Dataset WITHOUT statistical features (old version)

---

## Known Issues

1. **Dataset Missing Statistical Features** ⚠️
   - Current dataset built with old script
   - Needs complete rebuild before training
   - Estimated rebuild time: 5-10 minutes

2. **Multiple Background Processes Running** ⚠️
   - Several old SMOTE training processes still running
   - Should be killed before Saturday session:
     - `ca8a74`, `ae6a73`, `a06275`, `64592d`, `cfeecc`, `ff42b9`

3. **"nul" File Created** ⚠️
   - Strange file `nul` exists in project root
   - Should be deleted (Windows redirect artifact)

---

## User Requirements (Hard Constraints)

1. ✅ **Must use BiLSTM+clustering architecture** (per presentation)
2. ✅ **Must use TESS Sector 1 ground truth dataset** (team requirement)
3. ✅ **No data leakage in clustering** (use statistical features, not labels)
4. ⏳ **Train model on Sector 1 data** (pending dataset rebuild)

---

## Performance Targets

### Previous Best (655 windows)
- AUC: 0.7572 (Optuna optimized)
- F1: 0.455
- Recall: 0.86
- Precision: 0.31

### Sector 1 Goals
- **AUC**: >0.75 (maintain or improve)
- **F1**: >0.50 (improve balance)
- **Recall**: >0.80 (keep high detection rate)
- **Precision**: >0.40 (reduce false positives)
- **Real planet detection**: >20/300 windows (6.7%) on test set

---

## Saturday Morning Checklist 📋

Before starting:
- [ ] Kill all background processes (SMOTE training, etc.)
- [ ] Delete `nul` file in project root
- [ ] Verify GPU availability (`nvidia-smi`)
- [ ] Activate conda environment (`conda activate exo-lstm-gpu`)

Then proceed with:
1. ✅ Rebuild Sector 1 dataset with statistical features
2. ✅ Verify metadata columns
3. 🎯 Train BiLSTM+clustering model
4. 📊 Evaluate and compare results
5. 📝 Update documentation

---

## Contact Info for Team
- Dataset location: `E:\lilith4_sector-1_groundtruth\sector-1\ground-truth`
- Processed windows: `C:\CS_4280_Project\Code\data\windows_sector1_full\`
- Model checkpoints: `C:\CS_4280_Project\Code\runs\bilstm_cluster_sector1\`

---

**End of Session**: 2025-11-14 05:28:37
**Next Session**: Saturday, November 16, 2025
**Status**: Dataset rebuild required, then ready to train
