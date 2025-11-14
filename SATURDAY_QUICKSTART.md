# Saturday Quickstart Guide - November 16, 2025

## What We're Doing
Transitioning to **TESS Sector 1 Ground Truth Dataset** (team alignment) - 40,623 training windows from 13,541 real light curves.

---

## Quick Status Check

**Last Session**: Thursday, November 14, 2025 at 5:28 AM
**Current Issue**: Dataset needs rebuild with statistical features for clustering
**Why**: Previous build missing features needed to prevent data leakage

---

## Saturday Morning Checklist

### 1. System Cleanup (5 minutes)
```powershell
# Kill old background processes
# (Use Task Manager or check running Python processes)

# Delete artifact file
Remove-Item C:\CS_4280_Project\nul -Force

# Verify GPU
nvidia-smi

# Activate environment
conda activate exo-lstm-gpu
cd C:\CS_4280_Project\Code
```

### 2. Rebuild Sector 1 Dataset (10 minutes)
```powershell
# Delete old dataset (missing statistical features)
Remove-Item data\windows_sector1_full\* -Force

# Rebuild with updated script
python build_windows_from_groundtruth.py `
  --data_dir "E:\lilith4_sector-1_groundtruth\sector-1\ground-truth" `
  --output_dir "data\windows_sector1_full" `
  --seq_len 2048 `
  --n_windows 3 `
  --seed 42
```

**Expected Output**:
```
Processing planets: 3146 files...
Processing stars: 8624 files...
Processing ebs: 900 files...
Processing backebs: 871 files...

DATASET SUMMARY
======================================================================
Total windows: 40623
Positive (planets): 9438 (23.2%)
Negative (non-planets): 31185 (76.8%)
Window shape: (40623, 2048)
```

### 3. Verify Statistical Features
```python
import pandas as pd
meta = pd.read_csv('data/windows_sector1_full/meta.csv')
print("Columns:", list(meta.columns))
# Must include: mean, std, var, skew, range, median, mad, peak_to_peak
```

### 4. Train BiLSTM+Clustering Model (60 minutes)
```powershell
python train_bilstm_cluster.py `
  --windows_dir "data\windows_sector1_full" `
  --n_clusters 5 `
  --epochs 80 `
  --batch_size 128 `
  --lr 0.000225 `
  --hidden 256 `
  --layers 4 `
  --dropout 0.311 `
  --save_dir "runs\bilstm_cluster_sector1" `
  --amp_dtype fp16 `
  --pos_weight 3.367 `
  --num_workers 0 `
  --seed 42
```

**Watch For**:
- Training should start without KeyError or NaN warnings
- Clustering message: "Using statistical features (mean, std, var, skew, range, median, mad, peak_to_peak)"
- AUC should NOT reach 1.0 on epoch 1 (that indicates data leakage)

---

## Expected Performance Targets

| Metric | Target | Previous (655 windows) |
|--------|--------|------------------------|
| AUC | >0.75 | 0.7572 |
| F1 | >0.50 | 0.455 |
| Recall | >0.80 | 0.86 |
| Precision | >0.40 | 0.31 |

---

## Key Reference Files

- **Full Progress Log**: `PROGRESS_LOG_NOV_14_2025.md`
- **Updated Documentation**: `CLAUDE.md` (see "TESS Sector 1 Ground Truth Dataset" section)
- **Processing Script**: `Code/build_windows_from_groundtruth.py`
- **Training Script**: `Code/train_bilstm_cluster.py`
- **Batch Files**:
  - `Code/build_sector1_dataset.bat`
  - `Code/train_sector1.bat`

---

## Common Issues & Fixes

### Issue: "KeyError: 'period' or 'mean'"
**Fix**: Dataset missing features. Rebuild dataset (Step 2 above).

### Issue: AUC = 1.0 on epoch 1, then NaN
**Fix**: Data leakage in clustering. Verify statistical features present (Step 3 above).

### Issue: "sklearn import error" during training
**Fix**: Use `--num_workers 0` (Windows multiprocessing issue).

### Issue: CUDA out of memory
**Fix**: Reduce batch size from 128 to 64 in training command.

---

## After Training Completes

1. **Check validation metrics**:
   - Best checkpoint: `runs/bilstm_cluster_sector1/best.pt`
   - Look for: AUC, F1, precision, recall

2. **Compare with previous models**:
   - Previous best: AUC 0.7572 (655 windows)
   - SMOTE model: 3/19 true positives (15.8% precision)

3. **Test on real planets** (if time permits):
   ```powershell
   python inference_cluster_model.py `
     --model_path "runs\bilstm_cluster_sector1\best.pt" `
     --windows_dir "data\windows_planet_test" `
     --output_file "reports\sector1_planet_predictions.csv"
   ```

---

## Success Criteria

✅ Dataset rebuilt with 40,623 windows
✅ Metadata contains statistical features
✅ Training completes without errors
✅ AUC > 0.70 on validation set
✅ No data leakage (AUC ≠ 1.0 on epoch 1)

---

## Questions to Answer

1. Does larger dataset improve AUC?
2. How does class balance (23.2% vs 22.9%) affect performance?
3. What do the 5 K-means clusters represent?
4. How many real planets can we detect in test set?

---

**Good luck! See you Saturday! 🚀**
