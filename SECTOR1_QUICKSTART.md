# Quick Start Guide - TESS Sector 1 Dataset

**Status**: Dataset is processing (ETA: 2-3 minutes)

---

## What's Happening Right Now

Your Sector 1 ground truth dataset is being processed into training windows:

**Progress**:
- ✅ Planets: 3,146 files (DONE)
- ⏳ Stars: 8,624 files (in progress)
- ⏳ EBs: 900 files (pending)
- ⏳ BackEBs: 871 files (pending)

**Output**: `C:\CS_4280_Project\Code\data\windows_sector1_full`

---

## Once Processing Completes

### Step 1: Verify Dataset

```powershell
cd C:\CS_4280_Project\Code
python -c "import numpy as np; X=np.load('data/windows_sector1_full/X.npy'); y=np.load('data/windows_sector1_full/y.npy'); print(f'Dataset: {X.shape[0]} windows, {sum(y)} planets ({sum(y)/len(y)*100:.1f}%)')"
```

Expected output: ~40,623 windows, ~9,438 planets (23.2%)

### Step 2: Train Model on Full Dataset

```powershell
conda activate exo-lstm-gpu
cd C:\CS_4280_Project\Code

python train_bilstm_cluster.py `
  --windows_dir "data/windows_sector1_full" `
  --n_clusters 5 `
  --epochs 80 `
  --batch_size 128 `
  --lr 0.000225 `
  --hidden 256 `
  --layers 4 `
  --dropout 0.311 `
  --save_dir "runs/bilstm_cluster_sector1" `
  --amp_dtype fp16 `
  --pos_weight 3.367 `
  --num_workers 0
```

**Expected training time**: 2-3 hours (62× more data than before!)

### Step 3: Test on Real Planets

Once training completes:

```powershell
python inference_cluster_model.py `
  --model_path "runs/bilstm_cluster_sector1/best.pt" `
  --windows_dir "data/windows_planet_test" `
  --output_file "reports/sector1_planet_predictions.csv"
```

Then analyze false positives:

```powershell
python analyze_false_positives.py
# (Update the script to use "sector1_planet_predictions.csv")
```

---

## Expected Performance Improvements

| Metric | Old Model | Sector 1 Model (Expected) |
|--------|-----------|---------------------------|
| Training data | 655 windows | 40,623 windows |
| AUC | 0.7572 | 0.80-0.85 |
| F1 Score | 0.52 | 0.60-0.70 |
| Precision | 0.31 | 0.50-0.60 |
| Recall | 0.86 | 0.70-0.80 |

With 62× more training data, you should see significant improvements!

---

## Alternative: Apply SMOTE (Optional)

If you want to experiment with balanced training, apply SMOTE to the Sector 1 dataset:

```powershell
python balance_with_true_smote.py `
  --windows_dir "data/windows_sector1_full" `
  --output_dir "data/windows_sector1_smote" `
  --method hybrid `
  --target_size 10000 `
  --seed 42
```

This creates ~20,000 balanced windows (10k planets + 10k non-planets).

Then train:

```powershell
python train_bilstm_cluster.py `
  --windows_dir "data/windows_sector1_smote" `
  --n_clusters 5 `
  --epochs 80 `
  --batch_size 128 `
  --lr 0.000225 `
  --hidden 256 `
  --layers 4 `
  --dropout 0.311 `
  --save_dir "runs/bilstm_cluster_sector1_smote" `
  --amp_dtype fp16 `
  --pos_weight 1.0 `
  --num_workers 0
```

Note: `pos_weight 1.0` since SMOTE makes data balanced.

---

## Key Benefits of Sector 1 Dataset

1. **62× more data** (655 → 40,623 windows)
2. **Ground truth labels** (no BLS detection errors)
3. **Diverse negatives** (stars, EBs, BackEBs)
4. **Team alignment** (all partners using same dataset)
5. **Natural class balance** (23.2% positive)

---

## Files Created

1. `build_windows_from_groundtruth.py` - Dataset builder
2. `build_sector1_dataset.bat` - Batch script to run builder
3. `data/windows_sector1_full/` - Full dataset (processing now)
4. `SECTOR1_DATASET_SUMMARY.md` - Detailed documentation
5. `SECTOR1_QUICKSTART.md` - This file

---

## Troubleshooting

**Q: Dataset processing failed?**
A: Check `data/windows_sector1_full/` exists and has X.npy, y.npy, meta.csv

**Q: Training runs out of memory?**
A: Reduce batch size from 128 to 64 or 32

**Q: Training too slow?**
A: Check `nvidia-smi` to ensure GPU is being used

**Q: Want to use smaller subset for testing?**
A: Run with `--max_files 100` to process only 100 files per category

---

*Created: November 13, 2025*
*Next: Wait for dataset processing to complete, then train model*
