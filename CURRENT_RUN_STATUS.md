# Current Run Status - Balanced Dataset Pipeline

**Started**: November 11, 2025 at 04:12 UTC
**Goal**: Complete pipeline on NEW balanced synthetic dataset for presentation comparison

---

## 🔄 Currently Running

### Optuna Hyperparameter Optimization on NEW Balanced Dataset
**Process ID**: 488495
**Dataset**: `Code/data/windows_train_400/` (1,522 windows, 30.3% positive)
**Output**: `Code/optuna_results_balanced/`

**Configuration**:
- Trials: 30
- Epochs per trial: 30
- Windows: 1,522 (vs 655 in old dataset)
- Positive rate: 30.3% (vs 23% in old dataset)
- Estimated runtime: 1.5-2 hours

**Expected Completion**: ~6:00 AM

---

## ✅ Already Completed

1. **Balanced Synthetic Dataset**: 400 light curves (200 planets + 200 non-planets)
2. **Training Windows**: 1,522 windows built successfully
3. **Script Fixes**: NaN handling and string label support

---

## ⏳ Next Steps (After Optuna Completes)

### 1. Train Final Model on NEW Dataset
Using optimized hyperparameters from Optuna:
```powershell
conda activate exo-lstm-gpu
cd C:\CS_4280_Project\Code
python train_bilstm_cluster.py `
  --windows_dir data\windows_train_400 `
  --n_clusters [FROM_OPTUNA] `
  --epochs 80 `
  --batch_size [FROM_OPTUNA] `
  --lr [FROM_OPTUNA] `
  --hidden [FROM_OPTUNA] `
  --layers [FROM_OPTUNA] `
  --dropout [FROM_OPTUNA] `
  --save_dir runs\bilstm_cluster_balanced_final `
  --amp_dtype fp16 `
  --pos_weight 2.302 `
  --num_workers 0
```

### 2. Test on TESS Data (Same 100 Planets as Before)
```powershell
python inference_cluster_model.py `
  --model_path runs\bilstm_cluster_balanced_final\best.pt `
  --windows_dir data\windows_planet_test `
  --output_file reports\balanced_model_planet_predictions.csv
```

### 3. Generate Comparison Report
Compare OLD model (trained on 655 windows) vs NEW model (trained on 1,522 balanced windows):
- Performance metrics (AUC, F1, Precision, Recall)
- Predictions on same 100 TESS planets
- Visualizations (ROC curves, confusion matrices)

---

## 📊 Expected Results

### OLD Model (Baseline):
- Training data: 655 windows (23% positive, imbalanced)
- AUC: 0.6947
- Tested on 100 planets: 0/300 positive predictions (too conservative)

### NEW Model (Balanced - Expected):
- Training data: 1,522 windows (30% positive, better balanced)
- Expected AUC: 0.72-0.76
- Expected on 100 planets: 10-30 positive predictions (better calibration)
- **Key advantage**: Better generalization from balanced training

---

## 🔍 Monitoring Progress

### Check if Optuna is creating files:
```powershell
dir C:\CS_4280_Project\Code\optuna_results_balanced\
```

Look for:
- `best_params_YYYYMMDD_HHMMSS.json`
- `trials_YYYYMMDD_HHMMSS.csv`
- `optuna_study_YYYYMMDD_HHMMSS.pkl`

### Estimated Timeline:
- **Now - 6:00 AM**: Optuna running (30 trials)
- **6:00 - 7:00 AM**: Train final model (80 epochs)
- **7:00 - 7:15 AM**: Test on TESS data
- **7:15 - 7:30 AM**: Generate comparison report

**Total time**: ~3-4 hours from start

---

## 📝 Notes

- This NEW pipeline uses completely different training data (balanced synthetic)
- Results will be directly comparable to OLD model for presentation
- Both models tested on SAME 100 confirmed TESS exoplanets
- Key research question: Does balanced synthetic training improve performance?

---

*Status will be updated as tasks complete*
