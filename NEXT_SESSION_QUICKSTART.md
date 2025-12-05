# Next Session Quick Start Guide

## Last Session: December 5, 2025 (6:00 AM)

### Status: HIDDEN SIZE BENCHMARKED - READY FOR FINAL TRAINING

**What Was Done**:
- Ran overnight Optuna (only 1 trial completed due to hidden_size=512 being too slow)
- Trial 0: **AUC 0.9142** with hidden_size=256
- Created `benchmark_hidden_sizes.py` to test training times
- Discovered hidden_size=512 is catastrophically slow (28× slower than 128)
- Updated Optuna search space to [128, 192] for faster trials
- All pylint issues fixed (10.00/10)

---

## Hidden Size Benchmark Results

| Hidden Size | Parameters | Time/Epoch | Est. 20 Epochs |
|-------------|------------|------------|----------------|
| **128** | 1.4M | **0.36 min** | **7.2 min** |
| 192 | 3.1M | 1.44 min | 28.8 min |
| 256 | 5.4M | 2.11 min | 42.2 min |
| 512 | ~21M | ~10+ min | **UNUSABLE** |

**Recommendation**: Use hidden_size=128 for speed, or 256 for best AUC.

---

## Best Results So Far

| Trial | AUC | hidden_size | Time |
|-------|-----|-------------|------|
| Overnight Trial 0 | **0.9142** | 256 | 38.5 min |

**Best hyperparameters**:
```python
hidden_size = 256
batch_size = 112
num_layers = 4
dropout = 0.38
lr = 0.00026
n_clusters = 5
cluster_embed_dim = 64
```

---

## Priority 1: Final Model Training

Train the final model using best hyperparameters:

```powershell
powershell -Command "& 'C:\Users\manch\miniconda3\envs\exo-lstm-gpu\python.exe' -u 'D:\CS_4280_Project\Code\train_bilstm_cluster.py' --windows_dir 'D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train' --n_clusters 5 --epochs 60 --batch_size 112 --lr 0.00026 --hidden 256 --layers 4 --dropout 0.38 --save_dir 'D:\CS_4280_Project\Code\runs\sector1_final' --amp_dtype fp16 --pos_weight 7.41 --num_workers 0 --seed 42"
```

**Expected time**: ~2 hours (60 epochs × 2.11 min/epoch)

---

## Priority 2: Test Set Evaluation

After training, evaluate on test set:

```powershell
python inference_cluster_model.py --model_path "runs/sector1_final/best.pt" --windows_dir "D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\test" --output_file "reports/sector1_test_predictions.csv"
```

---

## Priority 3: Generate Figures

```powershell
python generate_bilstm_figures.py
```

---

## Alternative: Fast Training with hidden_size=128

If time is tight, use hidden_size=128 (6× faster):

```powershell
powershell -Command "& 'C:\Users\manch\miniconda3\envs\exo-lstm-gpu\python.exe' -u 'D:\CS_4280_Project\Code\train_bilstm_cluster.py' --windows_dir 'D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train' --n_clusters 5 --epochs 60 --batch_size 112 --lr 0.00026 --hidden 128 --layers 4 --dropout 0.38 --save_dir 'D:\CS_4280_Project\Code\runs\sector1_fast' --amp_dtype fp16 --pos_weight 7.41 --num_workers 0 --seed 42"
```

**Expected time**: ~22 min (60 epochs × 0.36 min/epoch)

---

## Data Locations

| Data | Location |
|------|----------|
| Training windows | `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train\` |
| Test windows | `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\test\` |
| Code | `D:\CS_4280_Project\Code\` |
| Benchmark results | `D:\CS_4280_Project\Code\hidden_size_benchmark.csv` |
| Optuna Trial 0 results | `D:\CS_4280_Project\Code\optuna_results_final2\intermediate_results.json` |

---

## Remaining Steps

1. **Final training** (60 epochs with best hyperparameters)
2. **Test set evaluation** (inference on held-out data)
3. **Generate figures** (ROC, confusion matrix, training curves)
4. **Demo video** (20 seconds)
5. **Presentation slides**

---

## Timeline

| Date | Task | Status |
|------|------|--------|
| ~~Dec 4-5~~ | ~~Optuna optimization~~ | **DONE** (1 trial, AUC 0.9142) |
| **Dec 5-6** | **Final training + figures** | **NEXT** |
| Dec 6-7 | Demo video + slides | Pending |
| **Dec 9-11** | **Presentations** | |
| **Dec 18** | **Final submission** | |

---

## Optuna Search Space (Updated Dec 5)

If you need to run more Optuna trials:
```python
hidden_size = [128, 192]  # Excluded 256+ for speed
batch_size = [96, 112, 128]
```

**Expected time per trial**:
- hidden_size=128: ~7 min
- hidden_size=192: ~29 min

---

**Last Updated**: December 5, 2025, 6:00 AM
