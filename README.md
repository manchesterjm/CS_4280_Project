# CS 4280 - Exoplanet Detection using Deep Learning

## Project Overview

This project uses deep learning (RNN/LSTM networks) to detect exoplanets from stellar light curve data. The model analyzes time-series brightness measurements from NASA's TESS/Kepler missions to identify the characteristic dips caused by planetary transits.

## Current Status (October 2025)

### Problem Identified
- ✅ Initial model trained only on clean planet signals
- ❌ When tested on real data (with flares, noise), it flagged everything as a planet
- ✅ Retraining on mixed dataset (planets + non-planets) to teach discrimination
- ❌ Simple LSTM architecture cannot learn from the data (AUC stuck at ~0.5)
- ✅ Data is confirmed learnable (Logistic Regression achieves 74% accuracy)
- 🔄 **Next step: Implement Conv-LSTM hybrid architecture**

### Key Findings from Diagnostics
- Dataset: 655 windows (150 positive, 505 negative) = 22.9% imbalanced
- Data quality: Clean (no NaN/Inf), properly normalized (mean=0, std=1)
- Windows are 2048 time steps each
- 101 unique stars with ~6.5 windows per star
- Simple LSTM fails but data is learnable → Need better architecture

## Project Structure

```
CS_4280_Project/
├── Code/
│   ├── diagnose_data.py          # Data quality checker
│   ├── investigate_data.py       # Deep data investigation
│   ├── train_lstm_fixed.py       # Fixed LSTM trainer (doesn't work well)
│   ├── build_windows_parallel_v6.py   # Window building (latest)
│   ├── build_windows_infer_v2.py      # Inference window building
│   ├── inference_rnn.py               # Model inference
│   ├── postfilter_inference_v3.py     # Post-processing (latest)
│   ├── evaluate_pr_v2.py              # Evaluation metrics
│   │
│   ├── data/
│   │   ├── windows_train/        # Training data (655 windows)
│   │   │   ├── X.npy             # Features (655, 2048)
│   │   │   ├── y.npy             # Labels (655,)
│   │   │   └── meta.csv          # Metadata
│   │   ├── windows_infer/        # Inference data
│   │   └── windows/              # Full dataset
│   │
│   ├── runs/
│   │   ├── lstm_fixed/           # Latest training attempt (failed)
│   │   └── lstm_v2/              # Previous attempt
│   │
│   ├── reports/                  # Evaluation outputs
│   │
│   └── Archive/                  # Old versions
│       ├── scripts/              # Deprecated scripts
│       ├── models/               # Old models
│       └── data/                 # Old data
│
├── Planet_LightCurve_Data/
│   └── processed/                # 100 confirmed exoplanet light curves
│
└── test_dataset/
    └── simulated_dataset/
        └── processed/            # 106 test light curves (planets + flares)
```

## Data Pipeline

### 1. Raw Data
- **Planet_LightCurve_Data**: 100 confirmed exoplanet host stars (positive examples)
- **test_dataset**: 106 light curves including flares, stellar activity, and planets

### 2. Window Building
- Script: `build_windows_parallel_v6.py`
- Extracts 2048-point sliding windows from light curves
- Labels based on `manifest.csv` (planet=1, non-planet=0)
- Output: `data/windows_train/` with X.npy, y.npy, meta.csv

### 3. Training
- **Current**: `train_lstm_fixed.py` (simple 2-layer LSTM) - **NOT WORKING**
- **Next**: Conv-LSTM hybrid (in development)
- Handles class imbalance with pos_weight=3.367
- Uses mixed precision (FP16) for faster training

### 4. Inference & Post-processing
- `inference_rnn.py`: Run model on new data
- `postfilter_inference_v3.py`: Clean up false positives
- `evaluate_pr_v2.py`: Generate precision-recall curves

## Environment Setup

```bash
conda activate exo-lstm-gpu
```

**Key packages:**
- PyTorch (with CUDA)
- NumPy, Pandas
- scikit-learn
- tqdm

## How to Run

### 1. Check Data Quality
```powershell
cd C:\CS_4280_Project\Code
python diagnose_data.py
```

### 2. Deep Investigation (if issues)
```powershell
python investigate_data.py
```

### 3. Train Model
```powershell
python train_lstm_fixed.py `
  --windows_dir "C:\CS_4280_Project\Code\data\windows_train" `
  --epochs 40 `
  --batch_size 256 `
  --lr 1e-4 `
  --hidden 128 `
  --layers 2 `
  --dropout 0.3 `
  --save_dir "C:\CS_4280_Project\Code\runs\lstm_fixed" `
  --amp_dtype fp16 `
  --pos_weight 3.367 `
  --num_workers 0
```

**Note**: `num_workers=0` is required on Windows to avoid multiprocessing crashes.

## Known Issues & Solutions

### Issue 1: Model Not Learning (CURRENT)
- **Symptom**: AUC stuck at ~0.5, loss not decreasing
- **Cause**: Simple LSTM architecture insufficient for this data
- **Solution**: Implementing Conv-LSTM hybrid (combines CNN for local features + BiLSTM for temporal patterns)

### Issue 2: Windows Multiprocessing Crash (FIXED)
- **Symptom**: Training crashes during epoch 3 with sklearn import error
- **Solution**: Set `num_workers=0` in DataLoader

### Issue 3: Class Imbalance (FIXED)
- **Symptom**: Model predicts everything as one class
- **Solution**: Use `pos_weight=3.367` in BCEWithLogitsLoss

### Issue 4: Training on Wrong Data (FIXED)
- **Symptom**: Model flags everything as planet when tested on real data
- **Cause**: Trained only on clean planet signals, never saw flares/noise
- **Solution**: Retrain on mixed dataset with both planets and non-planets

## Training Results

### Latest Attempt: Simple LSTM (train_lstm_fixed.py)
```
Best AUC: 0.5293 (epoch 2)
Final AUC: 0.4834 (epoch 40)
Status: FAILED - Model not learning
```

The model couldn't learn because simple LSTM is insufficient. Even though:
- Data is clean and properly normalized
- Class imbalance is handled
- Logistic Regression achieves 74% accuracy

This confirms we need a more sophisticated architecture.

## Next Steps

1. **Implement Conv-LSTM Architecture** (IN PROGRESS)
   - Add CNN layers before LSTM to extract local transit patterns
   - Use BiLSTM for bidirectional temporal context
   - Expected: AUC > 0.75

2. **Hyperparameter Tuning**
   - Grid search over learning rates, hidden sizes
   - Try different CNN kernel sizes
   - Experiment with attention mechanisms

3. **Ensemble Methods**
   - Combine multiple models for robustness
   - Vote on final predictions

4. **Production Pipeline**
   - Automate end-to-end inference
   - Add confidence thresholds
   - Generate reports for astronomers

## Key Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of predicted planets, how many are real?
- **Recall**: Of real planets, how many did we find?
- **F1**: Harmonic mean of precision/recall
- **AUC**: Area under ROC curve (most important for imbalanced data)

**Target**: AUC > 0.8, F1 > 0.6

## Important Notes

- This is a **Windows 11** development environment
- GPU: CUDA-enabled (check with `nvidia-smi`)
- Data paths use Windows-style backslashes
- Always use `num_workers=0` in DataLoader on Windows
- Training logs are printed to console (not saved by default)

## Contact & Development History

**Current Session**: October 2025
- Identified simple LSTM failure
- Confirmed data is learnable
- Cleaned up project structure
- Next: Implementing Conv-LSTM

## References

- NASA TESS/Kepler missions
- Light curve analysis techniques
- Deep learning for time-series classification
