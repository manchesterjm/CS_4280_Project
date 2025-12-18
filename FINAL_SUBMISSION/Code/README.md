# Exoplanet Detection using BiLSTM with K-means Clustering

**CS 4280 - Machine Learning Term Project**
**Component:** RNN (BiLSTM + Clustering)
**Date:** December 2025

---

## Table of Contents
1. [Overview](#overview)
2. [Quick Start - Run in 5 Minutes](#quick-start---run-in-5-minutes)
3. [Detailed Setup Instructions](#detailed-setup-instructions)
4. [Getting the Light Curve Data](#getting-the-light-curve-data)
5. [Running Inference](#running-inference)
6. [Training a New Model](#training-a-new-model)
7. [Project Structure](#project-structure)
8. [Results](#results)
9. [Troubleshooting](#troubleshooting)

---

## Overview

This project uses a Bidirectional LSTM neural network with K-means clustering to detect exoplanet transits in stellar light curve data from NASA's TESS mission.

### Final Results

| Metric | Value |
|--------|-------|
| **AUC** | **0.9261** (92.61%) |
| **Recall** | **100%** (732/732 planets detected) |
| **F1 Score** | 0.5708 |
| **Precision** | 39.93% |
| **Accuracy** | 83.26% |

**Key Achievement:** 100% recall means the model detects every planet with zero false negatives.

---

## Quick Start - Run in 5 Minutes

This section gets you running inference with the pre-trained model as fast as possible.

### Prerequisites
- Windows 10/11 with NVIDIA GPU (CUDA capable)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download) installed

### Step 1: Open PowerShell and Navigate to Code Folder
```powershell
cd "C:\path\to\extracted\Code"
```

### Step 2: Create Conda Environment
```powershell
conda create -n exo-lstm python=3.10 -y
conda activate exo-lstm
```

### Step 3: Install Dependencies
```powershell
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install numpy pandas scikit-learn tqdm lightkurve
```

### Step 4: Download Sample Light Curves (5 confirmed exoplanet hosts)
```powershell
python -c "
import lightkurve as lk
import os
os.makedirs('sample_data', exist_ok=True)

# Download 5 confirmed TESS exoplanet host stars
tic_ids = [307210830, 261136679, 231663901, 149603524, 92226327]
for tic in tic_ids:
    print(f'Downloading TIC {tic}...')
    try:
        search = lk.search_lightcurve(f'TIC {tic}', mission='TESS')
        if len(search) > 0:
            lc = search[0].download()
            lc.to_csv(f'sample_data/TIC_{tic}.csv')
            print(f'  Saved to sample_data/TIC_{tic}.csv')
    except Exception as e:
        print(f'  Error: {e}')
print('Done!')
"
```

### Step 5: Build Windows from Sample Data
```powershell
python build_windows_from_groundtruth.py --data_dir sample_data --output_dir sample_windows --seq_len 2048 --n_windows 3 --seed 42
```

### Step 6: Run Inference
```powershell
python inference_cluster_model.py --model_path models/best.pt --windows_dir sample_windows --output_file predictions.csv
```

### Step 7: View Results
```powershell
type predictions.csv
```

The output shows planet probability for each light curve (0.0 = non-planet, 1.0 = planet).

---

## Detailed Setup Instructions

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 | Windows 11 |
| GPU | NVIDIA GTX 1060 (6GB) | RTX 3060 Ti or better |
| RAM | 8 GB | 16 GB |
| Disk Space | 5 GB | 50 GB (for full dataset) |
| CUDA | 11.8+ | 12.1+ |

### Step-by-Step Environment Setup

#### 1. Install Miniconda (if not already installed)
Download from: https://docs.conda.io/en/latest/miniconda.html

Run the installer and follow prompts. Use default settings.

#### 2. Open Anaconda PowerShell Prompt
Search for "Anaconda PowerShell Prompt" in Windows Start menu.

#### 3. Create and Activate Environment
```powershell
# Create new environment with Python 3.10
conda create -n exo-lstm python=3.10 -y

# Activate the environment
conda activate exo-lstm

# Verify activation (should show exo-lstm)
conda info --envs
```

#### 4. Install PyTorch with CUDA
```powershell
# For CUDA 12.1 (most common)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Verify GPU is detected
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060 Ti
```

#### 5. Install Remaining Dependencies
```powershell
pip install numpy pandas scikit-learn tqdm lightkurve astropy scipy
```

#### 6. Verify Installation
```powershell
python -c "
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import lightkurve as lk
print('All imports successful!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

---

## Getting the Light Curve Data

### Option A: Download Sample Data with Lightkurve (Recommended for Testing)

This downloads real TESS light curves for confirmed exoplanet hosts:

```powershell
python -c "
import lightkurve as lk
import pandas as pd
import os

os.makedirs('sample_data', exist_ok=True)

# 10 confirmed TESS exoplanet host stars (TOI catalog)
tic_ids = [
    307210830,  # L 98-59 (4 confirmed planets)
    261136679,  # TOI-700 (3 confirmed planets)
    231663901,  # TOI-175 (3 confirmed planets)
    149603524,  # TOI-421 (2 confirmed planets)
    92226327,   # TOI-561 (4 confirmed planets)
    470381900,  # TOI-1233 (4 confirmed planets)
    73649615,   # TOI-270 (3 confirmed planets)
    271893367,  # TOI-125 (3 confirmed planets)
    259962054,  # TOI-813 (1 confirmed planet)
    144065872,  # TOI-1130 (2 confirmed planets)
]

print('Downloading TESS light curves...')
print('This may take 5-10 minutes.\n')

for i, tic in enumerate(tic_ids):
    print(f'[{i+1}/10] Downloading TIC {tic}...')
    try:
        search = lk.search_lightcurve(f'TIC {tic}', mission='TESS', author='SPOC')
        if len(search) > 0:
            lc = search[0].download()
            # Save as CSV with time and flux columns
            df = pd.DataFrame({'time': lc.time.value, 'flux': lc.flux.value})
            df.to_csv(f'sample_data/TIC_{tic}.csv', index=False)
            print(f'    Saved: sample_data/TIC_{tic}.csv ({len(df)} points)')
        else:
            print(f'    No data found')
    except Exception as e:
        print(f'    Error: {e}')

print('\nDownload complete!')
print(f'Files saved to: sample_data/')
"
```

### Option B: Use the Full TESS Sector 1 Ground Truth Dataset

The model was trained on the TESS Sector 1 Ground Truth dataset (13,541 light curves).

#### Download Instructions:

1. **Go to ExoFOP-TESS**: https://exofop.ipac.caltech.edu/tess/

2. **Download TOI Catalog**:
   - Click "TOI Table"
   - Export as CSV
   - This gives you TIC IDs with disposition labels (PC = Planet Candidate, FP = False Positive, etc.)

3. **Download Light Curves using Lightkurve**:
```powershell
python -c "
import lightkurve as lk
import pandas as pd
import os

# Load your TOI list
toi_df = pd.read_csv('toi_catalog.csv')

os.makedirs('sector1_data', exist_ok=True)

for i, row in toi_df.iterrows():
    tic = row['TIC ID']
    try:
        search = lk.search_lightcurve(f'TIC {tic}', mission='TESS', sector=1)
        if len(search) > 0:
            lc = search[0].download()
            df = pd.DataFrame({'time': lc.time.value, 'flux': lc.flux.value})
            df.to_csv(f'sector1_data/TIC_{tic}.csv', index=False)
    except:
        pass
"
```

4. **Create Labels File** (`sector1_data/labels.csv`):
```csv
filename,label
TIC_307210830.csv,1
TIC_261136679.csv,1
TIC_123456789.csv,0
...
```
Where: `1` = planet, `0` = non-planet

### Option C: Generate Synthetic Data for Testing

If you cannot download real data, generate synthetic light curves:

```powershell
python -c "
import numpy as np
import pandas as pd
import os

os.makedirs('synthetic_data', exist_ok=True)

np.random.seed(42)

for i in range(20):
    # Generate time array (2 days of observations at 2-minute cadence)
    time = np.linspace(0, 2, 1440)  # 1440 points

    # Generate base flux with noise
    flux = 1.0 + np.random.normal(0, 0.001, len(time))

    # For half the samples, add a transit signal
    if i < 10:
        # Add box-shaped transit at random location
        transit_start = np.random.uniform(0.3, 1.5)
        transit_duration = 0.1
        transit_depth = np.random.uniform(0.005, 0.02)
        transit_mask = (time > transit_start) & (time < transit_start + transit_duration)
        flux[transit_mask] -= transit_depth
        label = 1
    else:
        label = 0

    df = pd.DataFrame({'time': time, 'flux': flux})
    df.to_csv(f'synthetic_data/synthetic_{i:03d}.csv', index=False)

# Create labels file
labels = pd.DataFrame({
    'filename': [f'synthetic_{i:03d}.csv' for i in range(20)],
    'label': [1]*10 + [0]*10
})
labels.to_csv('synthetic_data/labels.csv', index=False)

print('Generated 20 synthetic light curves (10 with planets, 10 without)')
print('Saved to: synthetic_data/')
"
```

---

## Running Inference

### With Pre-Trained Model

#### Step 1: Build Windows from Your Data
```powershell
python build_windows_from_groundtruth.py `
    --data_dir "sample_data" `
    --output_dir "sample_windows" `
    --seq_len 2048 `
    --n_windows 3 `
    --seed 42
```

**Arguments:**
- `--data_dir`: Folder containing CSV light curve files
- `--output_dir`: Where to save processed windows
- `--seq_len`: Window length (use 2048 to match training)
- `--n_windows`: Windows per light curve (use 3)
- `--seed`: Random seed for reproducibility

#### Step 2: Run Inference
```powershell
python inference_cluster_model.py `
    --model_path "models/best.pt" `
    --windows_dir "sample_windows" `
    --output_file "predictions.csv"
```

#### Step 3: Interpret Results

The output `predictions.csv` contains:
```csv
tic_id,probability,prediction
307210830,0.8234,1
261136679,0.7891,1
123456789,0.1234,0
```

- `probability`: Model confidence (0.0-1.0)
- `prediction`: Binary classification (1=planet, 0=non-planet)
- Threshold: 0.5 (probability > 0.5 → planet)

### Evaluate on Test Set (If You Have Labels)

```powershell
python evaluate_test.py `
    --model_path "models/best.pt" `
    --test_dir "sample_windows" `
    --output_file "evaluation_results.csv"
```

This outputs:
- AUC, F1, Precision, Recall, Accuracy
- Confusion matrix
- Per-sample predictions

---

## Training a New Model

### Prerequisites
- Training data in correct format (see "Getting the Light Curve Data")
- NVIDIA GPU with 8+ GB VRAM

### Training Command
```powershell
python train_bilstm_cluster.py `
    --windows_dir "data/windows_train" `
    --n_clusters 7 `
    --cluster_embed_dim 64 `
    --epochs 60 `
    --batch_size 128 `
    --lr 0.0001 `
    --hidden 192 `
    --layers 4 `
    --dropout 0.334 `
    --pos_weight 7.41 `
    --save_dir "runs/my_model" `
    --amp_dtype fp16 `
    --num_workers 0 `
    --seed 42
```

**Critical Parameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| `--num_workers` | 0 | **MUST be 0 on Windows** (prevents crash) |
| `--pos_weight` | 7.41 | Adjust based on your class ratio: `n_negative / n_positive` |
| `--batch_size` | 128 | Reduce to 64 or 32 if GPU memory error |
| `--lr` | 0.0001 | Lower if training becomes unstable (NaN loss) |

### Expected Output
```
[epoch  1/60] loss=0.5234 val_auc=0.7823 val_f1=0.4521 dt=1.75min
[epoch  2/60] loss=0.4123 val_auc=0.8934 val_f1=0.5234 dt=1.74min
[best] AUC improved to 0.8934; saved to runs/my_model/best.pt
...
```

Training completes in ~30 minutes on RTX 3060 Ti / RTX 5070 Ti.

---

## Project Structure

```
Code/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── models/                            # PRE-TRAINED MODEL
│   ├── best.pt                        # Trained model weights (USE THIS)
│   ├── config.json                    # Training configuration
│   └── FINAL_RESULTS.md               # Detailed results
│
├── train_bilstm_cluster.py            # Main training script
├── inference_cluster_model.py         # Run inference with trained model
├── evaluate_test.py                   # Evaluate model metrics
├── build_windows_from_groundtruth.py  # Convert light curves to windows
│
├── optuna_optimize.py                 # Hyperparameter optimization
├── generate_final_figures.py          # Generate visualizations
└── simulate_training_demo.py          # Demo for presentation
```

---

## Results

### Model Architecture: ClusterBiLSTM

```
Input: (batch, 2048, 1) - Normalized flux window
  │
  ├── K-means Cluster Assignment (7 clusters)
  │     Features: mean, std, variance, skewness, range, median, MAD, peak-to-peak
  │
  ├── Cluster Embedding Layer (7 → 64 dimensions)
  │
  ├── BiLSTM (4 layers, 192 hidden units, bidirectional)
  │     Output: 384 dimensions (192 forward + 192 backward)
  │
  ├── Concatenate [BiLSTM output + Cluster embedding] = 448 dimensions
  │
  ├── FC1 (448 → 192) + BatchNorm + ReLU + Dropout(0.334)
  ├── FC2 (192 → 96) + BatchNorm + ReLU + Dropout(0.334)
  └── FC3 (96 → 1) + Sigmoid → Planet probability

Total Parameters: 3,068,801
```

### Performance on TESS Sector 1 Test Set

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.9261** |
| **Recall** | **100%** (732/732) |
| Precision | 39.93% |
| F1 Score | 0.5708 |
| Accuracy | 83.26% |

### Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actual Positive** | TP = 732 | FN = 0 |
| **Actual Negative** | FP = 1,101 | TN = 4,746 |

---

## Troubleshooting

### "CUDA out of memory"
```powershell
# Reduce batch size
python train_bilstm_cluster.py --batch_size 32 ...
```

### "RuntimeError: DataLoader worker crashed" (Windows)
```powershell
# Always use num_workers=0 on Windows
python train_bilstm_cluster.py --num_workers 0 ...
```

### "NaN loss during training"
```powershell
# Reduce learning rate
python train_bilstm_cluster.py --lr 0.00005 ...
```

### "No module named 'torch'"
```powershell
# Make sure conda environment is activated
conda activate exo-lstm
```

### "CUDA not available"
```powershell
# Check NVIDIA driver and reinstall PyTorch
nvidia-smi  # Should show your GPU
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Lightkurve download fails
```powershell
# Try alternative download method
pip install astroquery
python -c "
from astroquery.mast import Observations
# Use MAST directly
"
```

---

## Contact

For questions about this project, please contact the course instructor or TA.

---

## License

This project is for educational purposes as part of CS 4280 coursework at University of Colorado Denver.
