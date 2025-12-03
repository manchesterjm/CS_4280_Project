# Exoplanet Detection using BiLSTM with K-means Clustering

**CS 4280 - Machine Learning Term Project**
**Author:** Josh Manchester
**Component:** RNN (BiLSTM + Clustering)

---

## Overview

This project uses a Bidirectional LSTM neural network with K-means clustering to detect exoplanet transits in stellar light curve data from NASA's TESS mission. The model achieves **AUC 0.91+** on the TESS Sector 1 ground truth dataset.

### Key Innovation
- K-means clustering groups light curves by statistical features before training
- Cluster embeddings provide context to the BiLSTM, allowing it to learn different patterns for different stellar types
- This approach improves AUC by ~3% compared to BiLSTM without clustering

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX 3060 Ti 8GB, RTX 5070 Ti 16GB)
- Minimum 16 GB RAM
- ~50 GB disk space for data

### Software
```
Python 3.10+
PyTorch 2.0+ (with CUDA)
NumPy
Pandas
scikit-learn
tqdm
```

### Environment Setup
```powershell
# Create conda environment
conda create -n exo-lstm-gpu python=3.10
conda activate exo-lstm-gpu

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
pip install numpy pandas scikit-learn tqdm
```

---

## Data

### Source: TESS Sector 1 Ground Truth Dataset

The training data comes from the TESS Sector 1 ground truth dataset, which contains labeled light curves for:
- **Planets**: 3,146 confirmed/candidate exoplanet hosts
- **Stars**: 8,624 non-planet stellar sources
- **Eclipsing Binaries (EB)**: 900 binary star systems
- **Background EBs**: 871 background eclipsing binaries

**Total**: 13,541 light curves → 33,051 training windows (after 80/20 train/test split)

### Data Location
The processed data should be placed in:
```
data/
├── windows_sector1_full/
│   ├── train/
│   │   ├── X.npy          # (26472, 2048) float32 - normalized flux windows
│   │   ├── y.npy          # (26472,) int64 - labels (1=planet, 0=non-planet)
│   │   └── meta.csv       # Metadata with statistical features
│   └── test/
│       ├── X.npy          # (6579, 2048) float32
│       ├── y.npy          # (6579,) int64
│       └── meta.csv
```

### Obtaining the Data

**Option 1: Use pre-processed windows (recommended)**
Contact the course instructor or TA for access to the pre-processed `windows_sector1_full/` directory.

**Option 2: Build from raw ground truth data**
If you have access to the raw TESS Sector 1 ground truth files:
```powershell
python build_windows_from_groundtruth.py `
  --data_dir "path/to/sector-1/ground-truth" `
  --output_dir "data/windows_sector1_full" `
  --seq_len 2048 `
  --n_windows 3 `
  --seed 42
```

The raw ground truth data is available from the TESS project or can be obtained through team collaboration.

---

## Quick Start

### 1. Train the Model
```powershell
cd C:\CS_4280_Project\Code
conda activate exo-lstm-gpu

python train_bilstm_cluster.py `
  --windows_dir "data/windows_sector1_full/train" `
  --n_clusters 5 `
  --epochs 60 `
  --batch_size 64 `
  --lr 0.0001 `
  --hidden 256 `
  --layers 4 `
  --dropout 0.311 `
  --pos_weight 7.41 `
  --save_dir "runs/my_model" `
  --amp_dtype fp16 `
  --num_workers 0 `
  --seed 42
```

**Expected output:**
- Training takes ~3 hours on RTX 3060 Ti (batch size 64)
- Model checkpoints saved to `runs/my_model/`
- Best model: `runs/my_model/best.pt`

### 2. Evaluate on Test Set
```powershell
python evaluate_test.py `
  --model_path "runs/my_model/best.pt" `
  --test_dir "data/windows_sector1_full/test"
```

### 3. Run Inference on New Data
```powershell
python inference_cluster_model.py `
  --model_path "runs/my_model/best.pt" `
  --windows_dir "data/windows_test" `
  --output_file "reports/predictions.csv"
```

---

## Project Structure

```
Code/
├── README.md                          # This file
│
├── # ESSENTIAL SCRIPTS (Core Pipeline)
├── build_windows_from_groundtruth.py  # Build training windows from raw data
├── train_bilstm_cluster.py            # Main training script (BiLSTM + K-means)
├── inference_cluster_model.py         # Run inference with trained model
├── evaluate_test.py                   # Evaluate model on test set
│
├── # OPTIONAL SCRIPTS (Optimization & Analysis)
├── optuna_optimize.py                 # Hyperparameter optimization with Optuna
├── benchmark_batch_sizes.py           # Find optimal batch size for your GPU
│
├── # DATA & OUTPUTS
├── data/                              # Training/test windows (not in git)
│   └── windows_sector1_full/
├── runs/                              # Model checkpoints (not in git)
└── reports/                           # Prediction outputs
```

---

## Model Architecture

### ClusterBiLSTM

```
Input: (batch, 2048, 1) - Normalized flux window
  │
  ├── K-means Cluster Assignment (based on statistical features)
  │     Features: mean, std, variance, skewness, range, median, MAD, peak-to-peak
  │
  ├── Cluster Embedding Layer (5 clusters → 32 dimensions)
  │
  ├── BiLSTM (4 layers, 256 hidden units, bidirectional)
  │     Output: Forward + Backward hidden states (512 total)
  │
  ├── Concatenate [BiLSTM output (512) + Cluster embedding (32)] = 544
  │
  ├── FC1 (544 → 256) + BatchNorm + ReLU + Dropout
  ├── FC2 (256 → 128) + BatchNorm + ReLU + Dropout
  └── FC3 (128 → 1) + Sigmoid → Planet probability
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| seq_len | 2048 | Window length (flux points) |
| hidden_size | 256 | BiLSTM hidden units |
| num_layers | 4 | BiLSTM layers |
| n_clusters | 5 | K-means clusters |
| dropout | 0.311 | Dropout rate |
| learning_rate | 0.0001 | Adam learning rate |
| batch_size | 64 | Training batch size |
| pos_weight | 7.41 | Class imbalance weight (neg/pos ratio) |

---

## Results

### Performance on TESS Sector 1 Test Set

| Metric | Value |
|--------|-------|
| **AUC** | 0.91+ |
| Accuracy | ~85% |
| Precision | ~0.45 |
| Recall | ~0.80 |
| F1 Score | ~0.58 |

### Training Details
- **Dataset**: 26,472 training windows, 6,579 test windows
- **Class distribution**: 11.9% planets, 88.1% non-planets
- **Training time**: ~3 hours (60 epochs, RTX 3060 Ti)
- **Convergence**: Best model typically at epoch 40-50

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```
Reduce batch size: --batch_size 32
```

**2. Training crash on Windows (multiprocessing error)**
```
Always use: --num_workers 0
```

**3. NaN loss during training**
```
Reduce learning rate: --lr 0.00005
```

**4. Model not learning (AUC stuck at 0.5)**
```
Check pos_weight matches your data: pos_weight = n_negative / n_positive
```

---

## Citation

If you use this code, please cite:

```
Manchester, J. (2025). Exoplanet Detection using BiLSTM with K-means Clustering.
CS 4280 Machine Learning Term Project, University of Colorado Denver.
```

---

## License

This project is for educational purposes as part of CS 4280 coursework.
