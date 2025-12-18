# Exoplanet Detection using BiLSTM with K-means Clustering

**CS 4280 - Machine Learning Term Project**
**Author:** Josh Manchester
**Component:** RNN (BiLSTM + Clustering)
**Status:** COMPLETE ✅ (December 6, 2025)

---

## Overview

This project uses a Bidirectional LSTM neural network with K-means clustering to detect exoplanet transits in stellar light curve data from NASA's TESS mission.

### Final Test Results (December 6, 2025)

| Metric | Value |
|--------|-------|
| **AUC** | **0.9261** (92.61%) |
| **Recall** | **100%** (732/732 planets) |
| **F1 Score** | 0.5708 |
| **Precision** | 39.93% |
| **Accuracy** | 83.26% |

**Confusion Matrix**: TP=732, FP=1,101, TN=4,746, FN=0

### Key Innovation
- K-means clustering groups light curves by statistical features before training
- Cluster embeddings provide context to the BiLSTM, allowing it to learn different patterns for different stellar types
- **100% recall** - the model detects every planet with zero false negatives

---

## Quick Start (Using Pre-Trained Model)

**A pre-trained model is included in this submission.** To run inference without training:

```powershell
# 1. Set up environment
conda create -n exo-lstm-gpu python=3.10
conda activate exo-lstm-gpu
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy pandas scikit-learn tqdm

# 2. Run inference with pre-trained model
python inference_cluster_model.py --model_path "models/best.pt" --windows_dir "data/windows_test" --output_file "predictions.csv"

# 3. Or evaluate on test set (requires test data)
python evaluate_test.py --model_path "models/best.pt" --test_dir "data/windows_sector1_full/test"
```

**Included files:**
- `models/best.pt` - Pre-trained model checkpoint (35 MB)
- `models/config.json` - Hyperparameters used for training
- `models/FINAL_RESULTS.md` - Detailed results summary

---

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX 3060 Ti 8GB, RTX 5070 Ti 16GB)
- Minimum 16 GB RAM
- ~50 GB disk space for data (if downloading raw data)

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

### Obtaining the Raw Data

The TESS Sector 1 ground truth data can be obtained from:
1. **ExoFOP-TESS**: https://exofop.ipac.caltech.edu/tess/
2. **MAST Portal**: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
3. **Lightkurve Python package**: `pip install lightkurve`

The ground truth labels are from the TESS Objects of Interest (TOI) catalog and community vetting efforts.

### Data Directory Structure
After processing, the data should be organized as:
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

### Building Training Windows from Raw Data

If you have the raw TESS Sector 1 ground truth files:
```powershell
python build_windows_from_groundtruth.py `
  --data_dir "path/to/sector-1/ground-truth" `
  --output_dir "data/windows_sector1_full" `
  --seq_len 2048 `
  --n_windows 3 `
  --seed 42
```

---

## Training (Optional - Pre-Trained Model Included)

To train a new model from scratch:
```powershell
conda activate exo-lstm-gpu

python train_bilstm_cluster.py `
  --windows_dir "data/windows_sector1_full/train" `
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

**Expected output:**
- Training takes ~30 minutes on RTX 5070 Ti (batch size 128)
- Model checkpoints saved to `runs/my_model/`
- Best model: `runs/my_model/best.pt`

---

## Project Structure

```
Code/
├── README.md                          # This file
│
├── # PRE-TRAINED MODEL
├── models/
│   ├── best.pt                        # Trained model checkpoint (USE THIS)
│   ├── config.json                    # Training hyperparameters
│   └── FINAL_RESULTS.md               # Results summary
│
├── # CORE SCRIPTS
├── train_bilstm_cluster.py            # Main training script (BiLSTM + K-means)
├── inference_cluster_model.py         # Run inference with trained model
├── evaluate_test.py                   # Evaluate model on test set
├── build_windows_from_groundtruth.py  # Build training windows from raw data
│
├── # OPTIONAL SCRIPTS
├── optuna_optimize.py                 # Hyperparameter optimization with Optuna
├── generate_final_figures.py          # Generate result visualizations
└── simulate_training_demo.py          # Demo visualization for presentation
```

---

## Model Architecture

### ClusterBiLSTM (Final Architecture)

```
Input: (batch, 2048, 1) - Normalized flux window
  │
  ├── K-means Cluster Assignment (based on statistical features)
  │     Features: mean, std, variance, skewness, range, median, MAD, peak-to-peak
  │
  ├── Cluster Embedding Layer (7 clusters → 64 dimensions)
  │
  ├── BiLSTM (4 layers, 192 hidden units, bidirectional)
  │     Output: Forward + Backward hidden states (384 total)
  │
  ├── Concatenate [BiLSTM output (384) + Cluster embedding (64)] = 448
  │
  ├── FC1 (448 → 192) + BatchNorm + ReLU + Dropout(0.334)
  ├── FC2 (192 → 96) + BatchNorm + ReLU + Dropout(0.334)
  └── FC3 (96 → 1) + Sigmoid → Planet probability

Total Parameters: 3,068,801
```

### Final Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| seq_len | 2048 | Window length (flux points) |
| hidden_size | 192 | BiLSTM hidden units |
| num_layers | 4 | BiLSTM layers |
| n_clusters | 7 | K-means clusters |
| cluster_embed_dim | 64 | Cluster embedding dimension |
| dropout | 0.334 | Dropout rate |
| learning_rate | 0.0001 | Adam learning rate |
| batch_size | 128 | Training batch size |
| pos_weight | 7.41 | Class imbalance weight (neg/pos ratio) |

---

## Results

### Final Performance on TESS Sector 1 Test Set (December 6, 2025)

| Metric | Value |
|--------|-------|
| **AUC** | **0.9261** |
| **Recall** | **100%** |
| Accuracy | 83.26% |
| Precision | 39.93% |
| F1 Score | 0.5708 |

### Confusion Matrix
| | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actual Positive** | TP = 732 | FN = 0 |
| **Actual Negative** | FP = 1,101 | TN = 4,746 |

### Training Details
- **Dataset**: 26,472 training windows, 6,579 test windows
- **Class distribution**: 11.9% planets, 88.1% non-planets
- **Training time**: ~30 minutes (14 epochs until NaN, best at epoch 2) on RTX 5070 Ti
- **Best model**: `runs/sector1_final_0918/best.pt`

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
