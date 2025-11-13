# Methodology: Cluster-Enhanced BiLSTM for Exoplanet Transit Detection

## 1. Overview

This study presents a deep learning approach for detecting exoplanetary transits in stellar light curve data from NASA's TESS (Transiting Exoplanet Survey Satellite) mission. We employ a Bidirectional Long Short-Term Memory (BiLSTM) network enhanced with K-means clustering to classify time-series windows as containing planetary transits or non-planetary signals.

## 2. Data Collection and Preprocessing

### 2.1 Dataset Composition

The dataset consists of stellar light curves from two primary sources:
- **Positive examples**: 100 confirmed exoplanet host stars from TESS/Kepler archives
- **Negative examples**: Stellar light curves containing flares, noise, and non-planetary variability
- **Total windows**: 655 time-series windows (150 positive, 505 negative)
- **Class distribution**: 23% positive class, requiring careful handling of class imbalance

### 2.2 Light Curve Preprocessing

Raw light curve data undergoes a multi-stage preprocessing pipeline:

1. **Robust Detrending**: Remove long-term stellar variability using robust statistical methods
2. **Median Normalization**: Normalize flux values to median = 1.0
3. **Z-score Standardization**: Standardize to zero mean and unit variance
4. **Quality Filtering**: Remove cadences with poor quality flags

### 2.3 Window Extraction Strategy

We employ a phase-folding strategy based on detected periodicities:

**Positive Windows** (3 per light curve):
- 1 window centered at exact transit phase (φ = 0.0)
- 2 windows with small jitter (±5% phase offset) for augmentation
- Each window contains 2048 consecutive flux measurements

**Negative Windows** (5 per positive window):
- Extracted from phase regions far from transits (|φ - 0.5| > 0.18)
- Ensures negative examples do not contain transit signals
- Maintains temporal structure of stellar noise

**Window length**: 2048 time points (selected to capture full transit events while maintaining computational efficiency)

## 3. Feature Extraction via Box Least Squares (BLS)

To enhance model learning, we extract physically-motivated features from each light curve using the Box Least Squares periodogram:

### 3.1 BLS Algorithm

BLS is a phase-folding algorithm optimized for detecting box-shaped transit signals. For each light curve, we compute:

- **Period (P)**: Orbital period in days
- **Transit depth (δ)**: Fractional brightness decrease
- **Transit duration (T)**: Length of transit event in days
- **BLS power (S/N)**: Signal-to-noise ratio of the detected periodicity
- **Reference epoch (t₀)**: Time of first detected transit

### 3.2 Feature Utility

These features capture the physical characteristics of planetary transits:
- **Period**: Distinguishes hot Jupiters (P ~ days) from temperate planets (P ~ weeks)
- **Depth**: Correlates with planet size (δ ≈ (Rₚ/R*)²)
- **Duration**: Relates to orbital geometry and stellar density
- **BLS power**: Quantifies signal strength and confidence

## 4. Clustering-Based Stratification

### 4.1 Motivation

Stellar systems exhibit diverse characteristics (stellar type, noise properties, signal strength). A single model may struggle to learn optimal decision boundaries across this heterogeneous space. We hypothesize that clustering windows by their physical features allows the model to specialize for different regimes.

### 4.2 Clustering Procedure

**Algorithm**: K-means clustering (k=5 clusters)

**Feature vector**: [period, depth, duration, BLS_power]

**Preprocessing**: StandardScaler (zero mean, unit variance) applied to features before clustering

**Cluster assignment**: Each window is assigned to the cluster with nearest centroid in feature space

### 4.3 Cluster Interpretation

The 5 clusters naturally separate windows into categories such as:
- High S/N, short-period transits (hot Jupiters)
- Low S/N, long-period transits (temperate planets)
- Deep transits with strong BLS detections
- Shallow transits with weak signals
- Noise-dominated windows

## 5. Neural Network Architecture

### 5.1 Model: ClusterBiLSTM

Our architecture combines sequence modeling with cluster-specific embeddings:

```
Input: (batch_size, 2048, 1) flux window + cluster_id

┌─────────────────────────────────────┐
│  Cluster Embedding Layer            │
│  Input: cluster_id (0-4)            │
│  Output: 32-dimensional embedding   │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  BiLSTM Layers (×3)                 │
│  - 256 hidden units per direction   │
│  - Bidirectional (512 total)        │
│  - Layer norm between layers        │
│  - Dropout = 0.4                    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Concatenate [LSTM_out, embedding]  │
│  Dimension: 512 + 32 = 544          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  FC1: 544 → 256                     │
│  + BatchNorm + ReLU + Dropout(0.4)  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  FC2: 256 → 128                     │
│  + BatchNorm + ReLU + Dropout(0.4)  │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  FC3: 128 → 1                       │
│  + Sigmoid → Probability [0, 1]     │
└─────────────────────────────────────┘
```

### 5.2 Key Architectural Choices

**Bidirectional LSTM**: Processes sequences in both forward and backward directions, allowing the model to leverage context from both past and future time points when evaluating each cadence.

**Cluster Embeddings**: Learnable 32-dimensional representations for each cluster. These embeddings are concatenated with the final LSTM hidden state, allowing the fully-connected layers to specialize their decision boundaries based on cluster membership.

**Layer Normalization**: Applied between LSTM layers to stabilize training with deep recurrent networks.

**Dropout Regularization**: Heavy dropout (p=0.4) prevents overfitting on the small dataset.

**Total Parameters**: ~2.1M trainable parameters

### 5.3 Why BiLSTM?

- **Temporal dependencies**: Captures the characteristic shape of transit light curves (ingress → minimum → egress)
- **Variable-length patterns**: Transit durations vary widely (hours to days)
- **Context sensitivity**: Distinguishes transits from stellar flares and instrumental artifacts by examining surrounding baseline
- **Bidirectionality**: Critical for identifying symmetric transit shapes

## 6. Training Procedure

### 6.1 Data Splitting

- **Training set**: 85% (557 windows)
- **Validation set**: 15% (98 windows)
- **Stratified split**: Maintains 23% positive class ratio in both sets

### 6.2 Loss Function: Weighted Binary Cross-Entropy

Given severe class imbalance (23% positive), we apply class weighting:

**Loss**: L = -[w₁·y·log(ŷ) + w₀·(1-y)·log(1-ŷ)]

Where:
- w₁ = 3.367 (positive class weight = 505/150)
- w₀ = 1.0 (negative class weight)

This weighting increases the penalty for misclassifying rare positive examples.

### 6.3 Optimization

- **Optimizer**: AdamW (Adam with decoupled weight decay)
- **Learning rate**: 1×10⁻⁴
- **Weight decay**: 1×10⁻⁵
- **LR scheduler**: Cosine annealing over 80 epochs
- **Gradient clipping**: max_norm = 1.0 (prevents exploding gradients)
- **Batch size**: 64

### 6.4 Mixed Precision Training

- **Precision**: FP16 (automatic mixed precision)
- **Device**: CUDA-enabled GPU
- **Speedup**: ~1.6× faster per epoch vs FP32 (25s vs 40s)
- **Gradient scaling**: Applied to maintain numerical stability

### 6.5 Training Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 80 |
| Hidden units | 256 |
| LSTM layers | 3 |
| Dropout | 0.4 |
| Cluster embed dim | 32 |
| Batch size | 64 |
| Learning rate | 1×10⁻⁴ |
| Pos weight | 3.367 |
| Gradient clip | 1.0 |

### 6.6 Early Stopping

- **Metric**: Validation AUC (Area Under ROC Curve)
- **Patience**: 15 epochs without improvement
- **Checkpoint**: Model with best validation AUC saved

### 6.7 Reproducibility

- **Random seed**: 42 (fixed for NumPy, PyTorch, Python)
- **Deterministic algorithms**: Enabled where possible
- **Hardware**: Windows 11, NVIDIA GPU (CUDA), conda environment

## 7. Evaluation Metrics

### 7.1 Primary Metric: AUC-ROC

**Area Under the Receiver Operating Characteristic Curve**

- Range: [0, 1], where 0.5 = random guessing, 1.0 = perfect classifier
- **Advantages**: Threshold-independent, robust to class imbalance
- **Interpretation**: Probability that a randomly chosen positive example ranks higher than a randomly chosen negative example

### 7.2 Secondary Metrics

**Precision**: Of predicted planets, what fraction are true planets?
- P = TP / (TP + FP)

**Recall (Sensitivity)**: Of true planets, what fraction did we detect?
- R = TP / (TP + FN)

**F1 Score**: Harmonic mean of precision and recall
- F1 = 2PR / (P + R)

**Accuracy**: Overall fraction of correct predictions
- Acc = (TP + TN) / (TP + TN + FP + FN)
- Note: Less meaningful with imbalanced classes

### 7.3 Confusion Matrix

```
              Predicted
              0    1
Actual  0   [TN   FP]
        1   [FN   TP]
```

## 8. Inference Pipeline

For new TESS data, the complete pipeline consists of:

1. **Download**: Retrieve TESS light curves via MAST API
2. **Preprocess**: Apply detrending, normalization, quality filtering
3. **Convert**: Export to CSV format (time, flux columns)
4. **Window extraction**: Build 2048-point sliding windows
5. **Feature extraction**: Compute BLS features
6. **Cluster assignment**: Use saved K-means model to assign clusters
7. **Prediction**: Run BiLSTM model to generate probabilities
8. **Aggregation**: Average predictions across windows for each star

## 9. Model Checkpointing

Saved checkpoints include:
- `model_state_dict`: Neural network weights
- `config`: All training hyperparameters
- `scaler_params`: StandardScaler mean and scale for feature normalization
- `kmeans_centers`: K-means cluster centroids
- `val_metrics`: Best validation performance metrics

This ensures complete reproducibility of clustering and inference.

## 10. Innovation and Contributions

**Key Innovation**: Cluster-enhanced BiLSTM architecture
- Without clustering: AUC ~ 0.67 (baseline BiLSTM)
- With clustering: AUC ~ 0.69 (+3% improvement)

**Contributions**:
1. Novel integration of physical feature clustering with deep sequence models
2. Demonstrated effectiveness on real TESS data (TIC 307210830 validation)
3. End-to-end reproducible pipeline for exoplanet transit detection
4. Mixed-precision training optimization for resource-limited environments

## 11. Limitations and Future Work

**Current Limitations**:
- Small dataset (655 windows, 150 positives)
- Model performance (AUC 0.69) below production threshold (~0.85)
- Limited to Windows environment
- No ensemble methods

**Future Directions**:
- Expand dataset with additional TESS sectors and Kepler data
- Implement attention mechanisms to identify critical transit phases
- Explore ensemble methods (model averaging, stacking)
- Investigate transfer learning from simulated to real light curves
- Add multi-task learning (predict period, depth jointly with classification)
