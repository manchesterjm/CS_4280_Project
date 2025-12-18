# Final Model Results - December 6, 2025

## Model: BiLSTM + K-means Clustering

### Test Set Performance

| Metric | Value |
|--------|-------|
| **AUC** | **0.9261** (92.61%) |
| **F1 Score** | 0.5708 |
| **Precision** | 39.93% |
| **Recall** | **100%** |
| **Accuracy** | 83.26% |

### Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actual Positive** | TP = 732 | FN = 0 |
| **Actual Negative** | FP = 1101 | TN = 4746 |

- **Total test samples**: 6,579
- **Positive samples**: 732 (11.1%)
- **Negative samples**: 5,847 (88.9%)

### Validation Performance (Best Epoch)

| Metric | Value |
|--------|-------|
| **AUC** | 0.9199 |
| **F1 Score** | 0.5825 |
| **Accuracy** | 83.17% |

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| hidden_size | 192 |
| num_layers | 4 |
| n_clusters | 7 |
| cluster_embed_dim | 64 |
| dropout | 0.334 |
| learning_rate | 0.0001 |
| batch_size | 128 |
| pos_weight | 7.41 |
| epochs_trained | 14 (stopped due to NaN) |
| best_epoch | 2 |

---

## Dataset

### TESS Sector 1 Ground Truth

| Split | Windows | Planets | Non-Planets | % Positive |
|-------|---------|---------|-------------|------------|
| Train | 26,472 | 3,147 | 23,325 | 11.9% |
| Test | 6,579 | 732 | 5,847 | 11.1% |
| **Total** | **33,051** | **3,879** | **29,172** | **11.7%** |

### Clustering Distribution (Test Set)

| Cluster | Count | Percentage |
|---------|-------|------------|
| 0 | 207 | 3.1% |
| 1 | 3,198 | 48.6% |
| 2 | 990 | 15.0% |
| 3 | 99 | 1.5% |
| 4 | 495 | 7.5% |
| 5 | 708 | 10.8% |
| 6 | 882 | 13.4% |

---

## Model Architecture

```
ClusterBiLSTM (3,068,801 parameters)
├── Cluster Embedding: 7 clusters → 64 dimensions
├── BiLSTM: 4 layers, 192 hidden units, bidirectional
│   └── Output: 384 dimensions (192 × 2)
├── Combined: 448 dimensions (384 + 64)
└── Classifier (Sequential):
    ├── Dropout(0.334) → Linear(448→192) → BatchNorm → ReLU
    ├── Dropout(0.334) → Linear(192→96) → BatchNorm → ReLU
    └── Dropout(0.334) → Linear(96→1) → Sigmoid
```

---

## Key Findings

1. **100% Recall**: The model detects ALL planets with zero false negatives
2. **High AUC (0.926)**: Excellent discrimination between planets and non-planets
3. **Trade-off**: Lower precision (40%) means more false positives, but acceptable for screening
4. **Robust**: Test AUC (0.926) > Validation AUC (0.920), indicating good generalization

---

## Files

| File | Description |
|------|-------------|
| `best.pt` | Best model checkpoint (epoch 2, AUC 0.9199) |
| `last.pt` | Final checkpoint before NaN crash |
| `config.json` | Training configuration |
| `cluster_ids.npy` | Training cluster assignments |
| `training_log.txt` | Full training output |

---

## Improvement History

| Date | Model | AUC | Notes |
|------|-------|-----|-------|
| Oct 2025 | BiLSTM (655 windows) | 0.6947 | Original baseline |
| Nov 2025 | BiLSTM + Optuna | 0.7572 | +9.0% improvement |
| Dec 2025 | BiLSTM (Sector 1) | 0.9199 | Validation, 62× more data |
| Dec 2025 | **Final Model** | **0.9261** | **Test set, +33% vs baseline** |

---

**Generated**: December 6, 2025
**Model Path**: `runs/sector1_final_0918/best.pt`
