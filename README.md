# CS 4280 - Exoplanet Detection using Deep Learning

## Project Overview

This project uses deep learning (BiLSTM with K-means clustering) to detect exoplanets from stellar light curve data. The model analyzes time-series brightness measurements from NASA's TESS mission to identify the characteristic dips caused by planetary transits.

## FINAL STATUS: COMPLETE ✅ (December 6, 2025)

### Final Test Results

| Metric | Value |
|--------|-------|
| **AUC** | **0.9261** (92.61%) |
| **Recall** | **100%** (all planets detected) |
| **F1 Score** | 0.5708 |
| **Precision** | 39.93% |
| **Accuracy** | 83.26% |

### Confusion Matrix
- **TP = 732** (all planets correctly identified)
- **FP = 1,101** (false alarms for follow-up)
- **TN = 4,746** (correct rejections)
- **FN = 0** (zero missed planets!)

### Key Achievement
**100% recall** means the model detects every planet in the test set with zero false negatives. For exoplanet detection, this is critical - missing a real planet is worse than flagging a candidate for verification.

### Improvement History
| Version | Date | AUC | Notes |
|---------|------|-----|-------|
| Baseline | Oct 2025 | 0.6947 | Initial BiLSTM |
| Optuna optimized | Nov 2025 | 0.7572 | +9.0% |
| **Final (Sector 1)** | **Dec 2025** | **0.9261** | **+33.3%** |

---

## Project Structure

```
CS_4280_Project/
├── Code/
│   ├── train_bilstm_cluster.py      # Main training script
│   ├── inference_cluster_model.py   # Run inference
│   ├── evaluate_test.py             # Evaluate on test set
│   ├── generate_final_figures.py    # Generate publication figures
│   ├── runs/
│   │   └── sector1_final_0918/      # Final model checkpoint
│   │       ├── best.pt              # Best model (AUC 0.9261)
│   │       └── FINAL_RESULTS.md     # Detailed results
│   └── reports/                     # Prediction outputs
│
├── term_project_files/              # LaTeX paper and figures
│   ├── Merged_Proposal_as_of_12.2.2025.tex
│   └── Images/RNN/                  # Publication figures
│
├── CLAUDE.md                        # Development guide
├── NEXT_SESSION_QUICKSTART.md       # Quick reference
└── PROGRESS_LOG_DEC_6_2025.md       # Session log
```

---

## Quick Start

### Environment Setup
```powershell
conda activate exo-lstm-gpu
cd D:\CS_4280_Project\Code
```

### Evaluate the Final Model
```powershell
python evaluate_test.py `
  --model_path "runs\sector1_final_0918\best.pt" `
  --test_dir "D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\test"
```

### Run Inference on New Data
```powershell
python inference_cluster_model.py `
  --model_path "runs\sector1_final_0918\best.pt" `
  --windows_dir "path\to\new\windows" `
  --output_file "reports\predictions.csv"
```

### Generate Figures
```powershell
python generate_final_figures.py
```

---

## Final Model Configuration

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
| Parameters | 3,068,801 |

---

## Dataset: TESS Sector 1 Ground Truth

| Split | Windows | Planets | Non-Planets | % Positive |
|-------|---------|---------|-------------|------------|
| Train | 26,472 | 3,147 | 23,325 | 11.9% |
| Test | 6,579 | 732 | 5,847 | 11.1% |
| **Total** | **33,051** | **3,879** | **29,172** | **11.7%** |

---

## Deadlines

| Task | Date | Status |
|------|------|--------|
| Final model training | Dec 6 | ✅ DONE |
| Presentations | Dec 9-11 | Pending |
| Final submission | Dec 18 | Pending |

---

## Contact

**Author**: Josh Manchester
**Course**: CS 4280 - Machine Learning
**Term**: Fall 2025
**Component**: RNN (BiLSTM + Clustering)

---

**Last Updated**: December 6, 2025
