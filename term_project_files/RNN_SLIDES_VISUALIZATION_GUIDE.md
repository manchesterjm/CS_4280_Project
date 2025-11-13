# RNN Midterm Slides - Visualization Guide

**Created**: November 12, 2025
**Location**: All visualizations saved to `C:\CS_4280_Project\term_project_files\materials\figures\rnn_slides\`
**Resolution**: 300 DPI (publication-ready)

---

## Slide-to-Visualization Mapping

### SLIDE 6: Methodology - BiLSTM + Clustering
**File**: `preprocessing_pipeline.png`

**Description**: Data preprocessing pipeline diagram showing:
1. Raw Light Curve
2. Cleaned + Detrended
3. Phase-Folded + Normalized
4. BLS Features Extracted
5. 2048-point Window

**Insert this diagram** at the bottom of Slide 6 after the text describing preprocessing steps.

---

### SLIDE 7: Methodology - BiLSTM + Clustering (Cont.)
**File**: `bilstm_architecture.png`

**Description**: Complete BiLSTM + Clustering architecture diagram showing:
- Input Window (2048 points)
- 4 BiLSTM Layers (256 hidden units each)
- Cluster Embedding (32-dim, shown in yellow box on left)
- Concatenation Layer (512+32)
- Fully Connected Layers (544→256→128→1)
- Sigmoid Output (Probability)
- Annotation: "~2.1M parameters"

**Insert this diagram** at the bottom of Slide 7 after the architecture specifications.

---

### SLIDE 8: Experimental Results
**Files**:
1. `metrics_bar_chart.png`
2. `confusion_matrix.png`

#### Figure 1: Performance Metrics Bar Chart
Shows 5 colored bars for:
- **AUC**: 75.72% (blue)
- **F1 Score**: 51.45% (red)
- **Recall**: 88.67% (green)
- **Precision**: 38.27% (orange)
- **Accuracy**: 68.37% (purple)

Includes horizontal line at 75% showing target threshold.

#### Figure 2: Confusion Matrix
Heatmap showing test set performance (98 windows):
- **TN (True Negatives)**: 44
- **FP (False Positives)**: 32
- **FN (False Negatives)**: 2
- **TP (True Positives)**: 20

**Layout for Slide 8**:
```
[LEFT SIDE]                    [RIGHT SIDE]
Training Data:                 Bar Chart (metrics_bar_chart.png)
655 windows total
• Training: 459 (70%)          [BELOW BAR CHART]
• Validation: 98 (15%)         Confusion Matrix (confusion_matrix.png)
• Test: 98 (15%)

Class Distribution:
• Planets: 22.9% (150)
• Non-planets: 77.1% (505)
```

---

### SLIDE 12: Model Comparison (Progressive Improvement Summary)
**File**: `model_progression.png`

**Description**: Bar chart comparing three training approaches:
- **Planets Only** (Failed): AUC ~50% (red bar) with ❌ Failed status
- **Baseline** (Working): AUC 69.47% (orange bar) with ✓ Working status
- **Optimized** (Best): AUC 75.72% (green bar) with ✓✓ Best status

Includes:
- Red dashed line at 50% (Random Baseline)
- Green dashed line at 80% (Target)

**Insert this chart** above or beside the comparison table on Slide 12.

---

## All Generated Files

| File | Size | Purpose | Slide |
|------|------|---------|-------|
| `metrics_bar_chart.png` | 300 DPI | Performance metrics comparison | Slide 8 |
| `confusion_matrix.png` | 300 DPI | Test set confusion matrix | Slide 8 |
| `model_progression.png` | 300 DPI | Training progression comparison | Slide 12 |
| `preprocessing_pipeline.png` | 300 DPI | Data pipeline flowchart | Slide 6 |
| `bilstm_architecture.png` | 300 DPI | Model architecture diagram | Slide 7 |

---

## Slide Content Summary (from RNN_MIDTERM_SLIDES_FINAL.md)

### Slides 1-4: Research Papers (NEW)
- Slide 1: Title page listing all 3 papers
- Slide 2: Speiser 2020 (Machine learning + clustering)
- Slide 3: Vu 2024 (LSTM for time series)
- Slide 4: Ding 2024 (LSTM for astronomical photometry)

### Slides 5-7: Methodology
- Slide 5: Progress Since Proposal
- Slide 6: Methodology overview + **preprocessing_pipeline.png**
- Slide 7: Architecture details + **bilstm_architecture.png**

### Slides 8-12: Results
- Slide 8: Experimental Results + **metrics_bar_chart.png** + **confusion_matrix.png**
- Slide 9: Initial Training Failure (100 planets only)
- Slide 10: Balanced Training Success (100 + 300)
- Slide 11: Optuna Optimization Results
- Slide 12: Model Comparison + **model_progression.png**

### Slides 13-14: Conclusion
- Slide 13: Demo video (demo_video.mp4)
- Slide 14: What's Next?

---

## Key Metrics (for Reference)

### Optimized Model Performance
- **AUC**: 0.7572 (75.72%)
- **F1 Score**: 0.5145 (51.45%)
- **Recall**: 0.8867 (88.67%)
- **Precision**: 0.3827 (38.27%)
- **Accuracy**: 0.6837 (68.37%)

### Training Data
- **Total windows**: 655
- **Training**: 459 (70%)
- **Validation**: 98 (15%)
- **Test**: 98 (15%)
- **Positive rate**: 22.9% (150 planets / 655 total)
- **Class weight**: pos_weight=3.367

### Model Architecture
- **Layers**: 4 BiLSTM layers
- **Hidden units**: 256 per layer (bidirectional)
- **Clusters**: 5 (K-means on BLS features)
- **Cluster embedding**: 32-dim
- **Total parameters**: ~2.1M
- **Dropout**: 0.311
- **Batch size**: 128
- **Learning rate**: 0.000225

### Optuna Optimization
- **Trials**: 30
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Improvement**: 0.6947 → 0.7572 (+9.0% AUC)

### Real-World Validation
- **Test set**: 100 confirmed TESS exoplanet systems
- **Windows**: 300 test windows
- **Baseline predictions**: 0/300 (too conservative)
- **Optimized predictions**: 16/300 (5.3%)
- **Top candidate**: TIC 261337380 (p=0.6666)

---

## Presentation Notes

### Visual Style
All graphs match the style from your group's CNN/Transformer slides:
- Clean, professional appearance
- High-contrast colors for visibility
- Clear labels and annotations
- Consistent fonts and sizing
- Publication-ready quality (300 DPI)

### Color Scheme
- **Blue**: Primary metric (AUC)
- **Red**: Warning/failure indicators
- **Green**: Success indicators
- **Orange**: Moderate performance
- **Purple**: Supporting metrics

### Text Integration
When inserting into PowerPoint/Google Slides:
1. Keep text "very spartan" as requested
2. Let visualizations speak for themselves
3. Use "Key Innovation" (light gray) and "Key Takeaway" (dark blue) boxes
4. Maintain consistent APA citation format at bottom

---

## Files Referenced

### Slide Content
- **RNN_MIDTERM_SLIDES_FINAL.md** - Complete slide text content (14 slides)
- **demo_video.mp4** - 20-second demonstration video (Slide 13)

### Data Sources
- **C:\CS_4280_Project\Code\runs\bilstm_cluster_optimized\** - Training metrics
- **C:\CS_4280_Project\Code\reports\optimized_planet_predictions.csv** - Test predictions

### Documentation
- **CLAUDE.md** - Project overview and commands
- **CURRENT_STATUS_NOV_11_2025.md** - Latest status
- **OPTUNA_OPTIMIZATION_SUMMARY.md** - Hyperparameter tuning results

---

**Status**: ✅ All visualizations complete and ready for presentation
**Next Step**: Insert images into PowerPoint/Google Slides following this guide

---

*Generated by Claude Code for CS 4280 Exoplanet Detection Project*
*November 12, 2025*
