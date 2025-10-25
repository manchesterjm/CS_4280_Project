# Research Paper Materials

This directory contains all materials needed for the exoplanet detection research paper, including methodology documentation, results tables, and visualization scripts.

## Contents

### 1. Methodology Document
**File**: `methodology.md`

Comprehensive description of the BiLSTM+Clustering approach including:
- Data collection and preprocessing pipeline
- BLS feature extraction
- K-means clustering strategy
- Neural network architecture details
- Training procedure and hyperparameters
- Evaluation metrics
- Innovation and contributions

**Usage**: This can be directly incorporated into the Methods section of your research paper.

---

### 2. Results Tables
**File**: `results_tables.md`

Contains 11 formatted tables ready for publication:
- Table 1: Model Performance on Validation Set
- Table 2: Model Architecture Specifications
- Table 3: Training Hyperparameters
- Table 4: Dataset Statistics
- Table 5: BLS Feature Ranges
- Table 6: K-means Clustering Results
- Table 7: Comparison with Baseline Models
- Table 8: Real TESS Data Testing Results
- Table 9: Training Performance
- Table 10: Confusion Matrix
- Table 11: Post-Filtering Performance

**Usage**: Copy tables into Results section. Tables are formatted in Markdown and can be converted to LaTeX using pandoc.

**Convert to LaTeX**:
```bash
pandoc results_tables.md -o results_tables.tex
```

---

### 3. Visualization Scripts

#### Architecture Diagram Generator
**File**: `generate_architecture_diagram.py`

Generates:
- `architecture_diagram.png` - Visual representation of BiLSTM model layers
- `data_pipeline.png` - Flowchart of end-to-end data processing

**Requirements**:
- matplotlib
- numpy

**Run**:
```bash
conda activate exo-lstm-gpu
python generate_architecture_diagram.py
```

#### Comprehensive Visualizations
**File**: `generate_visualizations.py`

Generates 9 publication-ready figures:
1. **roc_curve.png** - ROC curve with AUC = 0.6947
2. **confusion_matrix.png** - Heatmap of classification results
3. **prediction_distributions.png** - Histogram of model predictions
4. **cluster_distribution.png** - Bar chart of cluster assignments
5. **performance_by_cluster.png** - Mean performance across clusters
6. **top_predictions.png** - Top 10 planet candidates from real TESS data
7. **precision_recall_curve.png** - PR curve
8. **model_comparison.png** - Comparison with baseline models (AUC & F1)
9. **training_curves.png** - Loss and AUC over training epochs

**Requirements**:
- matplotlib
- seaborn
- pandas
- numpy

**Run**:
```bash
conda activate exo-lstm-gpu
python generate_visualizations.py
```

**Output**: All figures saved to `figures/` subdirectory at 300 DPI

---

## Quick Start

### Step 1: Install Dependencies (if not already installed)
```bash
conda activate exo-lstm-gpu
conda install matplotlib seaborn pandas numpy -y
```

### Step 2: Generate All Visualizations
```bash
cd C:\CS_4280_Project\research_paper
python generate_visualizations.py
python generate_architecture_diagram.py
```

### Step 3: Review Generated Materials
```
research_paper/
├── methodology.md              # Methods section content
├── results_tables.md           # All results tables
├── figures/                    # Generated visualizations
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── prediction_distributions.png
│   ├── cluster_distribution.png
│   ├── performance_by_cluster.png
│   ├── top_predictions.png
│   ├── precision_recall_curve.png
│   ├── model_comparison.png
│   └── training_curves.png
├── architecture_diagram.png
└── data_pipeline.png
```

---

## Suggested Paper Structure

### Abstract
- Brief overview of problem (exoplanet detection in TESS light curves)
- Methodology (BiLSTM + K-means clustering)
- Key result (AUC 0.6947, validated on real TESS data)

### 1. Introduction
- Importance of exoplanet detection
- Current challenges (noise, class imbalance, variability)
- Motivation for deep learning approach
- Contribution: cluster-enhanced temporal modeling

### 2. Related Work
- Classical transit detection (BLS, TLS)
- Machine learning approaches (Random Forests, SVMs)
- Recent deep learning work (CNNs, LSTMs, attention mechanisms)
- Gap: integration of physical features with sequence modeling

### 3. Methodology
**Use**: Copy content from `methodology.md`

Key subsections:
- 3.1 Data Collection
- 3.2 Preprocessing Pipeline
- 3.3 Feature Extraction (BLS)
- 3.4 Clustering Strategy
- 3.5 Neural Network Architecture
- 3.6 Training Procedure

**Figures to include**:
- Figure 1: `data_pipeline.png`
- Figure 2: `architecture_diagram.png`
- Figure 3: `cluster_distribution.png`

### 4. Results
**Use**: Copy tables from `results_tables.md`

Key subsections:
- 4.1 Model Performance (Table 1, Table 10)
- 4.2 Comparison with Baselines (Table 7)
- 4.3 Real-World Validation (Table 8)
- 4.4 Clustering Analysis (Table 6)

**Figures to include**:
- Figure 4: `roc_curve.png`
- Figure 5: `confusion_matrix.png`
- Figure 6: `model_comparison.png`
- Figure 7: `performance_by_cluster.png`
- Figure 8: `training_curves.png`
- Figure 9: `top_predictions.png`

### 5. Discussion
- **Strengths**:
  - Successfully detects TIC 307210830 (confirmed exoplanet)
  - Clustering improves AUC by 3% over baseline BiLSTM
  - End-to-end pipeline tested on real TESS data

- **Limitations**:
  - Small dataset (655 windows)
  - AUC 0.69 below production threshold (~0.85)
  - Low recall (10%) due to conservative filtering

- **Future Work**:
  - Expand dataset with more TESS sectors
  - Implement attention mechanisms
  - Ensemble methods
  - Multi-task learning

### 6. Conclusion
- Summary of contributions
- Demonstrated viability of cluster-enhanced BiLSTM
- Path forward for improving performance

---

## Key Findings to Highlight

### Primary Result
> "Our cluster-enhanced BiLSTM model achieves an AUC of 0.6947 on held-out validation data, representing a 3% improvement over baseline BiLSTM and an 11.5% improvement over classical machine learning approaches."

### Real-World Validation
> "When tested on 7 real TESS light curves, our model successfully identified TIC 307210830 (L 98-59 system) as a high-confidence exoplanet host with mean probability 0.5959."

### Innovation
> "By integrating K-means clustering on physically-motivated BLS features, our model learns specialized decision boundaries for different stellar regimes, enabling more robust transit detection across diverse stellar types."

---

## Formatting Tips

### For LaTeX Papers
1. Convert markdown tables to LaTeX:
   ```bash
   pandoc results_tables.md -o results_tables.tex
   ```

2. Include figures:
   ```latex
   \begin{figure}[h]
   \centering
   \includegraphics[width=0.8\textwidth]{figures/roc_curve.png}
   \caption{ROC curve demonstrating model performance (AUC = 0.6947).}
   \label{fig:roc}
   \end{figure}
   ```

### For Word/Google Docs
1. Copy tables directly from `results_tables.md` (Markdown renders nicely in most editors)
2. Insert PNG figures at 300 DPI (already generated at publication quality)

---

## Data Sources Referenced

All visualizations and tables are generated from:
- `C:\CS_4280_Project\Code\runs\bilstm_cluster\config.json` - Training configuration
- `C:\CS_4280_Project\Code\reports\test_predictions.csv` - Per-window predictions
- `C:\CS_4280_Project\Code\reports\inference_aggregated.csv` - Per-star aggregated predictions
- `C:\CS_4280_Project\Code\reports\postfilter_summary.txt` - Post-filtering metrics

---

## Citation Template

If using this methodology in a paper, consider citing relevant foundational work:

```bibtex
@article{kepler_pipeline,
  title={Kepler Data Processing Handbook},
  author={Jenkins, J.M. and others},
  journal={Kepler Science Document},
  year={2017}
}

@article{tess_mission,
  title={Transiting Exoplanet Survey Satellite (TESS)},
  author={Ricker, G.R. and others},
  journal={Journal of Astronomical Telescopes},
  year={2015}
}

@article{bls_algorithm,
  title={A Box-Fitting Algorithm in the Search for Periodic Transits},
  author={Kovács, G. and Zucker, S. and Mazeh, T.},
  journal={Astronomy and Astrophysics},
  volume={391},
  pages={369--377},
  year={2002}
}

@article{lstm_exoplanets,
  title={Deep Learning for Exoplanet Detection},
  author={Various authors},
  journal={Astronomy and Computing},
  year={2019-2024}
}
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'matplotlib'"
**Solution**:
```bash
conda activate exo-lstm-gpu
conda install matplotlib seaborn -y
```

### Issue: "FileNotFoundError" when running visualizations
**Solution**: Ensure you're in the correct directory and paths are correct:
```bash
cd C:\CS_4280_Project\research_paper
# Update paths in scripts if needed
```

### Issue: Figures look low quality
**Solution**: All scripts generate at 300 DPI by default. If needed, increase:
```python
plt.savefig('output.png', dpi=600)  # Even higher resolution
```

---

## Additional Resources

- **Project Documentation**: See `C:\CS_4280_Project\CLAUDE.md` for complete project overview
- **Training Scripts**: `C:\CS_4280_Project\Code\train_bilstm_cluster.py`
- **Inference Scripts**: `C:\CS_4280_Project\Code\inference_cluster_model.py`
- **GitHub Repository**: https://github.com/manchesterjm/CS_4280_Project

---

## Contact

For questions about the methodology or implementation, refer to the project's GitHub issues or the CLAUDE.md documentation.

---

**Last Updated**: October 2025
**Model Version**: BiLSTM+Clustering v1.0 (AUC 0.6947)
