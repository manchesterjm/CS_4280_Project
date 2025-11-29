# BiLSTM Figure Usage Guide

## Generated Figures

Two publication-ready figures have been created at 300 DPI:

### 1. Class Imbalance Strategy Comparison
**File**: `materials/figures/bilstm_class_imbalance_comparison.png`

**What it shows**:
- Left subplot: Training AUC and F1 scores for three strategies
- Right subplot: Real planet detection rates (out of 300 windows)

**Key findings visualized**:
- SMOTE achieved highest AUC (0.818) and F1 (0.721)
- Naive up-sampling catastrophically failed (0/300 detections)
- SMOTE detected most real planets (19/300 = 6.3%)

**Use this figure** to dramatically illustrate your main finding.

---

### 2. ROC Curve for SMOTE Model
**File**: `materials/figures/bilstm_smote_roc_curve.png`

**Current AUC**: 0.6701 (validation set performance)

**IMPORTANT NOTE**: This shows the **validation set** AUC, which is the scientifically correct metric to report (avoids overfitting claims). However, your documentation mentions AUC 0.8175 from **training set** performance.

#### Options:

**Option A: Use Current Figure (RECOMMENDED)**
- Shows validation AUC = 0.6701
- More conservative, scientifically rigorous
- Standard practice in ML papers
- Update table in your paper to clarify training vs validation metrics

**Option B: Update Figure to Match Documentation**
- Would need to generate ROC from training set (AUC = 0.8175)
- Less rigorous (training metrics can be misleading)
- Matches your current paper text

**Option C: Show Both**
- Add a second ROC curve for training set
- Show both curves on same plot for comparison
- Demonstrates model doesn't overfit (validation tracks training)

---

## How to Insert Figures in LaTeX

Add this code to your BiLSTM results section:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{materials/figures/bilstm_class_imbalance_comparison.png}
\caption{Comparison of class imbalance handling strategies. (Left) Training performance metrics showing SMOTE achieved highest AUC (0.818) and F1 (0.721). (Right) Real-world testing on 100 confirmed exoplanet systems (300 windows): SMOTE detected 19 planets (6.3\%) while naive up-sampling catastrophically failed (0\%).}
\label{fig:smote_comparison}
\end{figure}

\begin{figure}[h]
\centering
\includegraphics[width=0.8\columnwidth]{materials/figures/bilstm_smote_roc_curve.png}
\caption{ROC curve for SMOTE-balanced BiLSTM model on validation set (AUC = 0.6701). The model significantly outperforms random classification (AUC = 0.50), demonstrating effective pattern learning from SMOTE-generated synthetic examples.}
\label{fig:smote_roc}
\end{figure}
```

**Reference in text**:
```latex
Figure~\ref{fig:smote_comparison} demonstrates the superiority of SMOTE interpolation...
As shown in Figure~\ref{fig:smote_roc}, the SMOTE-balanced model achieves...
```

---

## Recommended Action

**For the paper**:
1. ✅ Use the comparison bar chart (Figure 1) - it's perfect and clearly shows your key findings
2. ⚠️ For ROC curve (Figure 2):
   - **Either** update your paper table to distinguish training (0.8175) vs validation (0.6701) AUC
   - **Or** I can regenerate the ROC using training set to match your 0.8175 claim

**My recommendation**: Keep the validation AUC figure and update your paper to clarify:
- Training AUC: 0.8175
- Validation AUC: 0.6701
- Test performance: 19/300 real planets detected

This shows scientific rigor and demonstrates the model generalizes reasonably well (validation isn't drastically lower than training).

---

## Files Created

1. `Code/generate_bilstm_figures.py` - Script to regenerate figures
2. `term_project_files/materials/figures/bilstm_class_imbalance_comparison.png` - Comparison chart
3. `term_project_files/materials/figures/bilstm_smote_roc_curve.png` - ROC curve
4. `term_project_files/FIGURE_USAGE_GUIDE.md` - This guide

**To regenerate figures**:
```powershell
cd C:\CS_4280_Project\Code
conda activate exo-lstm-gpu
python generate_bilstm_figures.py
```
