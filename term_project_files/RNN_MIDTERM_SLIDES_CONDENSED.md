# RNN Section - Midterm Presentation
## Josh Manchester

---

## SLIDE 1: Related Work - Three NEW Papers

**Speiser et al. (2020)** - Nature Communications **(H5: 399)**
Machine learning + clustering for large-scale data

**Vu et al. (2024)** - Scientific Reports **(H5: 234)**
LSTM for noisy time series patterns

**Ding et al. (2024)** - MNRAS **(H5: 151)**
LSTM for astronomical photometry (33% fewer outliers)

---

## SLIDE 2: Why BiLSTM + Clustering?

**From the Papers:**
• Clustering improves ML on large datasets (Speiser 2020)
• LSTM captures long-term dependencies (Vu 2024)
• LSTM reduces outliers in astronomy (Ding 2024)

**Our Approach:**
BiLSTM + K-means clustering on BLS features

---

## SLIDE 3: Methodology

**Data:** 655 windows (150 planets, 505 non-planets)

**Features:** Period, depth, duration, BLS power

**Architecture:**
• K-means (5 clusters)
• 4-layer BiLSTM (256 hidden, bidirectional)
• Cluster embeddings (32-dim)

**[INSERT: preprocessing_pipeline.png]**

---

## SLIDE 4: BiLSTM Architecture

**[INSERT: bilstm_architecture.png]**

Input (2048) → BiLSTM Layers → Cluster Embedding → FC Layers → Output

~2.1M parameters

---

## SLIDE 5: Results

**[LEFT: metrics_bar_chart.png]**
**[RIGHT: confusion_matrix.png]**

**AUC: 75.72%**
Recall: 88.67% | Precision: 38.27%

Tested on 100 confirmed exoplanet systems

---

## SLIDE 6: Learning from Failure

**100 Planets Only**
❌ Predicted everything as planet
❌ 100% false positives

**Solution: Add Non-Planets**
✓ 100 planets + 300 non-planets
✓ AUC: 0.69 → Model learned to distinguish

---

## SLIDE 7: Optuna Optimization

**[INSERT: model_progression.png]**

| Parameter | Baseline | Optimized |
|-----------|----------|-----------|
| Layers | 3 | 4 |
| Batch size | 64 | 128 |
| Learning rate | 1e-4 | 2.25e-4 |

**Result:** AUC 0.69 → **0.76** (+9%)

---

## SLIDE 8: Demo

**[INSERT: demo_video.mp4]**

Model identifying TIC 307210830 (L 98-59 multi-planet system)

---

## SLIDE 9: What's Next?

**Cross-Mission Testing**
Train on TESS → Test on Kepler

**Goal:** Verify generalization vs overfitting

**If works:** Model learned physics ✓
**If fails:** Need domain adaptation ⚠️

---
