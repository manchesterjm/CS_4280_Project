# RNN Section - Midterm Presentation
## Josh Manchester

---

## SLIDE 1: Related Work - Three NEW Papers

Speiser, A., Müller, L. R., Matti, U., Obholzer, N. D., Legant, W. R., Kreshuk, A., ... & Hufnagel, L. (2020). Machine learning for cluster analysis of localization microscopy data. Nature Communications, 11(1), 1493. **(H5 index: 399)**

Vu, M. T., Vo, N. D., Ngo, T. D., Pham, T. D., Huynh, T. T. M., Nguyen, N. T., ... & Ly, H. B. (2024). Harnessing LSTM and XGBoost algorithms for storm prediction. Scientific Reports, 14(1), 11516. **(H5 index: 234)**

Ding, Y., Ji, K., Xiao, M., Zheng, X., Chen, Y., Liang, J., ... & Qi, Y. (2024). Photometric redshift estimation for CSST survey with LSTM neural networks. Monthly Notices of the Royal Astronomical Society, 535(2), 1844-1858. **(H5 index: 151)**

---

## SLIDE 2: Why BiLSTM + Clustering?

**From the Papers:**
• Clustering improves ML on large datasets (Speiser 2020)
• LSTM captures long-term dependencies (Vu 2024)
• LSTM reduces outliers in astronomy (Ding 2024)

**My Approach:**
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
✓ 100 planets + 300 non-planets → AUC: 0.69
✓ Real TESS: Identified TIC 307210830 (prob: 0.5959)

**But: Imbalanced Data Problem**
❌ 150 planets vs 505 non-planets (23% positive)
❌ High recall (88.67%) but low precision (38.27%)
❌ Too many false positives → Model biased toward negatives

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

**Balanced Data Attempt Failed**
• Tried 50/50 balanced synthetic data (200 planets + 200 non-planets)
• AUC dropped to 0.45 on real TESS data (domain shift)

**Solution: Hybrid Training**
• Mix real TESS + synthetic (90/10 ratio)
• Better balance than 23% but maintains domain fidelity

**Cross-Mission Testing**
Train on TESS → Test on Kepler to verify generalization

---
