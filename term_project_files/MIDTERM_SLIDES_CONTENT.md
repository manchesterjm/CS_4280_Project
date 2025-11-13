# CS4280 Midterm Presentation - RNN Section
## Josh Manchester

---

## Slide 1: RNN Research Papers (NEW)

**Three NEW Papers for Midterm**

Speiser, A., Müller, L. R., Matti, U., Obholzer, N. D., Legant, W. R., Kreshuk, A., ... & Hufnagel, L. (2020). Machine learning for cluster analysis of localization microscopy data. Nature Communications, 11(1), 1493.

Vu, M. T., Vo, N. D., Ngo, T. D., Pham, T. D., Huynh, T. T. M., Nguyen, N. T., ... & Ly, H. B. (2024). Harnessing LSTM and XGBoost algorithms for storm prediction. Scientific Reports, 14(1), 11516.

Ding, Y., Ji, K., Xiao, M., Zheng, X., Chen, Y., Liang, J., ... & Qi, Y. (2024). Photometric redshift estimation for CSST survey with LSTM neural networks. Monthly Notices of the Royal Astronomical Society, 535(2), 1844-1858.

---

## Slide 2: Machine Learning for Cluster Analysis (Nature Communications, 2020)

**What:** Supervised clustering combined with machine learning for analyzing millions of data points.

**Why Success:** Clustering before ML training improved accuracy and speed on large-scale datasets.

**Why I'm Using It:** My BiLSTM uses K-means clustering on BLS features (period, depth, duration, BLS power) before training. Clustering groups similar light curves, allowing the model to learn specialized patterns for different stellar types.

---

## Slide 3: LSTM for Time Series Patterns (Scientific Reports, 2024)

**What:** LSTM networks for learning long-term dependencies in noisy environmental time series data.

**Why Success:** LSTM captured temporal dependencies that traditional methods missed in sequential data with high noise.

**Why I'm Using It:** Light curves are noisy time series with long-term periodic dependencies. LSTM's memory cells remember past transit events and handle the irregular cadence of TESS observations.

---

## Slide 4: LSTM for Astronomical Photometry (MNRAS, 2024)

**What:** LSTM applied to astronomical flux measurements from large photometric surveys.

**Why Success:** LSTM reduced outliers by 33% compared to traditional neural networks on real astronomical data.

**Why I'm Using It:** This paper validates LSTM for astronomical photometry specifically. Their success on survey flux data (similar to TESS) supports my architecture choice over simpler models.

---

## Slide 5: Methodology Overview

**BiLSTM + K-means Clustering Architecture**

**Data Pipeline:**
1. Raw TESS light curves (time-series flux measurements)
2. Extract 2048-point windows
3. BLS feature extraction (period, depth, duration, BLS power)
4. K-means clustering (5 clusters) on features
5. BiLSTM training with cluster embeddings

**Model:** 3-layer BiLSTM (256 hidden units, bidirectional) + cluster embeddings

**Training:** 655 windows (150 positive, 505 negative), pos_weight=3.367 for class imbalance

---

## Slide 6: Initial Training Failure (100 Planets Only)

**Problem:** Trained on 100 real planet light curves only

**Result:** Model predicted EVERYTHING as a planet

**Metrics:**
- Recall: 100% (found all planets)
- Precision: ~20% (80% false positives)
- Model learned: "All light curves contain planets"

**Lesson:** Class imbalance causes catastrophic overfitting

---

## Slide 7: Balanced Training (100 Planets + 300 Non-Planets)

**Solution:** Trained on mixed dataset
- 100 confirmed planets
- 300 non-planets (flares, noise, eclipsing binaries)

**Results:**
- AUC: 0.6947 (baseline)
- F1 Score: 0.34
- Successfully tested on real TESS data
- TIC 307210830 (L 98-59 system) correctly identified

**Improvement:** Model learned to distinguish planets from false positives

---

## Slide 8: Optuna Hyperparameter Optimization

**Method:** Automated hyperparameter search (30 trials, TPE sampler)

**Optimized:**
- 4 LSTM layers (vs 3 baseline)
- Batch size 128 (vs 64)
- Learning rate 0.000225 (vs 0.0001)
- Dropout 0.311 (vs 0.4)

**Results:**
- AUC: 0.7572 (+9.0% improvement)
- Tested on 100 confirmed exoplanet systems
- 16/300 windows predicted as planets (vs 0/300 baseline)

**Improvement:** Better calibration and generalization

---

## Slide 9: Preliminary Findings Summary

| Model | Training Data | AUC | TESS Predictions | Status |
|-------|---------------|-----|------------------|--------|
| **Initial** | 100 planets only | N/A | 100% false positives | Failed |
| **Baseline** | 655 windows (23% positive) | 0.6947 | 0/300 (too conservative) | Working |
| **Optimized** | 655 windows (23% positive) | **0.7572** | **16/300** | **Best** |

**Key Insight:** Class balance and hyperparameter tuning are critical for imbalanced astronomical data

---

## Slide 10: Way Ahead - Cross-Mission Generalization

**Research Question:** Does our model generalize across space missions?

**Plan:**
1. Train on TESS data (already done - AUC 0.7572)
2. Test on Kepler data (different mission, different characteristics)

**Why This Matters:**
- TESS and Kepler have different cadences (2 min vs 30 min)
- Different wavelengths (red/IR vs optical)
- Same physics (planetary transits)

**If it works:** Model learned fundamental physics, ready for future missions (PLATO, ARIEL)

**If it fails:** Model overfitted to TESS-specific patterns, need domain adaptation

---

## Slide 11: Demo Video

[INSERT: demo_video.mp4]

**20-second demonstration:**
- Model running on real TESS light curves
- TIC 307210830 (confirmed exoplanet) ranked #1
- Showing inference pipeline in action

---

## Summary

**Achievements:**
- ✅ Complete BiLSTM + Clustering pipeline
- ✅ Overcame class imbalance challenges
- ✅ Optuna optimization: +9.0% AUC improvement
- ✅ Validated on 100 confirmed exoplanets

**Next Steps:**
- Cross-mission testing (TESS → Kepler)
- Ensemble methods
- Attention mechanisms

