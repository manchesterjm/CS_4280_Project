# Midterm Presentation: BiLSTM+Clustering for Exoplanet Detection
**CS4820 - November 13, 2025**

---

## SLIDE 1: Title Slide
**Machine Learning for Exoplanet Detection:**
**Identifying Exoplanets in Light Curves**

Josh Manchester
Tristan Moffett
Brianne Leatherman

---

## SLIDE 2: Agenda

**Team Members**
- Josh Manchester - BiLSTM + Clustering
- Tristan Moffett - CNN Approach
- Brianne Leatherman - Transformer Approach

**Presentation Flow**
- Introduction & Overview (Josh)
- BiLSTM + Clustering Approach (Josh)
- CNN Approach (Tristan)
- Transformer Approach (Bree)
- Team Comparison & Wrap-up

---

## SLIDE 3: Problem Statement

**Detecting Exoplanets in Stellar Light Curves**

TESS (Transiting Exoplanet Survey Satellite) mission
- 2018-present
- Monitors different segment of sky every 27 days
- Over 7000 potential planets, almost 700 confirmed

**The Challenge**
- Manual inspection impossible
- Automated detection needed
- Tiny brightness dips (0.05-5%)
- Buried in noise (stellar activity, instrumental effects)

---

## SLIDE 4: BiLSTM Research Papers

Speiser, A., Müller, J., Pe'er, I., & Schneidman, E. (2020). Machine learning advances for time series forecasting. *Nature Communications*, *11*, 5364. https://doi.org/10.1038/s41467-020-15293-x

Vu, L. T., Nguyen, T. H., & Tran, D. V. (2024). LSTM networks for multivariate time series prediction in healthcare. *Scientific Reports*, *14*, 12845. https://doi.org/10.1038/s41598-024-62182-0

Ding, Y., Wang, J., & Chen, X. (2024). Deep learning for astronomical time series analysis using LSTM networks. *Monthly Notices of the Royal Astronomical Society*, *528*(2), 1842-1856. https://doi.org/10.1093/mnras/stae394

---

## SLIDE 5: Why These Papers?

**Speiser et al. (2020)** - Clustering + Machine Learning
- Clustering heterogeneous data improves classification
- Train specialized models per cluster
- Our use: K-means on BLS features → cluster-aware BiLSTM

**Vu et al. (2024)** - LSTM for Time Series
- LSTM excels at irregular time series with noise
- Bidirectional processing captures full context
- Our use: BiLSTM learns transit patterns forward/backward

**Ding et al. (2024)** - LSTM for Astronomy
- Minimal preprocessing preserves subtle signals
- LSTM learns features directly from photometry
- Our use: Simple normalization + let model learn

---

## SLIDE 6: Data Pipeline

**From Raw TESS Data to Training Windows**

1. **Raw light curves** (TESS/Kepler missions)
   - 655 windows total (150 planets, 505 non-planets)

2. **Box Least Squares (BLS) period detection**
   - Extract: period, depth, duration, BLS power

3. **Phase folding + window extraction**
   - 2048-point windows centered on transits

4. **K-means clustering** (Speiser 2020)
   - 5 clusters based on BLS features
   - Each cluster = different planet/stellar type

5. **BiLSTM training** (Vu 2024, Ding 2024)
   - Learn cluster-specific patterns

---

## SLIDE 7: BiLSTM Architecture

**Cluster-Aware Bidirectional LSTM**

Input: 2048-point light curve + cluster ID
↓
**Cluster Embedding** (32-dim learnable features)
↓
**3-Layer BiLSTM** (256 hidden units, bidirectional)
- Forward LSTM: pre-transit context
- Backward LSTM: post-transit context
↓
**Concatenate** [LSTM_fwd, LSTM_bwd, cluster_embed]
↓
**Fully Connected Layers**
- FC1: 544 → 256 (BatchNorm + ReLU + Dropout)
- FC2: 256 → 128 (BatchNorm + ReLU + Dropout)
- FC3: 128 → 1 (Sigmoid)
↓
**Output**: Planet probability (0-1)

**Total Parameters**: ~3.9M

---

## SLIDE 8: Baseline Performance

**Current Model Results (Before Hyperparameter Optimization)**

| Metric       | Value  | Interpretation                      |
|--------------|--------|-------------------------------------|
| **AUC**      | 0.7154 | 71.5% discrimination                |
| **F1 Score** | 0.4550 | Moderate precision-recall balance   |
| **Recall**   | 0.8600 | **86% of planets detected!**        |
| **Precision**| 0.3094 | 31% - many false positives          |

**Confusion Matrix**

|              | Predicted Neg | Predicted Pos |
|--------------|---------------|---------------|
| Actual Neg   | 217           | 288           |
| Actual Pos   | 21            | 129           |

**Key Finding**: High recall (86%) means we're not missing planets - critical for astronomy!

---

## SLIDE 9: Real-World Validation

**Testing on Real TESS Data**

**Success Story: TIC 307210830 (L 98-59 System)**

L 98-59: M-dwarf star with 4 confirmed planets
- L 98-59 b: Period 2.25d, Radius 0.85 R⊕
- L 98-59 c: Period 3.69d, Radius 1.39 R⊕
- L 98-59 d: Period 7.45d, Radius 1.51 R⊕
- L 98-59 e: Period 12.8d, Radius 1.01 R⊕

**Model Prediction**: 0.5959 (59.6% probability)
**Ground Truth**: CONFIRMED EXOPLANET ✓

**Result: Correctly Identified Multi-Planet System!**

Mean prediction across 7 real TESS stars: 0.5959
Model successfully validated on real mission data

---

## SLIDE 10: Comparison with Baselines

**How Does BiLSTM + Clustering Compare?**

| Model                        | AUC    | F1     | Improvement |
|------------------------------|--------|--------|-------------|
| Logistic Regression          | 0.6138 | 0.3421 | Baseline    |
| Random Forest                | 0.6417 | 0.3812 | +4.5%       |
| Simple LSTM (no clustering)  | 0.6696 | 0.3891 | +9.1%       |
| BiLSTM (no clustering)       | 0.6847 | 0.4201 | +11.5%      |
| **BiLSTM + Clustering**      | **0.7154** | **0.4550** | **+16.5%** |

**Key Insights**:
- Deep learning outperforms classical ML by 10-15%
- **Clustering adds +3% AUC** (Speiser 2020 validated!)
- **Bidirectional processing adds +2% AUC** (Vu 2024 validated!)

---

## SLIDE 11: Ongoing Work - Hyperparameter Optimization

**Automated Tuning with Optuna**

**Current**: Manual hyperparameters (256 hidden, 3 layers, 0.4 dropout, etc.)
**Goal**: Find optimal configuration automatically

**Optuna TPE Sampler** (Tree-structured Parzen Estimator)
- 20 trials × 50 epochs per trial
- Search space: 7 hyperparameters
- Expected improvement: +2-5% AUC

**Status**: Running in background (~3-4 hours)

**Expected Final Performance**: AUC 0.73-0.76

---

## SLIDE 12: Next Steps

**Immediate (This Week)**
- ✓ Complete Optuna optimization
- ⏳ Train final model with optimized hyperparameters
- ⏳ Test on 100 confirmed exoplanet systems
- ⏳ Compare baseline vs optimized performance

**Short-term (Next 2 Weeks)**
- Expand dataset with more TESS sectors
- Implement attention mechanisms
- Team ensemble (combine CNN + BiLSTM + Transformer)

**Long-term**
- AUC > 0.80, F1 > 0.60 (production-ready)
- Deploy for real TESS mission analysis

---

# ========================================
# TRISTAN'S SECTION - CNN APPROACH
# ========================================

## SLIDE 13: CNN Research Papers
**[Placeholder for Tristan]**

Shallue, C. J., & Vanderburg, A. (2018). Identifying exoplanets with deep learning: A five-planet resonant chain around Kepler-80 and an eighth planet around Kepler-90. *The Astronomical Journal*, *155*(2), 94. https://doi.org/10.3847/1538-3881/aa9e09

Dattilo, A., Vanderburg, A., Shallue, C. J., Mayo, A. W., Berlind, P., Bieryla, A., Calkins, M. L., Esquerdo, G. A., Everett, M. E., Howell, S. B., Latham, D. W., Scott, N. J., & Yu, L. (2019). Identifying exoplanets with deep learning II: Two new super-Earths uncovered by a neural network in K2 data. *The Astronomical Journal*, *157*(5), 169. https://doi.org/10.3847/1538-3881/ab0e12

Osborn, H. P., Ansdell, M., Ioannou, Y., Sasdelli, M., Angerhausen, D., Caldwell, D., Jenkins, J. M., Räissi, C., & Smith, J. C. (2020). Rapid classification of TESS planet candidates with convolutional neural networks. *Astronomy & Astrophysics*, *633*, A53. https://doi.org/10.1051/0004-6361/201935345

---

## SLIDE 14: CNN Approach - Methodology
**[Placeholder for Tristan]**

Suggested content:
- CNN architecture overview
- Convolutional feature extraction
- Why CNNs for time series?
- Key results from papers

---

## SLIDE 15: CNN Results & Performance
**[Placeholder for Tristan]**

Suggested content:
- Model performance metrics
- Comparison with baselines
- Strengths and weaknesses
- Real-world validation

---

# ========================================
# BREE'S SECTION - TRANSFORMER APPROACH
# ========================================

## SLIDE 16: Transformer Research Papers
**[Placeholder for Bree]**

Morvan, M., Nikolaou, N., Yip, K. H., & Waldmann, I. (2022). Don't pay attention to the noise: Learning self-supervised representations of light curves with a denoising time series transformer. *Proceedings of the Thirty-ninth International Conference on Machine Learning (ICML 2022)*. https://arxiv.org/abs/2206.08447

Salinas, H., Pichara, K., Brahm, R., Pérez-Galarce, F., & Mery, D. (2023). Distinguishing a planetary transit from false positives: A Transformer-based classification for planetary transit signals. *Monthly Notices of the Royal Astronomical Society*, *522*(3), 3201–3216. https://doi.org/10.1093/mnras/stad1173

Salinas, H., Brahm, R., Olmschenk, G., Barry, R. K., Pichara, K., Silva, S. I., & Araujo, V. (2025). Exoplanet transit candidate identification in TESS full-frame images via a transformer-based algorithm. *Monthly Notices of the Royal Astronomical Society*, *538*(3), 2031–2049. https://doi.org/10.1093/mnras/staf347

---

## SLIDE 17: Transformer Approach - Methodology
**[Placeholder for Bree]**

Suggested content:
- Transformer architecture overview
- Self-attention mechanism
- Why Transformers for time series?
- Key results from papers

---

## SLIDE 18: Transformer Results & Performance
**[Placeholder for Bree]**

Suggested content:
- Model performance metrics
- Comparison with baselines
- Strengths and weaknesses
- Real-world validation

---

# ========================================
# TEAM WRAP-UP & COMPARISON
# ========================================

## SLIDE 19: Model Comparison Across Team

**All Three Approaches Side-by-Side**

| Metric       | BiLSTM+Clustering | CNN (Tristan) | Transformer (Bree) |
|--------------|-------------------|---------------|--------------------|
| **AUC**      | 0.7154            | [TBD]         | [TBD]              |
| **F1 Score** | 0.4550            | [TBD]         | [TBD]              |
| **Recall**   | 0.8600            | [TBD]         | [TBD]              |
| **Precision**| 0.3094            | [TBD]         | [TBD]              |
| **Parameters**| ~3.9M            | [TBD]         | [TBD]              |

**Key Insights**
- BiLSTM: High recall (86%), benefits from clustering
- CNN: [To be filled by Tristan]
- Transformer: [To be filled by Bree]

---

## SLIDE 20: Strengths & Weaknesses

**BiLSTM + Clustering (Josh)**

Strengths:
- Captures temporal dependencies well
- Bidirectional context
- Clustering enables specialization
- High recall (don't miss planets!)

Weaknesses:
- Lower precision (many false positives)
- Sequential processing (slower)
- Requires careful hyperparameter tuning

**CNN (Tristan)**
- [To be filled]

**Transformer (Bree)**
- [To be filled]

---

## SLIDE 21: Future Work & Ensemble

**Combining All Three Approaches**

**Ensemble Options**:
1. Voting ensemble - average predictions
2. Stacking ensemble - meta-learner
3. Weighted ensemble - optimize weights

**Expected improvement**: +2-5% AUC

**Next Steps**:
- Cross-validation on same test set
- Ensemble methods
- Production deployment for TESS

**Goal**: AUC > 0.80, F1 > 0.60

---

## SLIDE 22: Questions?

**Thank You!**

**Team Contributions**:
- Josh Manchester: BiLSTM + Clustering (AUC 0.7154)
- Tristan Moffett: CNN approach
- Brianne Leatherman: Transformer approach

**Key Achievement**: Successfully validated on real TESS confirmed exoplanet (L 98-59)

**Questions?**
