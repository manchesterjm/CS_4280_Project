# RNN Section - Midterm Presentation Slides
## Josh Manchester
### Matching Group Format

---

## SLIDE 1: RNN Research Papers (NEW)

**Title:** Related Work: RNN Research Papers

Speiser, A., Müller, L. R., Matti, U., Obholzer, N. D., Legant, W. R., Kreshuk, A., ... & Hufnagel, L. (2020). Machine learning for cluster analysis of localization microscopy data. Nature Communications, 11(1), 1493. **(H5 index: 399)**

Vu, M. T., Vo, N. D., Ngo, T. D., Pham, T. D., Huynh, T. T. M., Nguyen, N. T., ... & Ly, H. B. (2024). Harnessing LSTM and XGBoost algorithms for storm prediction. Scientific Reports, 14(1), 11516. **(H5 index: 234)**

Ding, Y., Ji, K., Xiao, M., Zheng, X., Chen, Y., Liang, J., ... & Qi, Y. (2024). Photometric redshift estimation for CSST survey with LSTM neural networks. Monthly Notices of the Royal Astronomical Society, 535(2), 1844-1858. **(H5 index: 151)**

---

## SLIDE 2: Machine Learning for Cluster Analysis

**Title:** Related Work: Machine learning for cluster analysis of localization microscopy data

**[LEFT SIDE - Light Gray Box]**
**Key Innovation**
Supervised clustering combined with machine learning for analyzing millions of data points

**[RIGHT SIDE - Dark Blue Box]**
**Key Takeaway**
K-means clustering before ML training improves both accuracy and computational speed on large-scale datasets

**Architecture**
• K-means clustering on extracted features
• Supervised learning on cluster assignments
• Scalable to millions of data points

**Results**
• Faster training convergence
• Improved classification accuracy
• Handles large-scale datasets efficiently

Speiser, A., Müller, L. R., Matti, U., Obholzer, N. D., Legant, W. R., Kreshuk, A., ... & Hufnagel, L. (2020). Machine learning for cluster analysis of localization microscopy data. Nature Communications, 11(1), 1493.

---

## SLIDE 3: LSTM for Time Series Patterns

**Title:** Related Work: Harnessing LSTM and XGBoost algorithms for storm prediction

**[LEFT SIDE - Light Gray Box]**
**Key Innovation**
LSTM networks capture long-term dependencies in noisy environmental time series data

**[RIGHT SIDE - Dark Blue Box]**
**Key Takeaway**
LSTM memory cells handle temporal dependencies that traditional methods miss in sequential data with high noise

**Architecture**
• Stacked LSTM layers for temporal pattern recognition
• Handles irregular sampling and data gaps
• Dropout regularization for noisy data

**Results**
• Superior performance on noisy time series
• Captures long-term dependencies effectively
• Generalizes to unseen weather patterns

Vu, M. T., Vo, N. D., Ngo, T. D., Pham, T. D., Huynh, T. T. M., Nguyen, N. T., ... & Ly, H. B. (2024). Harnessing LSTM and XGBoost algorithms for storm prediction. Scientific Reports, 14(1), 11516.

---

## SLIDE 4: LSTM for Astronomical Photometry

**Title:** Related Work: Photometric redshift estimation for CSST survey with LSTM neural networks

**[LEFT SIDE - Light Gray Box]**
**Key Innovation**
First application of LSTM to large-scale astronomical photometric flux measurements

**[RIGHT SIDE - Dark Blue Box]**
**Key Takeaway**
LSTM reduced outliers by 33% compared to traditional neural networks on real astronomical survey data

**Architecture**
• LSTM for photometric time series
• Handles large survey datasets
• Optimized for astronomical flux measurements

**Results**
• 33% fewer outliers than MLPs
• Better handling of temporal photometric data
• Validated on real survey observations

Ding, Y., Ji, K., Xiao, M., Zheng, X., Chen, Y., Liang, J., ... & Qi, Y. (2024). Photometric redshift estimation for CSST survey with LSTM neural networks. Monthly Notices of the Royal Astronomical Society, 535(2), 1844-1858.

---

## SLIDE 5: Progress Since Proposal

**Title:** Progress Since Proposal

• Complete data pipeline (download → process → train → test)
• BiLSTM + K-means clustering architecture implemented
• Overcame class imbalance challenges
• Optuna hyperparameter optimization
• Validated on 100 confirmed exoplanets

---

## SLIDE 6: Methodology: BiLSTM + Clustering

**Title:** Methodology: BiLSTM + Clustering

**Data**
• 655 windows from TESS light curves (70/15/15 train/val/test split)

**Preprocessing**
• Box Least Squares (BLS) period detection
• 2048-point window extraction
• Phase folding on detected period
• Z-score normalization

**Feature Extraction**
• Period (days)
• Transit depth (%)
• Transit duration (days)
• BLS power (signal strength)

**[INCLUDE PREPROCESSING DIAGRAM similar to CNN slides showing:]**
(1) Raw Light Curve → (2) Cleaned + Detrended → (3) Phase-Folded + Normalized → (4) BLS Features Extracted → (5) 2048-point Window

---

## SLIDE 7: Methodology: BiLSTM + Clustering (Cont.)

**Title:** Methodology: BiLSTM + Clustering (Cont.)

**Model Architecture**
• K-means clustering (5 clusters) on BLS features
• Cluster embeddings (32-dim)
• 3-layer BiLSTM (256 hidden units, bidirectional)
• Concatenate [LSTM output + cluster embedding]
• Fully connected layers: 512 → 256 → 128 → 1
• Sigmoid output (binary classification)

**Training**
• Batch size: 64
• Adam optimizer (lr=1e-4)
• Class-weighted loss (pos_weight=3.367 for 23% positive rate)
• Mixed precision (FP16) for GPU acceleration
• Early stopping (patience=15 epochs)
• Gradient clipping (max_norm=1.0)

**[INCLUDE ARCHITECTURE DIAGRAM showing:]**
Input Window (2048) → BiLSTM Layers → Concat with Cluster Embedding → FC Layers → Output

---

## SLIDE 8: Experimental Results

**Title:** Experimental Results

**Dataset & Training** | **Test Set Performance**

**Training Data** | **[BAR CHART showing:]**
655 windows total | AUC: 75.72
• Training: 459 (70%) | F1 Score: 51.45
• Validation: 98 (15%) | Recall: 88.67
• Test: 98 (15%) | Precision: 38.27
                    | Accuracy: 68.37

**Class Distribution** | 0  20  40  60  80  100
• Planets: 22.9% (150) |     Score (%)
• Non-planets: 77.1% (505) |

**Key Result** | **[CONFUSION MATRIX:]**
Optimized model with Optuna | Shows TN, FP, FN, TP
AUC improved 0.69 → 0.76 (+9%) | (similar to CNN/Transformer slides)

---

## SLIDE 9: Initial Training Failure

**Title:** Learning from Failure: Class Imbalance

**Problem: Trained on 100 Planets Only**
• Dataset: 100 confirmed planet light curves
• No non-planet examples

**Result: Catastrophic Overfitting**
• Model predicted everything as planet
• Recall: 100% (found all "planets")
• Precision: ~20% (80% false positives)

**Lesson Learned**
"The model learned: all light curves are planets"

**Solution**
Add 300 non-planet examples (flares, noise, eclipsing binaries)

---

## SLIDE 10: Balanced Training Success

**Title:** Solving Class Imbalance

**Improved Dataset**
• 100 confirmed planets
• 300 non-planets (flares, noise, eclipsing binaries)
• 655 total windows after augmentation

**Baseline Results**
• AUC: 0.6947
• F1 Score: 0.34
• Successfully distinguished planets from false positives

**Real-World Validation**
• Tested on 7 TESS targets
• Correctly identified TIC 307210830 (L 98-59 system)
• Multi-planet system ranked #1

---

## SLIDE 11: Optuna Optimization Results

**Title:** Hyperparameter Optimization with Optuna

**Method**
• 30 trials with TPE sampler
• Search space: layers, batch size, LR, dropout, clusters
• Early stopping per trial

**Optimized Hyperparameters**
| Parameter | Baseline | Optimized | Change |
|-----------|----------|-----------|--------|
| Layers | 3 | 4 | +33% |
| Batch size | 64 | 128 | +100% |
| Learning rate | 1e-4 | 2.25e-4 | +2.25× |
| Dropout | 0.4 | 0.311 | Optimized |
| Clusters | 5 | 5 | Same |

**Results**
• AUC: 0.6947 → **0.7572** (+9.0% improvement)
• Tested on 100 confirmed exoplanet systems
• 16/300 windows predicted as planets (vs 0/300 baseline)

---

## SLIDE 12: Model Comparison

**Title:** Progressive Improvement Summary

| Approach | Dataset | AUC | TESS Predictions | Status |
|----------|---------|-----|------------------|---------|
| **Planets Only** | 100 planets | N/A | 100% false pos | Failed |
| **Baseline** | 655 windows (23% pos) | 0.6947 | 0/300 | Working |
| **Optimized** | 655 windows (23% pos) | **0.7572** | **16/300** | **Best** |

**Key Insight**
Class balance and hyperparameter tuning critical for imbalanced astronomical data

---

## SLIDE 13: Demo

**Title:** Demo

[INSERT: demo_video.mp4 - Located at C:\CS_4280_Project\term_project_files\demo_video.mp4]

**20-second demonstration showing:**
• Model running on real TESS light curves
• TIC 307210830 (confirmed exoplanet) ranked highest
• Inference pipeline in action

---

## SLIDE 14: What's Next?

**Title:** What's Next?

**Short Term** | **Medium Term**
• Cross-mission testing (TESS → Kepler) | • Attention mechanisms
• Expand dataset beyond Sector 1 | • Ensemble with CNN/Transformer
• K-fold cross-validation | • Test on mixed datasets

**Long Term** | **Final Deliverables**
• Collect more real TESS data | • Research Paper
• Transfer learning approaches | • BiLSTM Model
• Domain adaptation for Kepler | • Code

---

## GRAPHS/VISUALIZATIONS NEEDED

### For Slide 8 (Experimental Results):
1. **Bar Chart** showing:
   - AUC: 75.72
   - F1 Score: 51.45
   - Recall: 88.67
   - Precision: 38.27
   - Accuracy: 68.37

2. **Confusion Matrix** heatmap showing:
   - True Negatives, False Positives
   - False Negatives, True Positives
   (Similar style to CNN/Transformer slides)

### For Slide 6 (Methodology):
3. **Preprocessing Pipeline Diagram** showing:
   - Raw light curve → Cleaned → Phase-folded → BLS features → Window extraction
   (Similar to the dual-view diagrams in CNN slides)

### For Slide 7 (Architecture):
4. **BiLSTM Architecture Diagram** showing:
   - Input (2048-point window)
   - BiLSTM layers (3 layers, 256 hidden)
   - Cluster embedding concatenation
   - Fully connected layers
   - Sigmoid output
   (Similar style to CNN/Transformer architecture diagrams)

---

## DATA FILES TO USE FOR GRAPHS:

**Location:** `C:\CS_4280_Project\Code\runs\bilstm_cluster_optimized\`
- Contains training metrics and confusion matrix data

**Location:** `C:\CS_4280_Project\Code\reports\`
- `optimized_planet_predictions.csv` - Predictions on 100 planets

**Metrics (from your documentation):**
- AUC: 0.7572
- F1: 0.5145
- Recall: 0.8867
- Precision: 0.3827
- Accuracy: 0.6837

