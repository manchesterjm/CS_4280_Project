# Cross-Mission Generalization: TESS → Kepler

**Created**: November 11, 2025
**Purpose**: Test if our model generalizes across space missions

---

## Why This is Better Than Synthetic Data

### The Problem with Synthetic Data
- Synthetic data had 8× depth mismatch with real data
- Generated "perfect" transits that don't exist in nature
- Model achieved AUC 1.0 on synthetic but AUC 0.45 on real data
- **Conclusion**: Synthetic ≠ Real, domain shift causes failure

### The Value of Cross-Mission Testing
**TESS vs Kepler are BOTH real astronomical data** but differ in:

| Characteristic | TESS | Kepler |
|----------------|------|--------|
| **Cadence** | 2 minutes | 30 minutes |
| **Wavelength** | 600-1000 nm (red/IR) | 430-890 nm (optical) |
| **Field of View** | 24° × 96° (whole sky) | 115 sq degrees (fixed) |
| **Mission Duration** | 2018-present | 2009-2018 |
| **Typical SNR** | Lower (all-sky survey) | Higher (stare at one field) |

**But the physics is the same**: A planet transiting a star causes the same fractional brightness drop regardless of which telescope observes it!

**If our model works on both**:
- ✅ Learned fundamental transit physics
- ✅ Truly generalized model
- ✅ Ready for deployment on future missions (PLATO, ARIEL, etc.)

**If our model fails on Kepler**:
- ⚠️ Overfitted to TESS-specific characteristics
- ⚠️ Needs domain adaptation techniques
- ⚠️ Still publishable as "challenges in cross-mission ML"

---

## Experimental Design

### Training Set (TESS)
- **Source**: Real TESS observations
- **Stars**: 101 unique systems
- **Windows**: 655 training windows
- **Positive rate**: 22.9% (150 planet windows)
- **Model**: Optimized BiLSTM+Clustering (AUC 0.805)

### Test Set 1 (TESS - Control)
- **Source**: Real TESS observations (different stars)
- **Stars**: 100 confirmed exoplanet systems
- **Windows**: 300 test windows
- **Performance**: 16/300 predicted as planets (5.3%)
- **Purpose**: Baseline same-mission performance

### Test Set 2 (Kepler - Experimental)
- **Source**: Real Kepler observations
- **Stars**: 50-100 confirmed exoplanet systems
- **Windows**: ~150-300 test windows
- **Expected**: ???
- **Purpose**: Test cross-mission generalization

---

## Hypothesis

**Null Hypothesis (H0)**: The model is mission-specific
- TESS performance: AUC 0.805, 16/300 predictions
- Kepler performance: AUC < 0.6, < 5/300 predictions
- Conclusion: Model overfitted to TESS characteristics

**Alternative Hypothesis (H1)**: The model is generalized
- TESS performance: AUC 0.805, 16/300 predictions
- Kepler performance: AUC > 0.7, 10-20/300 predictions
- Conclusion: Model learned fundamental transit physics

---

## Pipeline

### Step 1: Download Kepler Planet Light Curves

```bash
conda activate exo-lstm-gpu
cd C:\CS_4280_Project\Code

# Download 50 confirmed Kepler planets
python download_kepler_planets.py \
  --output_dir "C:\CS_4280_Project\kepler_test_data\raw" \
  --n_targets 50 \
  --save_list
```

**Time**: 10-20 minutes (network dependent)

**Output**:
- `kepler_test_data/raw/*.csv` - Raw light curves
- `kepler_test_data/kepler_planet_list.csv` - Planet metadata

### Step 2: Process Kepler Data

```bash
# Same processing as TESS
python process_tess_for_testing.py \
  --raw_dir "C:\CS_4280_Project\kepler_test_data\raw" \
  --output_dir "C:\CS_4280_Project\kepler_test_data\processed"
```

**Preprocessing**:
1. Remove NaNs and outliers (5-sigma clip)
2. Median normalization
3. Save as CSV with time, flux columns

### Step 3: Build Test Windows

```bash
python build_simple_windows.py \
  --data_dir "C:\CS_4280_Project\kepler_test_data\processed" \
  --output_dir "C:\CS_4280_Project\Code\data\windows_kepler_test" \
  --seq_len 2048 \
  --n_windows 3
```

**Output**: X.npy, y.npy, meta.csv (Kepler test windows)

### Step 4: Run Inference on Kepler Data

```bash
python inference_cluster_model.py \
  --model_path "C:\CS_4280_Project\Code\runs\bilstm_cluster_optimized\best.pt" \
  --windows_dir "C:\CS_4280_Project\Code\data\windows_kepler_test" \
  --output_file "C:\CS_4280_Project\Code\reports\kepler_predictions.csv"
```

**Time**: 2-3 minutes

**Output**: Predictions for Kepler planets

### Step 5: Compare TESS vs Kepler Performance

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load predictions
tess = pd.read_csv('reports/optimized_planet_predictions.csv')
kepler = pd.read_csv('reports/kepler_predictions.csv')

# Compare
print("TESS Results (Same Mission):")
print(f"  Mean probability: {tess['probability'].mean():.4f}")
print(f"  Predictions >0.5: {(tess['probability'] > 0.5).sum()} / 300")

print("\nKepler Results (Cross-Mission):")
print(f"  Mean probability: {kepler['probability'].mean():.4f}")
print(f"  Predictions >0.5: {(kepler['probability'] > 0.5).sum()} / {len(kepler)}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(tess['probability'], bins=20, alpha=0.7, label='TESS')
axes[0].axvline(0.5, color='red', linestyle='--', label='Threshold')
axes[0].set_xlabel('Prediction Probability')
axes[0].set_ylabel('Count')
axes[0].set_title('TESS Test Set')
axes[0].legend()

axes[1].hist(kepler['probability'], bins=20, alpha=0.7, label='Kepler', color='orange')
axes[1].axvline(0.5, color='red', linestyle='--', label='Threshold')
axes[1].set_xlabel('Prediction Probability')
axes[1].set_ylabel('Count')
axes[1].set_title('Kepler Test Set (Cross-Mission)')
axes[1].legend()

plt.tight_layout()
plt.savefig('reports/tess_vs_kepler_predictions.png', dpi=300)
plt.show()
```

---

## Expected Results Scenarios

### Scenario 1: Strong Generalization (Best Case)

| Metric | TESS | Kepler | Interpretation |
|--------|------|--------|----------------|
| Mean Prob | 0.236 | 0.20-0.25 | Similar confidence |
| Predictions >0.5 | 16/300 | 10-15/150 | Proportional detection |
| **Conclusion** | **Model learned physics, not mission artifacts** |

**For Paper**: "BiLSTM+Clustering generalizes across space missions"

### Scenario 2: Partial Generalization (Moderate)

| Metric | TESS | Kepler | Interpretation |
|--------|------|--------|----------------|
| Mean Prob | 0.236 | 0.10-0.15 | Lower but non-zero |
| Predictions >0.5 | 16/300 | 2-5/150 | Detects some planets |
| **Conclusion** | **Some transfer, but mission-specific biases exist** |

**For Paper**: "Domain adaptation needed for cross-mission deployment"

### Scenario 3: No Generalization (Worst Case)

| Metric | TESS | Kepler | Interpretation |
|--------|------|--------|----------------|
| Mean Prob | 0.236 | ~0.5 | Random guessing |
| Predictions >0.5 | 16/300 | 0/150 | No detections |
| **Conclusion** | **Model completely overfitted to TESS characteristics** |

**For Paper**: "Challenges in cross-mission astronomical ML"

---

## Why This Matters for Publication

### Novel Contribution
- **First study** testing BiLSTM exoplanet detection across missions
- Quantifies generalization vs overfitting in astronomical ML
- Provides benchmark for future cross-mission models

### Either Outcome is Valuable

**If it works**:
- "Generalizable Deep Learning for Multi-Mission Exoplanet Detection"
- Show that physics-based features (period, depth, duration) are mission-agnostic
- Demonstrate model is ready for PLATO, ARIEL, Roman Space Telescope

**If it fails**:
- "Domain Shift Challenges in Cross-Mission Astronomical Time Series"
- Quantify how much performance degrades across missions
- Propose domain adaptation solutions (fine-tuning, adversarial training)

### Comparison to Existing Work
Most exoplanet ML papers:
- Train and test on SAME mission (Kepler → Kepler or TESS → TESS)
- Don't test generalization
- Limited real-world applicability

**Our work**:
- Train on TESS, test on Kepler (and vice versa possible)
- Quantifies true generalization capability
- More rigorous experimental design

---

## Timeline

| Step | Time | Status |
|------|------|--------|
| 1. Download Kepler data | 10-20 min | ⏳ Pending |
| 2. Process Kepler data | 5 min | ⏳ Pending |
| 3. Build Kepler test windows | 2 min | ⏳ Pending |
| 4. Run inference | 3 min | ⏳ Pending |
| 5. Analyze results | 15 min | ⏳ Pending |

**Total**: ~35-45 minutes

---

## Advantages Over Synthetic/Hybrid Approach

| Approach | Training Data | Test Data | Insight Value |
|----------|---------------|-----------|---------------|
| **Synthetic** | 1522 synthetic | 655 real | ❌ Failed (AUC 0.45) |
| **Hybrid** | 727 mix | 655 real | ⏳ Unknown, risky |
| **Cross-Mission** | 655 real TESS | 150 real Kepler | ✅ **Best!** |

**Why Cross-Mission Wins**:
1. ✅ Both datasets are REAL (no domain shift artifacts)
2. ✅ Tests fundamental question: Did model learn physics or mission quirks?
3. ✅ Publishable either way (success or failure)
4. ✅ More relevant to real deployment (future missions)
5. ✅ Faster to execute than training hybrid models

---

## Files to Create

### Scripts
```
Code/
├── download_kepler_planets.py       [NEW] Download Kepler confirmed planets
├── process_kepler_data.py           [CAN REUSE] Same as TESS processing
└── compare_tess_kepler.py           [NEW] Cross-mission analysis
```

### Data
```
kepler_test_data/
├── raw/                             [NEW] Raw Kepler light curves
├── processed/                       [NEW] Processed Kepler light curves
├── kepler_planet_list.csv           [NEW] Planet metadata
└── windows_kepler_test/             [NEW] Test windows
    ├── X.npy, y.npy, meta.csv
```

### Reports
```
Code/reports/
├── kepler_predictions.csv           [NEW] Kepler inference results
├── tess_vs_kepler_predictions.png   [NEW] Comparison plot
└── CROSS_MISSION_REPORT.md          [NEW] Analysis and conclusions
```

---

## Summary

**Instead of** trying to make synthetic data work (failed) or hoping hybrid helps (uncertain):

**We should** test cross-mission generalization (TESS → Kepler):
- ✅ Both datasets are real
- ✅ Tests fundamental capability
- ✅ Fast to execute (~45 min)
- ✅ Publishable either way
- ✅ More scientifically meaningful

**This is the right experiment for a generalized model!**

---

*Created by Claude Code for CS4280 Exoplanet Detection Project*
*November 11, 2025*
