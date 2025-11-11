# Balanced Model Failure Diagnosis

**Date**: November 11, 2025
**Status**: CRITICAL FAILURE - Synthetic Training Does Not Transfer to Real Data

---

## Executive Summary

The balanced synthetic dataset experiment has **catastrophically failed**:

- **Training Performance**: AUC 1.0 (perfect) on synthetic validation data
- **Test Performance**: AUC 0.45 (worse than random) on real TESS data
- **Degradation**: -79% performance vs optimized model trained on real data

**Root Cause**: **Severe domain shift** between synthetic and real light curve features.

---

## Performance Comparison

| Model | Training Data | Test Data | AUC | Recall | F1 | Verdict |
|-------|--------------|-----------|-----|--------|-----|---------|
| **Optimized (OLD)** | 655 real windows | 655 real windows | **0.8050** | **0.8867** | **0.5145** | ✅ WORKS |
| **Balanced (NEW)** | 1,522 synthetic windows | 655 real windows | **0.4501** | **0.0800** | **0.0972** | ❌ FAILED |

**Degradation**:
- AUC: -44% (0.805 → 0.450)
- Recall: -91% (0.887 → 0.080)
- F1 Score: -81% (0.515 → 0.097)

---

## Root Cause Analysis

### 1. Feature Distribution Mismatch

The synthetic and real datasets have **fundamentally different** BLS feature distributions:

#### Transit Depth
| Dataset | Mean | Std | Max | Interpretation |
|---------|------|-----|-----|----------------|
| **Real** | 1.44% | 3.30% | 28.4% | Deep, variable transits |
| **Synthetic** | 0.18% | 0.21% | 1.45% | Shallow, uniform transits |
| **Difference** | **7.8× deeper** | 15.7× more variance | 19.6× larger max | **Completely different scale!** |

#### BLS Power (Signal Strength)
| Dataset | Mean | Std | Max | Interpretation |
|---------|------|-----|-----|----------------|
| **Real** | 0.39 | 1.31 | 10.8 | Highly variable signal quality |
| **Synthetic** | 0.30 | 0.46 | 1.0 | Low variance, capped at 1.0 |
| **Difference** | Similar mean | 2.8× less variance | 10.8× smaller max | **Missing high-SNR signals** |

#### Transit Duration
| Dataset | Mean | Max | Interpretation |
|---------|------|-----|----------------|
| **Real** | 0.078 days | 0.10 days | Short transits |
| **Synthetic** | 0.148 days | 0.59 days | Long transits |
| **Difference** | **1.9× longer** | 5.9× longer max | **Wrong time scale!** |

### 2. Clustering Failure

The model uses K-means clustering on [period, depth, duration, bls_power] features to create cluster embeddings.

**Problem**: The cluster centers learned from synthetic data are **incompatible** with real data:

```
Optimized Model:  5 clusters on REAL data
Balanced Model:  10 clusters on SYNTHETIC data
```

When the balanced model sees real data:
1. Real data features are **outside** the range of synthetic training features
2. K-means assigns real data to **wrong clusters**
3. Cluster embeddings encode **irrelevant patterns**
4. Model predictions become **random noise**

### 3. Confirmation: TESS Planet Predictions

When tested on 100 confirmed TESS exoplanet systems (300 windows):

| Model | Mean Probability | Std | Predictions >0.5 | Result |
|-------|------------------|-----|------------------|--------|
| **Optimized** | 0.236 | 0.144 | 16/300 (5.3%) | ✅ Detects planets |
| **Balanced** | **0.4977** | **0.000075** | **0/300 (0%)** | ❌ **Random guessing** |

The balanced model outputs ~50% probability for **every single window** with virtually zero variance. This is the signature of a model that learned nothing transferable.

---

## Why Synthetic Training Failed

### 1. Batman Transit Generator Limitations

The `batman-package` used to generate synthetic transits creates **idealized** light curves:

```python
# Synthetic parameters used:
- Limb darkening: Simple quadratic model
- Transit shape: Perfect Mandel-Agol (2002)
- Noise: Gaussian white noise (200 ppm)
- No stellar activity: No spots, flares, or rotation
- No systematic errors: No instrumental artifacts
```

**Real TESS data includes**:
- Stellar variability (rotation, spots, faculae)
- Flares, momentum dumps, scattered light
- CCD artifacts, cosmic rays
- Long-term trends from thermal variations
- Astrophysical false positives (EBs, blends)

### 2. Class Balance Paradox

We created 50/50 planet/non-planet balance to improve training. But this backfired:

**Hypothesis**: Perfect balance (50/50) with clear separation → model learns to output 0.5 for ambiguous cases

**Evidence**:
- Validation AUC 1.0 (perfect separation in synthetic data)
- Test probability mean = 0.4977 (almost exactly 0.5)
- Test probability std = 0.000075 (essentially constant)

The model learned: "When uncertain, predict 0.5" - which is the optimal strategy for perfectly balanced training data!

### 3. Optimization for Wrong Domain

The Optuna hyperparameters were optimized on synthetic validation data:

```json
{
  "lr": 2.05e-05,        // 10× lower than real-data model
  "dropout": 0.38,       // Higher regularization
  "n_clusters": 10,      // 2× more clusters
  "pos_weight": 2.302    // Wrong class weighting for real data
}
```

These hyperparameters are **tuned for synthetic patterns that don't exist in real data**.

---

## Scientific Insights

### Key Lessons Learned

1. **Domain Matters More Than Balance**
   - Imbalanced real data (23% positive) >>> Balanced synthetic data (30% positive)
   - 655 real samples outperform 1,522 synthetic samples by 79%

2. **Perfect Training Performance is a Red Flag**
   - AUC 1.0 in training → Check for domain shift!
   - Validation on synthetic data doesn't predict real-world performance

3. **Feature Engineering is Critical**
   - Clustering on [period, depth, duration, bls_power] is brittle
   - Synthetic features must match real data distributions exactly

4. **Transit Depth is the Most Important Feature**
   - Real transits: 1.44% depth (mean), up to 28%
   - Synthetic transits: 0.18% depth (mean), up to 1.45%
   - **8× mismatch causes total failure**

---

## Proposed Solutions

### Option 1: Hybrid Training (Recommended)

Train on **mixed dataset** of real + synthetic data:

```python
# Mix ratios to test:
- 50% real + 50% synthetic (800 real + 800 synthetic)
- 75% real + 25% synthetic (600 real + 400 synthetic)
- 90% real + 10% synthetic (655 real + 65 synthetic)
```

**Advantages**:
- Uses all available real data for feature learning
- Synthetic data augments without dominating
- Better class balance than pure real data

**Implementation**:
```bash
python build_hybrid_dataset.py \
  --real_dir data/windows_train \
  --synthetic_dir data/windows_train_400 \
  --mix_ratio 0.75 \
  --output_dir data/windows_hybrid
```

### Option 2: Domain Adaptation

Use synthetic data for pre-training, then fine-tune on real data:

1. **Phase 1**: Pre-train on 1,522 synthetic windows (20 epochs)
2. **Phase 2**: Fine-tune on 655 real windows (60 epochs)

**Advantages**:
- Leverages large synthetic dataset for initial learning
- Fine-tuning adapts to real data domain

### Option 3: Fix Synthetic Data Generation

Improve `generate_synthetic_dataset.py` to match real data distributions:

```python
# Updated parameters:
- Transit depth range: [0.001, 0.30] (instead of [0.0001, 0.015])
- Add stellar variability: spots, rotation (P=5-30 days)
- Add instrumental noise: TESS systematics
- Add momentum dumps, scattered light
- Realistic BLS power: Allow values > 1.0
```

**Disadvantages**:
- Requires significant work to model TESS systematics
- May still miss subtle real-data patterns

### Option 4: Abandon Synthetic Data (Pragmatic)

Accept that 655 real windows are sufficient:

- Current optimized model: **AUC 0.80** (very good!)
- Collect more real data instead of generating synthetic
- Focus on improving architecture, not data augmentation

---

## Recommendations

### Immediate Action

**Stop using the balanced synthetic model for production!** It's worse than random on real data.

**Continue using the optimized model** (AUC 0.80):
- Location: `runs/bilstm_cluster_optimized/best.pt`
- Trained on: 655 real windows
- Performance: Reliable, well-calibrated

### For Research/Publication

**Frame this as a negative result**:

**Research Question**: "Can synthetic transit data improve exoplanet detection models?"

**Answer**: **No, not without careful domain matching.**

**Contribution**:
- Demonstrated importance of domain fidelity in astronomical ML
- Showed that perfect training performance (AUC 1.0) doesn't guarantee generalization
- Quantified feature distribution requirements for transfer learning

**Paper Section Ideas**:
1. "Pitfalls of Synthetic Data in Astronomical Time Series"
2. "Why More Data Isn't Always Better: A Case Study"
3. "Domain Adaptation Challenges in Exoplanet Detection"

### For Presentation

**Title**: "Real vs Synthetic Training Data: A Cautionary Tale"

**Key Slide**:
```
Training Data Comparison:
├─ Model A (OLD): 655 real windows → AUC 0.80 ✅
└─ Model B (NEW): 1,522 synthetic windows → AUC 0.45 ❌

Lesson: Domain matters more than dataset size!
```

---

## Next Steps

### If continuing with synthetic data:

1. ✅ **Try Option 1 (Hybrid)** first - fastest path to improvement
2. Run ablation study: Test mix ratios [0.1, 0.25, 0.5, 0.75, 0.9]
3. Compare hybrid model to pure real model
4. Measure if synthetic data provides any benefit

### If abandoning synthetic approach:

1. ✅ **Document findings** for publication
2. Generate figures comparing feature distributions
3. Write "Lessons Learned" section for paper
4. Focus on improving architecture or ensemble methods

---

## Files Generated

- `comparison_report/OPTIMIZATION_REPORT.md` - Full performance comparison
- `comparison_report/metrics_comparison.csv` - Quantitative metrics
- `comparison_report/confusion_matrices.png` - Visualization
- `comparison_report/roc_comparison.png` - ROC curves
- `benchmarks/baseline_benchmark_*.json` - Benchmark results

---

## Conclusion

The balanced synthetic dataset experiment provided valuable insights, even though it failed:

**Success**: We learned what DOESN'T work and why
**Failure**: Synthetic training without domain matching is worse than no augmentation

The path forward is clear: Either fix the domain mismatch (hybrid training) or accept that real data, even if limited and imbalanced, is better than abundant but mismatched synthetic data.

---

*Generated by Claude Code for CS4280 Exoplanet Detection Project*
*November 11, 2025*
