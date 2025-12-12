# Final Experiment: Kepler Fine-Tuning for Cross-Mission Generalization

**Date:** December 11, 2025
**Status:** COMPLETED - Negative Result

## Objective

Following professor's recommendation, test whether fine-tuning the TESS-trained BiLSTM model on a mixed TESS+Kepler dataset (95% TESS, 5% Kepler) would improve cross-mission generalization.

## Methodology

### Data Preparation
1. Downloaded 398 additional Kepler light curves from MAST using `lightkurve`
   - Targets: Kepler-1 through Kepler-396
   - Failed downloads: Kepler-72, Kepler-73 (could not resolve sky position)
2. Combined with 93 existing Kepler light curves
3. Built 1,473 training windows from 491 total Kepler light curves (3 windows per light curve)

### Fine-Tuning Configuration
- **Base model:** sector1_final_0918 (AUC 0.9261 on TESS test set)
- **Training mix:** 26,472 TESS windows + 1,393 Kepler windows (5.0% Kepler)
- **Learning rate:** 1e-5
- **Epochs:** 20 (best model saved at epoch 2)
- **Batch size:** 128
- **pos_weight:** 5.14 (adjusted for new class balance)

### Training Observations
- Best validation AUC: 0.7703 (epoch 2)
- Training became unstable after epoch 13 (NaN losses)
- Model saved from epoch 2 before instability

## Results

### Kepler Planet Detection Rate (1,473 confirmed planet windows)

| Model | Kepler Mix | Detected | Rate | Mean Prob |
|-------|------------|----------|------|-----------|
| Original TESS | 0% | 55/1473 | **3.7%** | 0.1556 |
| Fine-tuned | 0.5% | 0/1473 | 0.0% | 0.0689 |
| Fine-tuned | 1.0% | 0/1473 | 0.0% | 0.0916 |
| Fine-tuned | 5.0% | 0/1473 | 0.0% | 0.0706 |

## Key Findings

1. **Fine-tuning made performance WORSE**, not better
   - Original model: 3.7% detection rate
   - All fine-tuned variants: 0% detection rate

2. **Domain shift is fundamental**
   - TESS: 2-minute cadence, 27-day sectors
   - Kepler: 30-minute cadence, 4-year continuous observation
   - These differences cannot be bridged by simple fine-tuning

3. **Model overfits to training distribution**
   - Adding Kepler data to training caused the model to become more conservative overall
   - Reduced mean probability across all test samples

## Conclusion

**Simple fine-tuning does not solve the cross-mission generalization problem.**

The BiLSTM+Clustering architecture learns mission-specific features (cadence patterns, noise characteristics) rather than generalizable transit detection features. This is a fundamental limitation of the approach that would require architectural changes to address, such as:
- Cadence-agnostic preprocessing (e.g., interpolation to common grid)
- Domain adaptation techniques (e.g., domain-adversarial training)
- Multi-task learning with mission identification

## Files Created/Modified

- `download_kepler_lightcurves.py` - Script to download Kepler data from MAST
- `build_kepler_windows.py` - Updated to handle multiple input directories
- `evaluate_kepler.py` - Updated with batched inference for large datasets
- `finetune_for_kepler.py` - Fine-tuning script with configurable Kepler fraction

## Cleanup

The following directories were deleted after experiment completion to free disk space:
- `D:\kepler_downloads` (~15 GB of Kepler light curves)
- `D:\CS_4280_Project\Code\data\windows_kepler_5pct`
- `D:\CS_4280_Project\Code\runs\finetuned_kepler_5pct`

## Impact on Paper

This experiment provides concrete evidence for the "Cross-Mission Generalization" limitation discussed in the RNN Findings section. The negative result is scientifically valuable as it:
1. Demonstrates the severity of domain shift between missions
2. Shows that naive fine-tuning is insufficient
3. Identifies future research directions for improving generalization
