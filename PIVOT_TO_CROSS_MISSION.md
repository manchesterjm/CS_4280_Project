# Pivot from Synthetic Data to Cross-Mission Testing

**Date**: November 11, 2025
**Decision**: Abandon synthetic/hybrid approach, pursue cross-mission generalization

---

## What Happened Today

### Morning: Analyzed Synthetic Model Failure
- Balanced synthetic model trained overnight
- Training performance: AUC 1.0 (perfect!)
- Testing performance: AUC 0.45 (worse than random!)
- **Root cause**: Domain shift - synthetic depth 8× shallower than real

### Afternoon: Created Hybrid Approach
- Built hybrid datasets (75% real + 25% synthetic, 90% real + 10% synthetic)
- Hypothesis: Maybe small amounts of synthetic data help
- **Problem**: Still mixing bad data with good data
- **Risk**: Even 10% synthetic might hurt performance

### Evening: USER INSIGHT - Cross-Mission Testing!
**User asked**: "Why not train on TESS and test on Kepler?"

**This is brilliant because**:
1. We've been training on TESS and testing on TESS (not true generalization!)
2. Both TESS and Kepler are REAL data (no synthetic artifacts)
3. Tests if model learned physics vs mission-specific patterns
4. Much more scientifically meaningful

---

## Why Cross-Mission is Superior

### Synthetic Approach (Failed)
✗ Synthetic ≠ Real (8× depth mismatch)
✗ Model learned wrong patterns
✗ AUC 1.0 → 0.45 (domain shift)
✗ Limited scientific value

### Hybrid Approach (Uncertain)
⚠️ Mixing bad data with good data
⚠️ Unknown if synthetic helps or hurts
⚠️ Still has domain shift issues
⚠️ Hours of training time

### Cross-Mission Approach (Best!)
✓ Both datasets are real
✓ Tests fundamental generalization
✓ Fast to execute (~45 minutes)
✓ Publishable either way
✓ Relevant to real deployment

---

## What We're Testing

### Current Status
- **Training**: 101 TESS stars, 655 windows
- **Testing**: 100 TESS stars, 300 windows
- **Performance**: AUC 0.805, 16/300 positive predictions

**This is same-mission testing** - not true generalization!

### New Experiment
- **Training**: Same 101 TESS stars, 655 windows
- **Testing**: 50 Kepler stars, ~150 windows
- **Question**: Does the model work on a different mission?

---

## Scientific Questions

### Research Question 1: Does it generalize?
**H0**: Model is mission-specific (overfitted to TESS)
- TESS performance: 16/300 positives
- Kepler performance: 0-5/150 positives
- Conclusion: Need domain adaptation

**H1**: Model is generalized (learned physics)
- TESS performance: 16/300 positives
- Kepler performance: 10-15/150 positives
- Conclusion: Ready for any mission

### Research Question 2: What breaks across missions?
If model fails on Kepler, what causes it?
- Cadence difference (2 min vs 30 min)?
- Wavelength difference (IR vs optical)?
- Noise characteristics?
- → Guides future improvements

---

## Timeline Update

### What We Abandoned
- ❌ Train hybrid 75% model (~30 min)
- ❌ Train hybrid 90% model (~25 min)
- ❌ Benchmark hybrid models (~10 min)
- ❌ Test hybrid on TESS (~5 min)
- **Total saved**: ~70 minutes of uncertain value

### What We're Doing Instead
- ⏳ Download Kepler data (~15 min) - **IN PROGRESS**
- ⏳ Process Kepler data (~5 min)
- ⏳ Build test windows (~2 min)
- ⏳ Test on Kepler (~3 min)
- ⏳ Analyze results (~10 min)
- **Total time**: ~35 minutes of high value

---

## Publication Value

### Old Approach (Synthetic/Hybrid)
**If hybrid works**: "Synthetic data augmentation helps a little"
- Minor contribution
- Limited generalizability
- Still concerns about domain shift

**If hybrid fails**: "Synthetic data doesn't work even at 10%"
- Negative result
- Confirms what we already know

### New Approach (Cross-Mission)
**If it works**: "First BiLSTM model generalizing across space missions"
- Major contribution
- Novel result
- Ready for PLATO, ARIEL, Roman Space Telescope

**If it fails**: "Domain shift challenges in cross-mission astronomical ML"
- Still valuable contribution
- Quantifies performance degradation
- Proposes domain adaptation solutions

**Either way, it's more publishable!**

---

## What Makes This Better

### 1. Addresses Fundamental Question
Old: "Can we fake more data?"
New: "Did we learn physics or mission artifacts?"

### 2. Uses Real Data Only
Old: Mixing synthetic (wrong domain) with real
New: Both TESS and Kepler are real observations

### 3. Practical Relevance
Old: Nobody deploys on synthetic data
New: Future missions need cross-mission models

### 4. Novel Contribution
Old: Many papers try data augmentation
New: Few papers test cross-mission generalization

### 5. Fast Execution
Old: Hours of training uncertain models
New: 45 minutes to definitive answer

---

## Current Status

### Running Now
```bash
python download_kepler_planets.py --n_targets 50
```
- Started: 16:00 UTC
- Expected completion: ~16:15 UTC
- Downloading 50 confirmed Kepler planet systems
- Output: `kepler_test_data/raw/*.csv`

### Next Steps (After Download)
1. Process Kepler light curves (5 min)
2. Build test windows (2 min)
3. Run inference (3 min)
4. Generate comparison report (10 min)

### Deliverable
Complete cross-mission analysis comparing:
- TESS → TESS performance (control)
- TESS → Kepler performance (experimental)
- Quantified generalization capability

---

## Key Insight

**The user was right**: For a generalized model, we should test generalization across real missions, not try to generate fake data that matches one mission.

This is what we should have been doing from the start!

---

## Summary

**Morning Decision**: Synthetic failed (AUC 0.45) → Try hybrid
**User Insight**: Why not test on Kepler?
**Our Decision**: Pivot to cross-mission testing

**Why**:
- More scientifically rigorous
- Faster to execute
- Higher publication value
- Tests true generalization
- Uses only real data

**Status**: Kepler data downloading now, complete analysis in ~45 minutes

---

*This pivot represents a significant improvement in experimental design based on user feedback*

🤖 Generated with [Claude Code](https://claude.com/claude-code)
