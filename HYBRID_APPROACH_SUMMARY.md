# Hybrid Approach: Real + Synthetic Data Mix

**Created**: November 11, 2025
**Status**: Hybrid datasets ready, training pending

---

## What Was Done

### 1. Root Cause Identified

The balanced synthetic model failed due to **domain shift** between synthetic and real data:
- Transit depth: 7.8× shallower in synthetic data
- BLS power: 10.8× weaker in synthetic data
- Result: Model learned patterns that don't exist in real TESS data

### 2. Hybrid Solution Created

Built a hybrid dataset approach that mixes real and synthetic data to get:
- **Domain fidelity** from real data
- **Better class balance** from synthetic data
- **More training examples** overall

### 3. Datasets Built

| Dataset | Real Windows | Synthetic Windows | Total | Positive Rate | Location |
|---------|--------------|-------------------|-------|---------------|----------|
| **Pure Real** | 655 | 0 | 655 | 22.90% | `data/windows_train` |
| **Hybrid 90** | 655 | 72 | **727** | 24.21% | `data/windows_hybrid_90` |
| **Hybrid 75** | 655 | 218 | **873** | 25.43% | `data/windows_hybrid_75` |
| **Pure Synthetic** | 0 | 1522 | 1522 | 30.29% | `data/windows_train_400` |

**Hypothesis**: Hybrid 90 (90% real, 10% synthetic) will perform best:
- Dominated by real data for domain fidelity
- Small amount of synthetic data improves balance
- Avoids synthetic patterns overwhelming the model

---

## How to Train and Evaluate

### Step 1: Train Hybrid Models

#### Hybrid 90% (Recommended)
```bash
conda activate exo-lstm-gpu
cd C:\CS_4280_Project\Code

python train_bilstm_cluster.py \
  --windows_dir data/windows_hybrid_90 \
  --n_clusters 5 \
  --epochs 80 \
  --batch_size 128 \
  --lr 0.000225 \
  --hidden 256 \
  --layers 4 \
  --dropout 0.311 \
  --save_dir runs/bilstm_cluster_hybrid_90 \
  --amp_dtype fp16 \
  --pos_weight 3.0 \
  --num_workers 0
```

**Time**: ~25 minutes (80 epochs × 19s/epoch)

#### Hybrid 75%
```bash
python train_bilstm_cluster.py \
  --windows_dir data/windows_hybrid_75 \
  --n_clusters 5 \
  --epochs 80 \
  --batch_size 128 \
  --lr 0.000225 \
  --hidden 256 \
  --layers 4 \
  --dropout 0.311 \
  --save_dir runs/bilstm_cluster_hybrid_75 \
  --amp_dtype fp16 \
  --pos_weight 3.0 \
  --num_workers 0
```

**Time**: ~28 minutes (80 epochs × 21s/epoch)

### Step 2: Benchmark Models

```bash
# Benchmark Hybrid 90
python benchmark_model.py \
  --model_path runs/bilstm_cluster_hybrid_90/best.pt \
  --output_dir benchmarks

# Benchmark Hybrid 75
python benchmark_model.py \
  --model_path runs/bilstm_cluster_hybrid_75/best.pt \
  --output_dir benchmarks
```

### Step 3: Test on Real TESS Planets

```bash
# Test Hybrid 90 on 100 confirmed planets
python inference_cluster_model.py \
  --model_path runs/bilstm_cluster_hybrid_90/best.pt \
  --windows_dir data/windows_planet_test \
  --output_file reports/hybrid_90_planet_predictions.csv

# Test Hybrid 75
python inference_cluster_model.py \
  --model_path runs/bilstm_cluster_hybrid_75/best.pt \
  --windows_dir data/windows_planet_test \
  --output_file reports/hybrid_75_planet_predictions.csv
```

### Step 4: Compare All Models

```python
import pandas as pd

# Load predictions
real = pd.read_csv('reports/optimized_planet_predictions.csv')
h90 = pd.read_csv('reports/hybrid_90_planet_predictions.csv')
h75 = pd.read_csv('reports/hybrid_75_planet_predictions.csv')
synth = pd.read_csv('reports/balanced_model_planet_predictions.csv')

# Compare
models = {
    'Pure Real (Optimized)': real,
    'Hybrid 90% Real': h90,
    'Hybrid 75% Real': h75,
    'Pure Synthetic': synth
}

for name, df in models.items():
    mean_prob = df['probability'].mean()
    std_prob = df['probability'].std()
    n_positive = (df['probability'] > 0.5).sum()
    print(f"{name:25s} | Mean: {mean_prob:.4f} | Std: {std_prob:.4f} | >0.5: {n_positive:3d}/300")
```

---

## Expected Results

### Baseline Performance (Already Measured)

| Model | AUC | Recall | F1 | TESS Positives |
|-------|-----|--------|-----|----------------|
| **Pure Real** | 0.805 | 0.887 | 0.515 | 16/300 ✅ |
| **Pure Synthetic** | 0.450 | 0.080 | 0.097 | 0/300 ❌ |

### Hybrid Predictions

| Model | Expected AUC | Expected TESS | Reasoning |
|-------|--------------|---------------|-----------|
| **Hybrid 90** | 0.79-0.82 | 15-20/300 | Best of both worlds |
| **Hybrid 75** | 0.75-0.80 | 10-18/300 | More synthetic noise |

**Success Criteria**:
- Hybrid 90 matches or beats Pure Real (AUC ≥ 0.805)
- Hybrid models outperform Pure Synthetic significantly
- Better class balance improves recall without hurting precision

---

## Files Created

### Scripts
- `Code/build_hybrid_dataset.py` - Combines real and synthetic data at any ratio

### Datasets
- `Code/data/windows_hybrid_90/` - 727 windows (90% real, 10% synthetic)
  - `X.npy`, `y.npy`, `meta.csv`, `config.json`
- `Code/data/windows_hybrid_75/` - 873 windows (75% real, 25% synthetic)
  - `X.npy`, `y.npy`, `meta.csv`, `config.json`

### Documentation
- `BALANCED_MODEL_FAILURE_DIAGNOSIS.md` - Root cause analysis of synthetic failure
- `HYBRID_APPROACH_SUMMARY.md` - This file
- `Code/comparison_report/` - Visualizations and metrics

---

## Scientific Questions to Answer

1. **Does mixing help?**
   - Can synthetic data improve model performance when mixed with real data?
   - Or does it hurt even at small amounts?

2. **What's the optimal mix ratio?**
   - Is 90% real / 10% synthetic best?
   - Or should we use 100% real data?

3. **Does balance matter more than domain?**
   - Pure real: 22.9% positive, AUC 0.805
   - Hybrid 90: 24.2% positive, AUC ???
   - Is the 1.3% better balance worth potential synthetic noise?

---

## Alternative Approaches (If Hybrid Fails)

### Option 1: Data Augmentation on Real Data
Instead of synthetic generation, augment real data:
- Time warping (stretch/compress light curves)
- Noise injection (add realistic TESS noise)
- Transit parameter perturbation (shift t0, vary depth slightly)

**Advantages**:
- Stays in real data domain
- Increases dataset size
- Preserves real TESS characteristics

### Option 2: Fix Synthetic Generation
Update `generate_synthetic_dataset.py` to match real distributions:
```python
# Current (wrong):
planet_radius_earth = np.random.uniform(1.0, 11.0)  # Too small

# Fixed (correct):
planet_radius_earth = np.random.uniform(1.0, 50.0)  # Include super-Jupiters
# OR directly sample depth:
depth = np.random.lognormal(mean=-4.5, sigma=1.5)  # Matches real distribution
```

### Option 3: Transfer Learning
1. Pre-train on large synthetic dataset (1522 windows)
2. Fine-tune on small real dataset (655 windows)

**Advantages**:
- Leverages large synthetic data for initial feature learning
- Fine-tuning adapts to real data domain

### Option 4: Accept Pure Real is Best
If hybrid doesn't improve over pure real (AUC 0.805):
- Document as negative result
- Focus on architecture improvements instead
- Collect more real data rather than generate synthetic

---

## Next Steps

### Immediate (Run Training)

1. Train Hybrid 90 model (25 minutes)
2. Train Hybrid 75 model (28 minutes)
3. Benchmark both models
4. Test on 100 TESS planets
5. Compare results

### If Hybrid Works

- **Document success**: Hybrid approach improves performance
- **Publish finding**: Mix ratios and their effects
- **Apply to paper**: Include hybrid results in methodology

### If Hybrid Fails

- **Document failure**: Even small amounts of synthetic data hurt
- **Conclude**: Domain fidelity > dataset size > class balance
- **Recommend**: Focus on real data collection or augmentation

---

## Key Insights So Far

1. **Domain matters most**: 655 real windows >> 1,522 synthetic windows
2. **Perfect training != good testing**: AUC 1.0 in training → AUC 0.45 in testing
3. **Feature distribution is critical**: 8× depth mismatch causes total failure
4. **Class balance is secondary**: 22.9% vs 30.3% positive less important than domain

**Bottom Line**: We're about to test if "a little bit of wrong data" helps or hurts when mixed with "not enough right data".

---

*Created by Claude Code for CS4280 Exoplanet Detection Project*
*November 11, 2025*
