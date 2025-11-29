# Class Imbalance Solutions: SMOTE, Down-sampling, Up-sampling

**Professor's Recommendation**: "SMOTE, down sample then up sample"

**Your Current Problem**: 150 planets (23%) vs 505 non-planets (77%)

---

## The Three Techniques Explained

### 1. SMOTE (Synthetic Minority Over-sampling Technique)

**What it does**: Creates **realistic** synthetic minority examples by interpolating between existing real examples.

**How it works**:
```
Real planet A: [period=5.0, depth=0.01, duration=3.0, bls_power=40]
Real planet B: [period=6.0, depth=0.012, duration=3.5, bls_power=45]

SMOTE creates planet C by interpolating:
Synthetic C: [period=5.5, depth=0.011, duration=3.25, bls_power=42.5]
```

**Why it's better than your batman synthetic**:
- ✅ Stays in **real data distribution** (no domain shift!)
- ✅ Guaranteed to be similar to real planets
- ✅ Works on feature space (BLS features)
- ❌ Limitation: Hard to interpolate raw time series (2048 points)

### 2. Down-sampling

**What it does**: Randomly remove examples from the majority class.

**Example**:
```
Before: 150 planets + 505 non-planets = 655 total
After:  150 planets + 150 non-planets = 300 total (removed 355 non-planets)
```

**Pros**:
- ✅ Perfect 50/50 balance
- ✅ Faster training (smaller dataset)
- ✅ No synthetic data issues

**Cons**:
- ❌ You **lose data** (throw away 355 examples!)
- ❌ Model sees fewer examples of majority class patterns

### 3. Up-sampling

**What it does**: Randomly duplicate examples from the minority class.

**Example**:
```
Before: 150 planets + 505 non-planets = 655 total
After:  505 planets (copies) + 505 non-planets = 1010 total
```

**Pros**:
- ✅ Perfect 50/50 balance
- ✅ No data loss
- ✅ More training examples

**Cons**:
- ❌ Model sees **same planets multiple times** (overfitting risk)
- ❌ No new information added

---

## Recommended Approaches (In Order)

### 🥇 BEST: Down-sample + Up-sample (Combined)

**Strategy**:
1. Down-sample non-planets: 505 → 300 (lose 205 examples, but keep most)
2. Up-sample planets: 150 → 300 (duplicate each ~2×)
3. Result: 300 planets + 300 non-planets = 600 total

**Why this is best**:
- ✅ Balanced (50/50)
- ✅ Don't lose too much data (only 205 vs 355)
- ✅ Don't over-duplicate (only 2× vs 3.4×)
- ✅ Middle ground approach
- ✅ **Works perfectly with time series data** (no interpolation issues)

### 🥈 ALTERNATIVE: SMOTE + Down-sample

**Strategy**:
1. SMOTE on BLS features to create synthetic planets
2. Down-sample majority class a bit
3. Result: Balanced with synthetic examples

**Why it's good**:
- ✅ SMOTE creates realistic feature combinations
- ✅ No exact duplicates (like up-sampling)

**Limitation for your project**:
- ⚠️ SMOTE works on **features** (period, depth, etc.), not raw time series
- ⚠️ You'd need to find nearest-neighbor time series for synthetic features
- ⚠️ More complex to implement correctly

### 🥉 SIMPLE: Up-sample only

**Strategy**: Just duplicate planets to match non-planets

**Use when**:
- Quick experiment needed
- You want to keep all data
- Overfitting not a major concern (you have regularization: dropout 0.4)

---

## How to Use the Script

### Install Required Package

```bash
conda activate exo-lstm-gpu
pip install imbalanced-learn
```

### Approach 1: Down-sample + Up-sample (RECOMMENDED)

```bash
cd C:\CS_4280_Project\Code

python balance_dataset_smote.py \
  --windows_dir "data/windows_train" \
  --output_dir "data/windows_balanced_downsample_upsample" \
  --method downsample_upsample \
  --target_size 300 \
  --seed 42
```

**Result**: 300 planets + 300 non-planets = 600 total

### Approach 2: SMOTE + Tomek (Advanced)

```bash
python balance_dataset_smote.py \
  --windows_dir "data/windows_train" \
  --output_dir "data/windows_balanced_smote" \
  --method smote_tomek \
  --seed 42
```

**Result**: Automatically balanced with synthetic examples

### Approach 3: Pure SMOTE

```bash
python balance_dataset_smote.py \
  --windows_dir "data/windows_train" \
  --output_dir "data/windows_balanced_smote_only" \
  --method smote \
  --seed 42
```

---

## Training on Balanced Dataset

After balancing, train your model:

```bash
python train_bilstm_cluster.py \
  --windows_dir "data/windows_balanced_downsample_upsample" \
  --n_clusters 5 \
  --epochs 80 \
  --batch_size 128 \
  --lr 0.000225 \
  --hidden 256 \
  --layers 4 \
  --dropout 0.311 \
  --save_dir "runs/bilstm_cluster_balanced" \
  --amp_dtype fp16 \
  --pos_weight 1.0 \
  --num_workers 0
```

**Note**: Set `--pos_weight 1.0` since data is now balanced!

---

## Comparison: Your Approaches vs Professor's

| Approach | Your Attempt | Professor's Suggestion | Key Difference |
|----------|--------------|------------------------|----------------|
| **Synthetic Data** | Batman package (physics-based) | SMOTE (data-driven) | SMOTE interpolates **real** examples |
| **Problem** | Domain shift (8× depth mismatch) | No domain shift (stays in real distribution) | SMOTE guarantees similarity |
| **Data Type** | Raw time series | BLS features → time series | SMOTE works on features |
| **Balance** | 50/50 (but wrong distribution) | 50/50 (correct distribution) | In-distribution is key |

**Key Insight**: Your batman approach tried to **simulate physics**. SMOTE just **mixes real examples**. Much safer!

---

## Expected Results

### Current (Unbalanced)
- Dataset: 150 planets (23%) + 505 non-planets (77%)
- AUC: 0.7572
- Recall: 0.8867 (88.67% of planets found)
- Precision: 0.3827 (38.27% of predictions are planets)
- Problem: High false positive rate, model biased

### After Balancing (Predicted)
- Dataset: 300 planets (50%) + 300 non-planets (50%)
- Expected AUC: 0.78-0.82
- Expected Recall: 0.85-0.90 (maintained)
- Expected Precision: 0.55-0.70 (improved!)
- Improvement: Better precision-recall balance

---

## Quick Decision Tree

```
Do you have BLS features in meta.csv?
│
├─ YES → Use SMOTE + Down-sample (learns feature patterns)
│
└─ NO (or want simplest) → Use Down-sample + Up-sample
                            ↓
                     RECOMMENDED FOR YOUR PROJECT
```

---

## Summary for Your Report

**What to write**:

> "To address class imbalance (23% positive), I applied combined down-sampling
> and up-sampling as recommended by the instructor. The majority class (non-planets)
> was down-sampled from 505 to 300 examples, while the minority class (planets) was
> up-sampled from 150 to 300 examples through random replication. This yielded a
> balanced dataset of 600 windows (50% positive) without the domain shift issues
> encountered in physics-based synthetic generation."

**Cite**:
- Chawla et al. (2002) for SMOTE
- He & Garcia (2009) for learning from imbalanced data survey

---

## Next Steps

1. **Install imbalanced-learn**: `pip install imbalanced-learn`
2. **Run down-sample + up-sample** (recommended, 2 minutes)
3. **Train model** on balanced dataset (~30 minutes)
4. **Compare AUC** vs unbalanced (0.7572 baseline)
5. **Document improvement** for final paper

**Expected outcome**: Better precision, maintained recall, improved F1 score!

---

*Created: November 2025*
*Based on professor's feedback: "SMOTE, down sample then up sample"*
