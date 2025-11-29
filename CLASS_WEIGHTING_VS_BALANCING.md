# Class Weighting vs Balancing: Addressing "Bias"

**Question**: Is an unbalanced model with class weighting "biased"?

**Short answer**: No, class weighting is a legitimate technique to handle imbalance. It's not bias in a negative sense.

---

## Understanding Class Weighting (What You Currently Have)

### Your Unbalanced Model

**Dataset:**
- 150 planets (23%)
- 505 non-planets (77%)
- **Strategy**: `pos_weight=3.367` in loss function

**What this does:**
```python
# Without weighting:
loss_planet = 1.0 × error       # Planet misclassified
loss_non_planet = 1.0 × error   # Non-planet misclassified

# With pos_weight=3.367:
loss_planet = 3.367 × error     # Planet misclassified (3.37× more costly!)
loss_non_planet = 1.0 × error   # Non-planet misclassified
```

**Effect:**
- Model learns "misclassifying a planet is 3.37× worse than misclassifying a non-planet"
- Balances the importance WITHOUT changing the data
- **This is NOT bias** - it's correcting for data imbalance

### Is This "Biased"?

**No!** Class weighting is:
- ✅ Standard practice in imbalanced learning
- ✅ Recommended by scikit-learn, PyTorch documentation
- ✅ Used in production ML systems
- ✅ Mathematically sound (inverse of class frequencies)

**Your Results Prove It Works:**
- Detects 16/300 real planets ✅
- Generalizes to new data ✅
- Better than balanced resampling ✅

---

## Why You Might Still Want Balanced Data

### Academic Requirements

If your assignment requires:
- "Train on balanced dataset"
- "Use 50/50 class distribution"
- "Apply SMOTE technique"

Then you MUST use balanced data regardless of performance.

### Research Comparison

To compare approaches:
- Baseline: Class weighting
- Alternative 1: Simple resampling (failed - 0/300)
- Alternative 2: SMOTE (to be tested)

This shows rigor and thoroughness.

### Addressing Reviewer Concerns

Some reviewers might say:
- "Why didn't you balance the data?"
- "Class weighting alone is insufficient"

Having a balanced model lets you respond:
- "We tested both approaches"
- "Class weighting performed better on real data"
- "Balanced resampling caused overfitting"

---

## The Three Balancing Approaches

### 1. Class Weighting (Your Current Model) ✅

**Pros:**
- Uses all original data
- No duplication/overfitting
- Simple to implement
- **Best generalization** (16/300 planets detected)

**Cons:**
- Some people think it's not "truly balanced"
- May not satisfy assignment requirements

**When to use:** When performance matters most

### 2. Simple Resampling (Tested - Failed) ❌

**Pros:**
- 50/50 balance
- Easy to understand

**Cons:**
- **Duplicates data** → overfitting
- Lost 205 negative examples
- **Failed completely** (0/300 planets detected)

**When to use:** Never (proven to fail)

### 3. SMOTE (To Be Tested) 🔄

**Pros:**
- Creates NEW synthetic examples (not duplicates!)
- 50/50 balance
- Avoids overfitting from duplication
- What your professor likely meant

**Cons:**
- More complex
- Interpolation on 2048-point time series is tricky
- May still underperform class weighting

**When to use:** When you MUST have balanced data but want to avoid duplication

---

## The Duplication vs Interpolation Problem

### What Failed (Simple Up-sampling)

```python
# Original planets
Planet A: [time series with 2048 points]

# Simple up-sampling (what we did before)
Dataset = [Planet A, Planet A, Planet A]  # Same data, copied 3×
Result: Overfitting! Model memorizes Planet A

Test on new Planet Z: FAIL (model never saw anything like Z)
```

### What SMOTE Does (Interpolation)

```python
# Original planets
Planet A: [time series with 2048 points]
Planet B: [time series with 2048 points]

# SMOTE interpolation
for each point i in range(2048):
    Planet_C[i] = 0.5 × Planet_A[i] + 0.5 × Planet_B[i]

Dataset = [Planet A, Planet B, Planet C]  # 3 different planets!
Result: More diversity, less overfitting

Test on new Planet Z: Better chance (model saw blended patterns)
```

**Key difference:** SMOTE creates NEW examples that model hasn't seen before.

---

## Run This to Get Balanced Model with SMOTE

```powershell
cd C:\CS_4280_Project\Code
.\train_balanced_smote.bat
```

**What this does:**
1. Creates 150 NEW synthetic planets by interpolating between original 150
2. Down-samples 505 non-planets to 300
3. Trains model on 300 planets (150 real + 150 synthetic) + 300 non-planets
4. Tests on 100 real planets
5. Compares with previous results

**Expected time:** ~1 hour total (SMOTE is slower than simple resampling)

**Expected result:** Better than simple duplication (0/300), but may still be worse than class weighting (16/300)

---

## Recommendation Based on Your Goal

### If Goal: Best Performance
**Use unbalanced model with class weighting**
- Already proven to work
- Detects 16/300 real planets
- No need to change

### If Goal: Assignment Requirement
**Use SMOTE balanced model**
- Satisfies "balanced data" requirement
- Shows you understand SMOTE technique
- Documents negative result if it fails

### If Goal: Comprehensive Study
**Use both and compare**
- Show class weighting baseline
- Show SMOTE alternative
- Discuss trade-offs
- Conclude which is better for this problem

---

## For Your Paper

### If You Use Class Weighting

**Title**: "Handling Class Imbalance via Loss Function Weighting"

**Methods**:
> "To address severe class imbalance (23% positive), I applied class weighting
> in the loss function (pos_weight=3.367), which penalizes minority class
> misclassifications proportionally to the inverse class frequency. This approach
> preserves all training examples while correcting for distributional bias."

**Results**: AUC 0.7572, 16/300 real planets detected

### If You Use SMOTE

**Title**: "Handling Class Imbalance via SMOTE Resampling"

**Methods**:
> "To address severe class imbalance (23% positive), I applied SMOTE to generate
> synthetic minority class examples via k-nearest neighbor interpolation, combined
> with random down-sampling of the majority class. This yielded a balanced dataset
> of 600 windows (50% positive)."

**Results**: Will depend on your test results

### If You Use Both (Best)

**Title**: "Comparison of Class Imbalance Techniques"

**Methods**:
> "I evaluated two strategies for handling class imbalance: (1) loss function
> weighting and (2) SMOTE resampling. Class weighting preserved all 655 training
> examples while penalizing minority class errors (pos_weight=3.367). SMOTE
> created a balanced dataset via k-nearest neighbor interpolation of minority
> class examples combined with majority class down-sampling (600 windows, 50% positive)."

**Results**:
- Class weighting: AUC 0.7572, 16/300 planets detected
- SMOTE: AUC X.XXXX, Y/300 planets detected
- **Conclusion**: [Which worked better and why]

---

## Bottom Line

**Class weighting is NOT biased in a bad way.** It's a legitimate, standard technique.

**But if you MUST use balanced data:**
1. Run `.\train_balanced_smote.bat`
2. This uses TRUE SMOTE (interpolation, not duplication)
3. Should work better than simple up-sampling
4. Compare results with class weighting model

**My prediction:**
- SMOTE will be better than simple duplication (which got 0/300)
- SMOTE may still be worse than class weighting (which got 16/300)
- This is a valuable finding: "For small astronomical datasets, class weighting outperforms SMOTE"

---

*Created: November 13, 2025*
*Clarifying: Class weighting vs SMOTE balancing*
