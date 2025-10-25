"""
Deep investigation of training data to find why model won't learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
data_dir = r"C:\CS_4280_Project\Code\data\windows_train"
X = np.load(os.path.join(data_dir, "X.npy"))
y = np.load(os.path.join(data_dir, "y.npy"))
meta = pd.read_csv(os.path.join(data_dir, "meta.csv"))

print("="*70)
print(" INVESTIGATING WHY MODEL WON'T LEARN")
print("="*70)

# Check if positive and negative examples are actually different
print("\n1. Comparing positive vs negative windows:")
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]

pos_mean = X[pos_idx].mean(axis=0)
neg_mean = X[neg_idx].mean(axis=0)

difference = np.abs(pos_mean - neg_mean).mean()
print(f"   Mean absolute difference: {difference:.6f}")

if difference < 0.01:
    print("   ⚠️  WARNING: Positive and negative examples look nearly identical!")
    print("   This means the model has nothing to learn from.")
else:
    print("   ✓ Positive and negative examples are distinguishable")

# Check variance within each class
pos_std = X[pos_idx].std()
neg_std = X[neg_idx].std()
print(f"\n2. Within-class variance:")
print(f"   Positive std: {pos_std:.4f}")
print(f"   Negative std: {neg_std:.4f}")

# Look at individual examples from same star
print("\n3. Checking windows from same star (TIC 100229772):")
star_mask = meta['tic_id'] == 100229772
star_data = meta[star_mask]
star_X = X[star_mask]
star_y = y[star_mask]

print(f"   Total windows: {len(star_y)}")
print(f"   Positive: {int(star_y.sum())}")
print(f"   Negative: {int(len(star_y) - star_y.sum())}")

# Compare first positive and first negative from same star
first_pos_idx = np.where(star_y == 1)[0][0] if (star_y == 1).any() else None
first_neg_idx = np.where(star_y == 0)[0][0] if (star_y == 0).any() else None

if first_pos_idx is not None and first_neg_idx is not None:
    pos_window = star_X[first_pos_idx]
    neg_window = star_X[first_neg_idx]
    
    window_diff = np.abs(pos_window - neg_window).mean()
    print(f"   Difference between pos/neg window: {window_diff:.6f}")
    
    if window_diff < 0.01:
        print("   ⚠️  CRITICAL: Positive and negative windows from same star are identical!")
        print("   This is the problem - labels may be wrong or windows aren't centered on transits")

# Check the metadata more carefully
print("\n4. Metadata analysis:")
print(f"   Unique stars: {meta['tic_id'].nunique()}")
print(f"   Windows per star: {len(meta) / meta['tic_id'].nunique():.1f}")

# Group by star and check label distribution
star_labels = meta.groupby('tic_id')['label'].agg(['sum', 'count', 'mean'])
star_labels.columns = ['positive', 'total', 'positive_rate']
print("\n   Label distribution per star (first 10):")
print(star_labels.head(10).to_string())

# Check if all windows from same star have identical metadata
print("\n5. Checking if metadata is duplicated:")
tic_100229772 = meta[meta['tic_id'] == 100229772][['period', 'duration', 'depth', 't0', 'bls_power']].drop_duplicates()
print(f"   Unique parameter sets for TIC 100229772: {len(tic_100229772)}")
if len(tic_100229772) == 1:
    print("   ⚠️  All windows from this star have identical metadata!")
    print("   This suggests windows are just sliding windows, not centered on events")

# Statistical test: can we predict labels from the data?
print("\n6. Quick separability test:")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Flatten X for simple logistic regression
X_flat = X.reshape(len(X), -1)
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=100, class_weight='balanced')
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)

print(f"   Logistic Regression accuracy: {score:.3f}")
if score < 0.6:
    print("   ⚠️  CRITICAL: Even simple model can't separate classes!")
    print("   This confirms the data is not learnable.")
else:
    print("   ✓ Data should be learnable (simple model works)")

# Final diagnosis
print("\n" + "="*70)
print(" DIAGNOSIS")
print("="*70)

issues = []

if difference < 0.01:
    issues.append("Positive and negative windows are nearly identical")

if window_diff < 0.01:
    issues.append("Windows from same star don't differ between labels")

if score < 0.6:
    issues.append("Data is not linearly separable (may not be learnable)")

if len(tic_100229772) == 1:
    issues.append("All windows from same star have identical metadata")

if issues:
    print("\n🚨 CRITICAL ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    
    print("\n📋 LIKELY ROOT CAUSE:")
    print("   Your window building script is NOT properly labeling transit vs non-transit.")
    print("   All windows might be getting the same label regardless of content.")
    print("   OR windows are not properly aligned with transit events.")
    
    print("\n🔧 HOW TO FIX:")
    print("   1. Check your window building script (build_windows_*.py)")
    print("   2. Ensure positive windows are CENTERED on transit events")
    print("   3. Ensure negative windows are AWAY from transit events")
    print("   4. Verify labels in manifest.csv are correct")
    print("   5. Visualize some windows to manually verify they look different")
else:
    print("\n✓ Data looks learnable, but model architecture may be wrong")
    print("   Try: BiLSTM, or add CNN layers before LSTM")

print("\n" + "="*70)

# Optionally save some example windows for visual inspection
print("\n7. Saving example windows for inspection...")
save_dir = r"C:\CS_4280_Project\Code\data\window_examples"
os.makedirs(save_dir, exist_ok=True)

# Save first 3 positive and negative from TIC 100229772
for i in range(min(3, (star_y == 1).sum())):
    idx = np.where(star_y == 1)[0][i]
    np.savetxt(os.path.join(save_dir, f"positive_{i}.txt"), star_X[idx])

for i in range(min(3, (star_y == 0).sum())):
    idx = np.where(star_y == 0)[0][i]
    np.savetxt(os.path.join(save_dir, f"negative_{i}.txt"), star_X[idx])

print(f"   Saved to: {save_dir}")
print("   Open these files and compare positive vs negative by eye")
