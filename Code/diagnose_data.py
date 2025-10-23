"""
Complete Diagnostic Script for Exoplanet Training Data
Run this first to check your data quality before training

Usage:
    python diagnose_data.py
"""

import numpy as np
import pandas as pd
import os
import sys

def main():
    # Path to your training data
    data_dir = r"C:\CS_4280_Project\Code\data\windows_train"
    
    print("=" * 70)
    print(" EXOPLANET TRAINING DATA DIAGNOSTICS")
    print("=" * 70)
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"\n❌ ERROR: Directory not found: {data_dir}")
        print("Please update data_dir in this script to point to your windows_train folder")
        sys.exit(1)
    
    # Check for required files
    required_files = ["X.npy", "y.npy", "meta.csv"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(data_dir, f))]
    
    if missing:
        print(f"\n❌ ERROR: Missing files: {missing}")
        sys.exit(1)
    
    print(f"\n✓ Found all required files in: {data_dir}")
    
    # Load arrays
    print("\nLoading data...")
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    meta = pd.read_csv(os.path.join(data_dir, "meta.csv"))
    
    print("\n" + "=" * 70)
    print(" DATA SHAPE")
    print("=" * 70)
    print(f"  X shape:    {X.shape} (samples, sequence_length, features)")
    print(f"  y shape:    {y.shape}")
    print(f"  meta shape: {meta.shape}")
    
    # Infer dimensions
    n_samples = X.shape[0]
    seq_len = X.shape[1] if len(X.shape) > 1 else None
    n_features = X.shape[2] if len(X.shape) > 2 else 1
    
    print(f"\n  Samples:        {n_samples}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Features:       {n_features}")
    
    print("\n" + "=" * 70)
    print(" CLASS DISTRIBUTION")
    print("=" * 70)
    
    n_positive = int(np.sum(y))
    n_negative = len(y) - n_positive
    pct_positive = 100 * np.mean(y)
    
    print(f"  Positive (planets):     {n_positive:4d} ({pct_positive:.1f}%)")
    print(f"  Negative (non-planets): {n_negative:4d} ({100-pct_positive:.1f}%)")
    
    if n_positive > 0:
        imbalance_ratio = n_negative / n_positive
        print(f"  Class imbalance ratio:  {imbalance_ratio:.2f}:1")
        print(f"  Recommended pos_weight: {imbalance_ratio:.3f}")
    
    print("\n" + "=" * 70)
    print(" DATA QUALITY CHECKS")
    print("=" * 70)
    
    # Check for problematic values
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    
    print(f"  NaN values:  {n_nan}")
    print(f"  Inf values:  {n_inf}")
    print(f"  X range:     [{X.min():.6f}, {X.max():.6f}]")
    print(f"  X mean:      {X.mean():.6f}")
    print(f"  X std:       {X.std():.6f}")
    
    print("\n" + "=" * 70)
    print(" METADATA")
    print("=" * 70)
    print(f"  Columns: {meta.columns.tolist()}")
    print(f"\n  First 5 rows:")
    print(meta.head(5).to_string())
    
    # Check for source distribution if column exists
    source_cols = [c for c in meta.columns if any(x in c.lower() for x in ['source', 'file', 'star', 'tic', 'kic'])]
    if source_cols:
        print(f"\n  Source distribution ({source_cols[0]}):")
        counts = meta[source_cols[0]].value_counts().head(10)
        for source, count in counts.items():
            print(f"    {source}: {count}")
    
    print("\n" + "=" * 70)
    print(" SAMPLE EXAMPLES")
    print("=" * 70)
    
    # Show positive examples
    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) > 0:
        print(f"\n  Positive Examples (showing up to 5):")
        for i, idx in enumerate(pos_idx[:5]):
            print(f"\n    Example {i+1} (index {idx}):")
            for col in meta.columns:
                print(f"      {col}: {meta.iloc[idx][col]}")
    
    # Show negative examples
    neg_idx = np.where(y == 0)[0]
    if len(neg_idx) > 0:
        print(f"\n  Negative Examples (showing up to 5):")
        for i, idx in enumerate(neg_idx[:5]):
            print(f"\n    Example {i+1} (index {idx}):")
            for col in meta.columns:
                print(f"      {col}: {meta.iloc[idx][col]}")
    
    print("\n" + "=" * 70)
    print(" RECOMMENDATIONS")
    print("=" * 70)
    
    issues = []
    warnings = []
    
    # Critical issues
    if n_nan > 0:
        issues.append(f"❌ CRITICAL: {n_nan} NaN values detected! Must clean data before training.")
    if n_inf > 0:
        issues.append(f"❌ CRITICAL: {n_inf} Inf values detected! Must clean data before training.")
    if n_positive == 0 or n_negative == 0:
        issues.append("❌ CRITICAL: Only one class present! Cannot train binary classifier.")
    
    # Warnings
    if pct_positive < 10 or pct_positive > 90:
        warnings.append(f"⚠️  Severe class imbalance ({pct_positive:.1f}% positive)")
        warnings.append(f"   → Use pos_weight={imbalance_ratio:.3f} in BCE loss")
        warnings.append("   → Or use WeightedRandomSampler")
    elif pct_positive < 30 or pct_positive > 70:
        warnings.append(f"⚠️  Moderate class imbalance ({pct_positive:.1f}% positive)")
        warnings.append(f"   → Consider pos_weight={imbalance_ratio:.3f}")
    
    if X.std() > 10:
        warnings.append(f"⚠️  High variance (std={X.std():.2f})")
        warnings.append("   → Consider normalization or standardization")
    
    if X.std() < 0.01:
        warnings.append(f"⚠️  Very low variance (std={X.std():.6f})")
        warnings.append("   → Check if data is properly scaled")
    
    if issues:
        print("\n🚨 CRITICAL ISSUES (must fix before training):")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n✅ Data looks good! Ready for training.")
    
    print("\n" + "=" * 70)
    print(" NEXT STEPS")
    print("=" * 70)
    
    if not issues:
        print("\n1. Use the fixed training script: train_lstm_fixed.py")
        print("2. Key parameters to set:")
        print(f"   --num_workers 0  (fixes Windows multiprocessing)")
        if n_positive > 0:
            print(f"   --pos_weight {imbalance_ratio:.3f}  (handles class imbalance)")
        print("   --lr 1e-4  (conservative learning rate)")
        print("\n3. Monitor training carefully:")
        print("   - Loss should decrease steadily")
        print("   - AUC should improve above 0.7")
        print("   - Precision and recall should balance")
    else:
        print("\n❌ Fix critical issues before training!")
        print("\nTo fix NaN/Inf values, rebuild your windows with:")
        print("  - Check source light curve files for bad data")
        print("  - Add data validation in window building script")
        print("  - Use np.nan_to_num() or drop bad windows")
    
    print("\n" + "=" * 70)
    print()

if __name__ == "__main__":
    main()
