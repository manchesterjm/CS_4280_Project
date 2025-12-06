"""
True SMOTE implementation using time-series flattening.

The previous approach used up-sampling (duplication), which caused overfitting.
This implements actual SMOTE that creates NEW synthetic examples by interpolation.

Author: Josh Manchester
Date: November 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import json


def load_windows(windows_dir):
    """Load window dataset."""
    windows_dir = Path(windows_dir)
    X = np.load(windows_dir / "X.npy")
    y = np.load(windows_dir / "y.npy")
    meta = pd.read_csv(windows_dir / "meta.csv")

    print(f"Loaded dataset:")
    print(f"  Total: {len(y)} windows")
    print(f"  Positive: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  Negative: {(1-y).sum()} ({100*(1-y.mean()):.1f}%)")

    return X, y, meta


def balance_with_smote_on_timeseries(X, y, meta, sampling_strategy='auto', random_state=42):
    """
    Apply SMOTE on flattened time series.

    This creates NEW synthetic planets by interpolating between real ones.
    Better than duplication!
    """
    print("\n" + "="*70)
    print("TRUE SMOTE: Interpolating Time Series")
    print("="*70)

    # Flatten time series (2048 points -> 2048 features)
    X_flat = X.reshape(len(X), -1)
    print(f"\nFlattened time series: {X_flat.shape}")

    # Apply SMOTE
    print("\nApplying SMOTE (this may take 1-2 minutes)...")
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=5,
        random_state=random_state
    )

    X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y)

    # Reshape back to time series (2D: samples, timesteps)
    X_resampled = X_resampled_flat.reshape(-1, X.shape[1])

    print(f"\nAfter SMOTE:")
    print(f"  Total: {len(y_resampled)} (was {len(y)})")
    print(f"  Positive: {y_resampled.sum()} (was {y.sum()})")
    print(f"  Negative: {(1-y_resampled).sum()} (was {(1-y).sum()})")
    print(f"  New synthetic planets created: {y_resampled.sum() - y.sum()}")

    # Update metadata (duplicate for synthetic samples)
    n_original = len(meta)
    n_total = len(y_resampled)

    # Replicate metadata
    meta_resampled = pd.concat([meta] * ((n_total // n_original) + 1), ignore_index=True)
    meta_resampled = meta_resampled.iloc[:n_total].copy()
    meta_resampled['label'] = y_resampled

    # Mark synthetic samples
    meta_resampled['synthetic'] = False
    meta_resampled.loc[n_original:, 'synthetic'] = True

    config = {
        "method": "smote_timeseries",
        "k_neighbors": 5,
        "sampling_strategy": str(sampling_strategy),
        "random_state": random_state,
        "original_size": len(y),
        "balanced_size": len(y_resampled),
        "synthetic_created": int(y_resampled.sum() - y.sum())
    }

    return X_resampled, y_resampled, meta_resampled, config


def balance_hybrid_smote_downsample(X, y, meta, target_size=None, random_state=42):
    """
    Hybrid: SMOTE on minority + down-sample majority.

    This is what your professor likely meant:
    1. Use SMOTE to increase planets (creates NEW interpolated planets)
    2. Down-sample non-planets
    3. Result: Balanced but with synthetic diversity
    """
    print("\n" + "="*70)
    print("HYBRID: SMOTE + Down-sample (Professor's Recommendation)")
    print("="*70)

    # Separate classes
    pos_mask = y == 1
    neg_mask = y == 0

    X_pos = X[pos_mask]
    y_pos = y[pos_mask]
    meta_pos = meta[pos_mask].copy()

    X_neg = X[neg_mask]
    y_neg = y[neg_mask]
    meta_neg = meta[neg_mask].copy()

    # Determine target size
    if target_size is None:
        target_size = (len(y_pos) + len(y_neg)) // 2

    print(f"\nTarget size per class: {target_size}")

    # Step 1: SMOTE to increase minority class
    if target_size > len(y_pos):
        print(f"\nStep 1: SMOTE to increase planets {len(y_pos)} -> {target_size}")

        # Flatten and apply SMOTE
        X_pos_flat = X_pos.reshape(len(X_pos), -1)

        # Calculate how many samples we need
        n_synthetic = target_size - len(y_pos)

        # SMOTE
        smote = SMOTE(
            sampling_strategy={1: target_size, 0: len(y_neg)},
            k_neighbors=min(5, len(y_pos)-1),  # Can't have more neighbors than samples
            random_state=random_state
        )

        # Combine pos and neg for SMOTE (SMOTE needs both classes)
        X_combined = np.vstack([X_pos, X_neg])
        y_combined = np.hstack([y_pos, y_neg])
        X_combined_flat = X_combined.reshape(len(X_combined), -1)

        X_smote_flat, y_smote = smote.fit_resample(X_combined_flat, y_combined)
        # Reshape back to 2D time series (samples, timesteps)
        X_smote = X_smote_flat.reshape(-1, X.shape[1])

        # Extract positive samples (now includes SMOTE-generated ones)
        pos_indices = np.where(y_smote == 1)[0]
        X_pos_balanced = X_smote[pos_indices][:target_size]
        y_pos_balanced = y_smote[pos_indices][:target_size]

        # Update metadata
        meta_pos_balanced = pd.concat([meta_pos] * ((target_size // len(meta_pos)) + 1), ignore_index=True)
        meta_pos_balanced = meta_pos_balanced.iloc[:target_size].copy()
        meta_pos_balanced['label'] = 1
        meta_pos_balanced['synthetic'] = False
        meta_pos_balanced.loc[len(y_pos):, 'synthetic'] = True

        print(f"  Created {n_synthetic} synthetic planets via SMOTE")
    else:
        X_pos_balanced = X_pos
        y_pos_balanced = y_pos
        meta_pos_balanced = meta_pos

    # Step 2: Down-sample majority class
    print(f"\nStep 2: Down-sample non-planets {len(y_neg)} -> {target_size}")

    if target_size < len(y_neg):
        indices_neg = resample(
            np.arange(len(y_neg)),
            n_samples=target_size,
            replace=False,
            random_state=random_state
        )
        X_neg_balanced = X_neg[indices_neg]
        y_neg_balanced = y_neg[indices_neg]
        meta_neg_balanced = meta_neg.iloc[indices_neg].copy()
    else:
        X_neg_balanced = X_neg
        y_neg_balanced = y_neg
        meta_neg_balanced = meta_neg

    meta_neg_balanced['synthetic'] = False

    # Combine
    X_balanced = np.vstack([X_pos_balanced, X_neg_balanced])
    y_balanced = np.hstack([y_pos_balanced, y_neg_balanced])
    meta_balanced = pd.concat([meta_pos_balanced, meta_neg_balanced], ignore_index=True)

    # Shuffle
    shuffle_idx = np.random.RandomState(random_state).permutation(len(y_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    meta_balanced = meta_balanced.iloc[shuffle_idx].reset_index(drop=True)

    print(f"\nFinal balanced dataset:")
    print(f"  Total: {len(y_balanced)}")
    print(f"  Planets: {y_balanced.sum()} ({100*y_balanced.mean():.1f}%)")
    print(f"  Non-planets: {(1-y_balanced).sum()} ({100*(1-y_balanced.mean()):.1f}%)")

    config = {
        "method": "hybrid_smote_downsample",
        "target_size_per_class": target_size,
        "random_state": random_state,
        "original_size": len(y),
        "balanced_size": len(y_balanced),
        "synthetic_created": int(y_balanced.sum() - y.sum()) if target_size > len(y_pos) else 0
    }

    return X_balanced, y_balanced, meta_balanced, config


def save_windows(X, y, meta, output_dir, config):
    """Save balanced dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X.astype(np.float32))
    np.save(output_dir / "y.npy", y.astype(np.int64))
    meta.to_csv(output_dir / "meta.csv", index=False)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="True SMOTE balancing (interpolation, not duplication)")
    parser.add_argument("--windows_dir", type=str, default="data/windows_train")
    parser.add_argument("--output_dir", type=str, default="data/windows_smote_true")
    parser.add_argument("--method", type=str, default="hybrid",
                       choices=["smote_only", "hybrid"],
                       help="Balancing method")
    parser.add_argument("--target_size", type=int, default=300,
                       help="Target size per class")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load
    X, y, meta = load_windows(args.windows_dir)

    # Balance
    if args.method == "smote_only":
        X_bal, y_bal, meta_bal, config = balance_with_smote_on_timeseries(
            X, y, meta, sampling_strategy='auto', random_state=args.seed
        )
    elif args.method == "hybrid":
        X_bal, y_bal, meta_bal, config = balance_hybrid_smote_downsample(
            X, y, meta, target_size=args.target_size, random_state=args.seed
        )

    # Save
    save_windows(X_bal, y_bal, meta_bal, args.output_dir, config)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print("\nNext: Train on this dataset and compare with unbalanced model")
    print(f'python train_bilstm_cluster.py --windows_dir "{args.output_dir}" ...')


if __name__ == "__main__":
    main()
