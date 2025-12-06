"""
Balance dataset using SMOTE, down-sampling, and up-sampling.

Based on professor's recommendation: "SMOTE, down sample then up sample"

Three approaches implemented:
1. SMOTE only (interpolate minority class)
2. Down-sample + Up-sample (combined resampling)
3. SMOTE + Down-sample (recommended)

Author: Josh Manchester
Date: November 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
import json


def load_windows(windows_dir):
    """Load window dataset."""
    windows_dir = Path(windows_dir)

    X = np.load(windows_dir / "X.npy")
    y = np.load(windows_dir / "y.npy")
    meta = pd.read_csv(windows_dir / "meta.csv")

    print(f"Loaded dataset:")
    print(f"  Total: {len(y)} windows")
    print(f"  Positive (planets): {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  Negative (non-planets): {(1-y).sum()} ({100*(1-y.mean()):.1f}%)")

    return X, y, meta


def save_windows(X, y, meta, output_dir, config):
    """Save balanced dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X.npy", X.astype(np.float32))
    np.save(output_dir / "y.npy", y.astype(np.int64))
    meta.to_csv(output_dir / "meta.csv", index=False)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved balanced dataset to: {output_dir}")
    print(f"  Total: {len(y)} windows")
    print(f"  Positive: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  Negative: {(1-y).sum()} ({100*(1-y.mean()):.1f}%)")


def balance_smote(X, y, meta, random_state=42):
    """
    Balance using SMOTE (Synthetic Minority Over-sampling Technique).

    Creates synthetic minority examples by interpolating between existing ones.
    """
    print("\n" + "="*60)
    print("APPROACH 1: SMOTE (Synthetic Minority Over-sampling)")
    print("="*60)

    # SMOTE works on 2D features, so we need to reshape our windows
    # Option 1: Use BLS features (recommended - what SMOTE is designed for)
    # Option 2: Flatten time series (less ideal but works)

    # We'll use BLS features if available
    feature_cols = ['period', 'depth', 'duration', 'bls_power']
    if all(col in meta.columns for col in feature_cols):
        print("\nUsing BLS features for SMOTE")
        X_features = meta[feature_cols].values
    else:
        print("\nBLS features not found, flattening time series")
        print("WARNING: This may be less effective than using BLS features")
        X_features = X.reshape(len(X), -1)

    # Apply SMOTE
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_features_resampled, y_resampled = smote.fit_resample(X_features, y)

    # For time series data, we need to handle this carefully
    # If we used BLS features, we need to generate corresponding time series
    if all(col in meta.columns for col in feature_cols):
        # Find which samples are original vs synthetic
        n_original = len(X)
        n_synthetic = len(X_features_resampled) - n_original

        print(f"\nSMOTE created {n_synthetic} synthetic minority examples")

        # For synthetic examples, we'll use nearest neighbor's time series
        # This is a limitation - SMOTE works best with feature vectors
        print("WARNING: SMOTE on BLS features only. Using nearest-neighbor time series for synthetic samples.")
        print("Consider using SMOTE on flattened time series or stick with down/up sampling.")

        # Get indices of original samples
        original_indices = np.where(y == 1)[0]  # Original positive samples

        # For simplicity, duplicate nearest neighbor time series
        # In practice, you might want to interpolate the time series too
        X_resampled = np.zeros((len(y_resampled), X.shape[1], X.shape[2]), dtype=np.float32)
        X_resampled[:n_original] = X  # Original samples

        # For synthetic samples, copy nearest neighbor
        for i in range(n_original, len(y_resampled)):
            # Find nearest original positive sample
            synthetic_features = X_features_resampled[i]
            distances = np.linalg.norm(X_features[original_indices] - synthetic_features, axis=1)
            nearest_idx = original_indices[np.argmin(distances)]
            X_resampled[i] = X[nearest_idx]

        # Update metadata
        meta_resampled = pd.concat([meta] * ((len(y_resampled) // len(meta)) + 1), ignore_index=True)
        meta_resampled = meta_resampled.iloc[:len(y_resampled)].copy()
        meta_resampled['label'] = y_resampled

    else:
        # We flattened the time series, reshape back
        X_resampled = X_features_resampled.reshape(-1, X.shape[1], X.shape[2])

        # Update metadata
        meta_resampled = pd.concat([meta] * ((len(y_resampled) // len(meta)) + 1), ignore_index=True)
        meta_resampled = meta_resampled.iloc[:len(y_resampled)].copy()
        meta_resampled['label'] = y_resampled

    config = {
        "method": "SMOTE",
        "k_neighbors": 5,
        "random_state": random_state,
        "original_size": len(y),
        "balanced_size": len(y_resampled)
    }

    return X_resampled, y_resampled, meta_resampled, config


def balance_downsample_upsample(X, y, meta, target_size=None, random_state=42):
    """
    Balance using combined down-sampling and up-sampling.

    Strategy:
    1. Down-sample majority class to target_size
    2. Up-sample minority class to target_size
    3. Result: 50/50 balance at target_size
    """
    print("\n" + "="*60)
    print("APPROACH 2: Down-sample + Up-sample")
    print("="*60)

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
        # Default: average of minority and majority
        target_size = (len(y_pos) + len(y_neg)) // 2

    print(f"\nTarget size per class: {target_size}")
    print(f"  Minority (planets): {len(y_pos)} → {target_size} ({'up' if target_size > len(y_pos) else 'down'}-sample)")
    print(f"  Majority (non-planets): {len(y_neg)} → {target_size} (down-sample)")

    # Down-sample majority class
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

    # Up-sample minority class
    if target_size > len(y_pos):
        indices_pos = resample(
            np.arange(len(y_pos)),
            n_samples=target_size,
            replace=True,  # Allow duplicates
            random_state=random_state
        )
        X_pos_balanced = X_pos[indices_pos]
        y_pos_balanced = y_pos[indices_pos]
        meta_pos_balanced = meta_pos.iloc[indices_pos].copy()
    else:
        indices_pos = resample(
            np.arange(len(y_pos)),
            n_samples=target_size,
            replace=False,
            random_state=random_state
        )
        X_pos_balanced = X_pos[indices_pos]
        y_pos_balanced = y_pos[indices_pos]
        meta_pos_balanced = meta_pos.iloc[indices_pos].copy()

    # Combine
    X_balanced = np.vstack([X_pos_balanced, X_neg_balanced])
    y_balanced = np.hstack([y_pos_balanced, y_neg_balanced])
    meta_balanced = pd.concat([meta_pos_balanced, meta_neg_balanced], ignore_index=True)

    # Shuffle
    shuffle_idx = np.random.RandomState(random_state).permutation(len(y_balanced))
    X_balanced = X_balanced[shuffle_idx]
    y_balanced = y_balanced[shuffle_idx]
    meta_balanced = meta_balanced.iloc[shuffle_idx].reset_index(drop=True)

    config = {
        "method": "downsample_upsample",
        "target_size_per_class": target_size,
        "random_state": random_state,
        "original_size": len(y),
        "balanced_size": len(y_balanced)
    }

    return X_balanced, y_balanced, meta_balanced, config


def balance_smote_downsample(X, y, meta, target_ratio=0.5, random_state=42):
    """
    RECOMMENDED: Combine SMOTE with down-sampling (SMOTETomek).

    This is the "best of both worlds" approach:
    1. SMOTE generates synthetic minority examples
    2. Tomek links remove boundary examples (cleaning)
    3. Optional down-sampling of majority
    """
    print("\n" + "="*60)
    print("APPROACH 3: SMOTE + Down-sample (RECOMMENDED)")
    print("="*60)

    # Use BLS features if available, otherwise flatten
    feature_cols = ['period', 'depth', 'duration', 'bls_power']
    if all(col in meta.columns for col in feature_cols):
        print("\nUsing BLS features")
        X_features = meta[feature_cols].values
        use_bls = True
    else:
        print("\nFlattening time series")
        X_features = X.reshape(len(X), -1)
        use_bls = False

    # Apply SMOTE + Tomek
    smotetomek = SMOTETomek(random_state=random_state)
    X_features_resampled, y_resampled = smotetomek.fit_resample(X_features, y)

    print(f"\nAfter SMOTE + Tomek:")
    print(f"  Total: {len(y_resampled)} (was {len(y)})")
    print(f"  Positive: {y_resampled.sum()} (was {y.sum()})")
    print(f"  Negative: {(1-y_resampled).sum()} (was {(1-y).sum()})")

    # Handle time series reconstruction
    if use_bls:
        print("\nWARNING: Using BLS features only. Time series are nearest-neighbor copies.")
        n_original = len(X)
        X_resampled = np.zeros((len(y_resampled), X.shape[1], X.shape[2]), dtype=np.float32)

        for i in range(len(y_resampled)):
            # Find nearest original sample
            distances = np.linalg.norm(X_features - X_features_resampled[i], axis=1)
            nearest_idx = np.argmin(distances)
            X_resampled[i] = X[nearest_idx]

        meta_resampled = pd.concat([meta] * ((len(y_resampled) // len(meta)) + 1), ignore_index=True)
        meta_resampled = meta_resampled.iloc[:len(y_resampled)].copy()
        meta_resampled['label'] = y_resampled
    else:
        X_resampled = X_features_resampled.reshape(-1, X.shape[1], X.shape[2])
        meta_resampled = pd.concat([meta] * ((len(y_resampled) // len(meta)) + 1), ignore_index=True)
        meta_resampled = meta_resampled.iloc[:len(y_resampled)].copy()
        meta_resampled['label'] = y_resampled

    config = {
        "method": "smote_tomek",
        "random_state": random_state,
        "original_size": len(y),
        "balanced_size": len(y_resampled)
    }

    return X_resampled, y_resampled, meta_resampled, config


def main():
    parser = argparse.ArgumentParser(description="Balance dataset using SMOTE and sampling techniques")
    parser.add_argument("--windows_dir", type=str, required=True, help="Input windows directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for balanced dataset")
    parser.add_argument("--method", type=str, default="downsample_upsample",
                       choices=["smote", "downsample_upsample", "smote_tomek"],
                       help="Balancing method")
    parser.add_argument("--target_size", type=int, default=None,
                       help="Target size per class (for downsample_upsample)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load data
    X, y, meta = load_windows(args.windows_dir)

    # Balance based on method
    if args.method == "smote":
        X_bal, y_bal, meta_bal, config = balance_smote(X, y, meta, random_state=args.seed)
    elif args.method == "downsample_upsample":
        X_bal, y_bal, meta_bal, config = balance_downsample_upsample(
            X, y, meta, target_size=args.target_size, random_state=args.seed
        )
    elif args.method == "smote_tomek":
        X_bal, y_bal, meta_bal, config = balance_smote_downsample(
            X, y, meta, random_state=args.seed
        )

    # Save
    save_windows(X_bal, y_bal, meta_bal, args.output_dir, config)

    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nBalanced dataset saved to: {args.output_dir}")
    print("\nNext step: Train model on balanced dataset:")
    print(f'python train_bilstm_cluster.py --windows_dir "{args.output_dir}" ...')


if __name__ == "__main__":
    main()
