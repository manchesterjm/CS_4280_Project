"""
Combine TESS Sector 1 and Kepler datasets for cross-mission training.

This creates a combined training set with:
- TESS Sector 1 ground truth (mixed positive/negative)
- Kepler confirmed planets (all positive)

The combined dataset should improve model generalization across missions.

Usage:
    python combine_datasets.py --tess_dir data/windows_sector1_full/train --kepler_dir data/windows_kepler --output_dir data/windows_combined
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def load_dataset(data_dir):
    """Load X, y, and metadata from a directory."""
    data_dir = Path(data_dir)

    X = np.load(data_dir / 'X.npy')
    y = np.load(data_dir / 'y.npy')
    meta = pd.read_csv(data_dir / 'meta.csv')

    return X, y, meta


def main():
    parser = argparse.ArgumentParser(description='Combine TESS and Kepler datasets')
    parser.add_argument('--tess_dir', type=str, required=True,
                       help='TESS Sector 1 training data directory')
    parser.add_argument('--kepler_dir', type=str, required=True,
                       help='Kepler training data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for combined dataset')
    parser.add_argument('--kepler_fraction', type=float, default=1.0,
                       help='Fraction of Kepler data to include (default: all)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("COMBINING TESS + KEPLER DATASETS")
    print("=" * 60)

    # Load TESS data
    print(f"\n[loading] TESS Sector 1: {args.tess_dir}")
    X_tess, y_tess, meta_tess = load_dataset(args.tess_dir)
    print(f"  X.shape={X_tess.shape}")
    print(f"  Positive: {y_tess.sum()} ({100*y_tess.mean():.1f}%)")
    print(f"  Negative: {len(y_tess) - y_tess.sum()}")

    # Add source column
    meta_tess['source'] = 'tess_sector1'

    # Load Kepler data
    print(f"\n[loading] Kepler: {args.kepler_dir}")
    X_kepler, y_kepler, meta_kepler = load_dataset(args.kepler_dir)
    print(f"  X.shape={X_kepler.shape}")
    print(f"  Positive: {y_kepler.sum()} ({100*y_kepler.mean():.1f}%)")
    print(f"  Negative: {len(y_kepler) - y_kepler.sum()}")

    # Subsample Kepler if requested
    if args.kepler_fraction < 1.0:
        n_kepler = int(len(X_kepler) * args.kepler_fraction)
        idx = np.random.choice(len(X_kepler), n_kepler, replace=False)
        X_kepler = X_kepler[idx]
        y_kepler = y_kepler[idx]
        meta_kepler = meta_kepler.iloc[idx].reset_index(drop=True)
        print(f"  Subsampled to {n_kepler} windows ({args.kepler_fraction*100:.0f}%)")

    # Ensure meta columns match
    common_cols = list(set(meta_tess.columns) & set(meta_kepler.columns))
    meta_tess = meta_tess[common_cols]
    meta_kepler = meta_kepler[common_cols]

    # Combine
    print("\n[combining]")
    X_combined = np.concatenate([X_tess, X_kepler], axis=0)
    y_combined = np.concatenate([y_tess, y_kepler], axis=0)
    meta_combined = pd.concat([meta_tess, meta_kepler], ignore_index=True)

    print(f"  Combined X.shape={X_combined.shape}")
    print(f"  Combined positives: {y_combined.sum()} ({100*y_combined.mean():.1f}%)")
    print(f"  Combined negatives: {len(y_combined) - y_combined.sum()}")

    # Shuffle
    shuffle_idx = np.random.permutation(len(X_combined))
    X_combined = X_combined[shuffle_idx]
    y_combined = y_combined[shuffle_idx]
    meta_combined = meta_combined.iloc[shuffle_idx].reset_index(drop=True)

    # Save
    print(f"\n[saving] {output_dir}")
    np.save(output_dir / 'X.npy', X_combined)
    np.save(output_dir / 'y.npy', y_combined)
    meta_combined.to_csv(output_dir / 'meta.csv', index=False)

    print(f"  X.npy: {X_combined.nbytes / 1e6:.1f} MB")
    print(f"  y.npy: {y_combined.nbytes / 1e3:.1f} KB")
    print(f"  meta.csv: {len(meta_combined)} rows")

    # Summary by source
    print("\n[summary by source]")
    source_counts = meta_combined.groupby('source').agg({
        'label': ['count', 'sum', 'mean']
    })
    print(source_counts)

    # Calculate new pos_weight for training
    n_pos = y_combined.sum()
    n_neg = len(y_combined) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    print("\n" + "=" * 60)
    print("COMBINATION COMPLETE")
    print("=" * 60)
    print(f"\nRecommended pos_weight for training: {pos_weight:.2f}")
    print(f"\nTo train on combined data:")
    print(f"  python train_simple_bilstm.py \\")
    print(f"    --windows_dir {output_dir} \\")
    print(f"    --pos_weight {pos_weight:.2f} \\")
    print(f"    --save_dir runs/combined_model")


if __name__ == '__main__':
    main()
