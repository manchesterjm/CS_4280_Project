"""
Build Hybrid Dataset: Combine Real and Synthetic Training Data

This script combines real TESS light curve windows with synthetic windows
at a specified ratio to test if synthetic data can improve model performance
when mixed with real data.

Usage:
    python build_hybrid_dataset.py --real_dir data/windows_train --synthetic_dir data/windows_train_400 --mix_ratio 0.75 --output_dir data/windows_hybrid
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle


def load_dataset(data_dir):
    """Load a window dataset (X, y, meta)"""
    data_dir = Path(data_dir)

    X = np.load(data_dir / 'X.npy')
    y = np.load(data_dir / 'y.npy')
    meta = pd.read_csv(data_dir / 'meta.csv')

    return X, y, meta


def combine_datasets(real_data, synthetic_data, real_ratio=0.75, seed=42):
    """
    Combine real and synthetic datasets at specified ratio

    Args:
        real_data: (X, y, meta) tuple for real data
        synthetic_data: (X, y, meta) tuple for synthetic data
        real_ratio: Fraction of real data to include (0-1)
        seed: Random seed

    Returns:
        Combined (X, y, meta) tuple
    """
    np.random.seed(seed)

    X_real, y_real, meta_real = real_data
    X_synth, y_synth, meta_synth = synthetic_data

    n_real = len(X_real)
    n_synth = len(X_synth)

    # Calculate target sizes
    if real_ratio >= 1.0:
        # Use all real data, no synthetic
        return real_data

    # Use all real data
    n_real_use = n_real

    # Calculate how much synthetic data to add
    # real_ratio = n_real / (n_real + n_synth_use)
    # Solve for n_synth_use:
    n_synth_use = int(n_real * (1 - real_ratio) / real_ratio)
    n_synth_use = min(n_synth_use, n_synth)  # Don't exceed available synthetic data

    print(f"Using {n_real_use} real windows + {n_synth_use} synthetic windows")
    print(f"Final ratio: {n_real_use / (n_real_use + n_synth_use):.2%} real")

    # Sample synthetic data
    synth_indices = np.random.choice(n_synth, n_synth_use, replace=False)
    X_synth_sample = X_synth[synth_indices]
    y_synth_sample = y_synth[synth_indices]
    meta_synth_sample = meta_synth.iloc[synth_indices].copy()

    # Combine
    X_combined = np.vstack([X_real, X_synth_sample])
    y_combined = np.concatenate([y_real, y_synth_sample])
    meta_combined = pd.concat([meta_real, meta_synth_sample], ignore_index=True)

    # Add source column
    meta_combined['source'] = ['real'] * n_real_use + ['synthetic'] * n_synth_use

    # Shuffle
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)

    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    meta_combined = meta_combined.iloc[indices].reset_index(drop=True)

    return X_combined, y_combined, meta_combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\data\windows_train',
                       help='Directory with real training windows')
    parser.add_argument('--synthetic_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\data\windows_train_400',
                       help='Directory with synthetic training windows')
    parser.add_argument('--mix_ratio', type=float, default=0.75,
                       help='Fraction of real data (0-1), e.g., 0.75 = 75% real, 25% synthetic')
    parser.add_argument('--output_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\data\windows_hybrid',
                       help='Output directory for hybrid dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    print("="*70)
    print(" BUILDING HYBRID DATASET")
    print("="*70)

    # Load datasets
    print("\nLoading real dataset...")
    real_data = load_dataset(args.real_dir)
    print(f"  Loaded {len(real_data[0])} real windows")
    print(f"  Positive rate: {real_data[1].mean():.2%}")

    print("\nLoading synthetic dataset...")
    synthetic_data = load_dataset(args.synthetic_dir)
    print(f"  Loaded {len(synthetic_data[0])} synthetic windows")

    # Handle label conversion for synthetic data
    if synthetic_data[2]['label'].dtype == 'object':
        # Convert string labels to numeric
        label_map = {'planet': 1, 'non-planet': 0, 'flare': 0, 'EB': 0, 'noise': 0, 'background': 0}
        synth_labels_numeric = synthetic_data[2]['label'].map(label_map).values
        synthetic_data = (synthetic_data[0], synth_labels_numeric, synthetic_data[2])
        print(f"  Converted string labels to numeric")

    print(f"  Positive rate: {synthetic_data[1].mean():.2%}")

    # Combine
    print(f"\nCombining datasets at {args.mix_ratio:.0%} real / {1-args.mix_ratio:.0%} synthetic...")
    X_hybrid, y_hybrid, meta_hybrid = combine_datasets(
        real_data, synthetic_data,
        real_ratio=args.mix_ratio,
        seed=args.seed
    )

    print(f"\nHybrid dataset created:")
    print(f"  Total windows: {len(X_hybrid)}")
    print(f"  Positive rate: {y_hybrid.mean():.2%}")
    print(f"  Shape: {X_hybrid.shape}")

    # Print class balance by source
    print(f"\nClass balance by source:")
    for source in ['real', 'synthetic']:
        mask = meta_hybrid['source'] == source
        n = mask.sum()
        pos_rate = y_hybrid[mask].mean()
        print(f"  {source:12s}: {n:4d} windows, {pos_rate:.2%} positive")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {output_dir}...")
    np.save(output_dir / 'X.npy', X_hybrid)
    np.save(output_dir / 'y.npy', y_hybrid)
    meta_hybrid.to_csv(output_dir / 'meta.csv', index=False)

    # Save config
    config = {
        'real_dir': str(args.real_dir),
        'synthetic_dir': str(args.synthetic_dir),
        'mix_ratio': args.mix_ratio,
        'seed': args.seed,
        'n_real': int((meta_hybrid['source'] == 'real').sum()),
        'n_synthetic': int((meta_hybrid['source'] == 'synthetic').sum()),
        'positive_rate': float(y_hybrid.mean()),
    }

    import json
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, indent=2, fp=f)

    print("✓ Hybrid dataset saved!")
    print("="*70)


if __name__ == '__main__':
    main()
