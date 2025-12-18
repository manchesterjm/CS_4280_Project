"""
Build training windows from downloaded Kepler CSV light curves.

This processes the CSV files downloaded by download_kepler_planets.py
and converts them to training windows compatible with our BiLSTM model.

Usage:
    python build_windows_from_kepler_csv.py --data_dir E:\kepler_lightcurves --output_dir data\windows_kepler
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy import stats


def read_kepler_csv(filepath):
    """Read Kepler light curve CSV file."""
    try:
        df = pd.read_csv(filepath)
        time = df['time'].values
        flux = df['flux'].values

        # Remove NaN
        mask = ~np.isnan(time) & ~np.isnan(flux)
        time = time[mask]
        flux = flux[mask]

        if len(flux) < 100:
            return None, None

        return time, flux
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None


def normalize_flux(flux):
    """Normalize flux to zero mean, unit variance."""
    flux = flux.copy()

    # Remove outliers (5-sigma clipping)
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    sigma = 1.4826 * mad  # robust sigma

    mask = np.abs(flux - median) < 5 * sigma
    if mask.sum() > len(flux) * 0.5:
        flux = flux[mask]

    # Normalize
    flux = (flux - np.median(flux)) / (np.std(flux) + 1e-8)

    return flux


def extract_windows(flux, seq_len=2048, n_windows=3, seed=42):
    """Extract multiple windows from a light curve.

    All Kepler confirmed planets are labeled as positive (label=1).
    """
    np.random.seed(seed)

    if len(flux) < seq_len:
        # Pad if too short
        padded = np.pad(flux, (0, seq_len - len(flux)), mode='edge')
        # Normalize padded window
        padded = (padded - np.mean(padded)) / (np.std(padded) + 1e-8)
        # Sanitize
        padded = np.nan_to_num(padded, nan=0.0, posinf=0.0, neginf=0.0)
        padded = np.clip(padded, -10, 10)
        return [padded]

    # Extract windows at evenly spaced positions
    max_start = len(flux) - seq_len
    windows = []

    if n_windows == 1:
        start = max_start // 2
        window = flux[start:start + seq_len]
        window = (window - np.mean(window)) / (np.std(window) + 1e-8)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        window = np.clip(window, -10, 10)
        windows.append(window)
    else:
        step = max(1, max_start // (n_windows - 1))
        for i in range(n_windows):
            start = min(i * step, max_start)
            window = flux[start:start + seq_len]
            # Normalize each window independently
            window = (window - np.mean(window)) / (np.std(window) + 1e-8)
            # Sanitize: replace NaN/Inf and clip extreme values
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
            window = np.clip(window, -10, 10)  # Clip to reasonable range
            windows.append(window)

    return windows


def compute_statistical_features(flux):
    """Compute statistical features for clustering (same as Sector 1)."""
    return {
        'mean': float(np.mean(flux)),
        'std': float(np.std(flux)),
        'var': float(np.var(flux)),
        'skew': float(stats.skew(flux)),
        'range': float(np.ptp(flux)),
        'median': float(np.median(flux)),
        'mad': float(np.median(np.abs(flux - np.median(flux)))),
        'peak_to_peak': float(np.max(flux) - np.min(flux))
    }


def main():
    parser = argparse.ArgumentParser(description='Build windows from Kepler CSV files')
    parser.add_argument('--data_dir', type=str, default='E:\\kepler_lightcurves',
                       help='Directory containing Kepler CSV files')
    parser.add_argument('--output_dir', type=str, default='data\\windows_kepler',
                       help='Output directory for windows')
    parser.add_argument('--seq_len', type=int, default=2048,
                       help='Window length')
    parser.add_argument('--n_windows', type=int, default=3,
                       help='Windows per light curve')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BUILD WINDOWS FROM KEPLER LIGHT CURVES")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Windows per curve: {args.n_windows}")

    # Find all CSV files
    csv_files = list(data_dir.glob('*.csv'))
    print(f"\nFound {len(csv_files)} CSV files")

    if len(csv_files) == 0:
        print("No CSV files found! Make sure to run download_kepler_planets.py first.")
        return

    all_windows = []
    all_labels = []
    all_metadata = []

    np.random.seed(args.seed)

    # Process each light curve
    print("\nProcessing light curves...")
    for filepath in tqdm(csv_files, desc="Processing"):
        time, flux = read_kepler_csv(filepath)

        if time is None:
            continue

        # Normalize
        flux = normalize_flux(flux)

        # Extract windows
        windows = extract_windows(flux, args.seq_len, args.n_windows, args.seed)

        # Compute statistical features for the full light curve
        stat_features = compute_statistical_features(flux)

        # All Kepler confirmed planets = positive labels
        for i, window in enumerate(windows):
            all_windows.append(window)
            all_labels.append(1)  # All confirmed planets

            # Build metadata
            filename = filepath.stem
            meta = {
                'filename': filename,
                'window_idx': i,
                'label': 1,
                'n_points': len(flux),
                'source': 'kepler',
                **stat_features
            }
            all_metadata.append(meta)

    if len(all_windows) == 0:
        print("No windows extracted!")
        return

    # Convert to arrays
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    meta_df = pd.DataFrame(all_metadata)

    # Final sanity check - remove any remaining NaN/Inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"[warning] Found {nan_count} NaN and {inf_count} Inf values - replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n[result] X.shape={X.shape} y.shape={y.shape}")
    print(f"[result] All labels = 1 (confirmed exoplanets)")

    # Save
    np.save(output_dir / 'X.npy', X)
    np.save(output_dir / 'y.npy', y)
    meta_df.to_csv(output_dir / 'meta.csv', index=False)

    print(f"\n[saved]")
    print(f"  X.npy: {X.nbytes / 1e6:.1f} MB ({X.shape})")
    print(f"  y.npy: {y.nbytes / 1e3:.1f} KB")
    print(f"  meta.csv: {len(meta_df)} rows")

    # Show statistics
    print(f"\n[statistics]")
    print(f"  Total windows: {len(X)}")
    print(f"  Light curves processed: {len(csv_files)}")
    print(f"  Windows per curve: ~{len(X) / len(csv_files):.1f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("\nNext step: Combine with TESS Sector 1 data:")
    print("  python combine_datasets.py \\")
    print("    --tess_dir data/windows_sector1_full/train \\")
    print("    --kepler_dir data/windows_kepler \\")
    print("    --output_dir data/windows_combined")


if __name__ == '__main__':
    main()
