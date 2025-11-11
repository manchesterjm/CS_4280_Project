"""
Build Training Windows from Synthetic Dataset
Uses known parameters from generation instead of BLS detection

Usage:
    python build_windows_from_synthetic.py --data_dir "C:\CS_4280_Project\synthetic_balanced_dataset" --output_dir "C:\CS_4280_Project\Code\data\windows_balanced_train" --seq_len 2048
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def extract_window(time, flux, center_idx, seq_len):
    """Extract a window of seq_len points centered at center_idx"""
    half_len = seq_len // 2
    start = max(0, center_idx - half_len)
    end = min(len(flux), center_idx + half_len)

    # If we don't have enough points, return None
    if end - start < seq_len:
        return None

    window_time = time[start:end]
    window_flux = flux[start:end]

    # Resample to exactly seq_len points
    if len(window_flux) != seq_len:
        indices = np.linspace(0, len(window_flux)-1, seq_len).astype(int)
        window_flux = window_flux[indices]

    return window_flux


def process_planet_lightcurve(lc_path, label_data, seq_len, n_windows_per_curve=3):
    """
    Process a planet light curve
    Extract windows centered on transit + random negative windows
    """
    # Load light curve
    df = pd.read_csv(lc_path)
    time = df['time'].values
    flux = df['flux'].values

    windows = []
    labels = []
    metadata = []

    # Get planet parameters from label_data
    period = label_data.get('period', None)

    if period is not None and not np.isnan(period):
        # Find transit times by detecting dips in the light curve
        # Smooth the light curve and find local minima
        from scipy.signal import find_peaks

        # Find dips (invert flux to find peaks)
        peaks, properties = find_peaks(-flux, distance=int(0.5 * period / (time[1] - time[0])), prominence=0.001)

        # Sort by prominence (deepest dips first)
        if len(peaks) > 0:
            prominences = properties['prominences']
            sorted_indices = np.argsort(prominences)[::-1]
            transit_indices = peaks[sorted_indices[:min(n_windows_per_curve, len(peaks))]]
        else:
            # If no peaks found, use first few deep points
            sorted_indices = np.argsort(flux)
            transit_indices = sorted_indices[:n_windows_per_curve]

        # Extract positive windows centered on detected transits
        for center_idx in transit_indices:
            window = extract_window(time, flux, center_idx, seq_len)
            if window is not None:
                windows.append(window)
                labels.append(1)  # Planet
                metadata.append({
                    'tic_id': label_data['tic_id'],
                    'label': 'planet',
                    'period': period,
                    'depth': label_data.get('depth', np.nan),
                    'duration': label_data.get('duration', np.nan),
                    't0': time[center_idx],
                    'bls_power': 1.0  # Synthetic data doesn't have BLS power
                })

    # Extract negative windows (random locations)
    n_neg = len(windows)  # Match number of positive windows
    for _ in range(n_neg):
        # Random center
        random_idx = np.random.randint(seq_len//2, len(time) - seq_len//2)

        window = extract_window(time, flux, random_idx, seq_len)
        if window is not None:
            windows.append(window)
            labels.append(0)  # Non-planet
            metadata.append({
                'tic_id': label_data['tic_id'],
                'label': 'non-planet',
                'period': period if period is not None else np.nan,
                'depth': np.nan,
                'duration': np.nan,
                't0': np.nan,
                'bls_power': 0.0
            })

    return windows, labels, metadata


def process_non_planet_lightcurve(lc_path, label_data, seq_len, n_windows_per_curve=3):
    """
    Process a non-planet light curve (flare, EB, noise, background)
    Extract random windows
    """
    # Load light curve
    df = pd.read_csv(lc_path)
    time = df['time'].values
    flux = df['flux'].values

    windows = []
    labels = []
    metadata = []

    # Extract random windows
    for _ in range(n_windows_per_curve):
        random_idx = np.random.randint(seq_len//2, len(time) - seq_len//2)

        window = extract_window(time, flux, random_idx, seq_len)
        if window is not None:
            windows.append(window)
            labels.append(0)  # Non-planet
            metadata.append({
                'tic_id': label_data['tic_id'],
                'label': label_data['label'],
                'period': np.nan,
                'depth': np.nan,
                'duration': np.nan,
                't0': np.nan,
                'bls_power': 0.0
            })

    return windows, labels, metadata


def build_windows(data_dir, output_dir, seq_len=2048, n_windows_per_curve=3, seed=42):
    """
    Build training windows from synthetic dataset
    """
    np.random.seed(seed)

    print("="*80)
    print(" BUILDING TRAINING WINDOWS FROM SYNTHETIC DATA")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Window length: {seq_len}")
    print(f"  Windows per curve: {n_windows_per_curve}")
    print(f"  Seed: {seed}\n")

    # Load manifest
    data_path = Path(data_dir)
    manifest_path = data_path / "manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(manifest_df)} light curves")
    print(f"Label distribution: {dict(manifest_df['label'].value_counts())}\n")

    # Process each light curve
    all_windows = []
    all_labels = []
    all_metadata = []

    print("Processing light curves...")
    for idx, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Light curves"):
        lc_path = data_path / row['curve_path']

        if not lc_path.exists():
            print(f"\n  Warning: Light curve not found: {lc_path}")
            continue

        # Convert row to dict
        label_data = row.to_dict()

        try:
            if row['label'] == 'planet':
                windows, labels, metadata = process_planet_lightcurve(
                    lc_path, label_data, seq_len, n_windows_per_curve
                )
            else:
                windows, labels, metadata = process_non_planet_lightcurve(
                    lc_path, label_data, seq_len, n_windows_per_curve
                )

            all_windows.extend(windows)
            all_labels.extend(labels)
            all_metadata.extend(metadata)

        except Exception as e:
            print(f"\n  Warning: Failed to process {lc_path.name}: {e}")
            continue

    # Convert to numpy arrays
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    print(f"\n{'='*80}")
    print(" DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"\nTotal windows: {len(X)}")
    print(f"Window shape: {X.shape}")
    print(f"Positive (planet): {np.sum(y == 1)} ({100*np.mean(y):.1f}%)")
    print(f"Negative (non-planet): {np.sum(y == 0)} ({100*np.mean(y == 0):.1f}%)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save windows
    print(f"\nSaving windows to: {output_dir}")
    np.save(output_path / "X.npy", X)
    np.save(output_path / "y.npy", y)

    # Save metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(output_path / "meta.csv", index=False)

    print(f"  X.npy: {X.shape}")
    print(f"  y.npy: {y.shape}")
    print(f"  meta.csv: {len(metadata_df)} rows")

    print(f"\n{'='*80}")
    print(" COMPLETE")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Build training windows from synthetic dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing synthetic dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for windows')
    parser.add_argument('--seq_len', type=int, default=2048,
                       help='Window length in points (default: 2048)')
    parser.add_argument('--n_windows', type=int, default=3,
                       help='Number of windows per light curve (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    args = parser.parse_args()

    build_windows(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        n_windows_per_curve=args.n_windows,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
