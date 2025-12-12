"""
Build training windows from Kepler light curves for fine-tuning.

This script processes Kepler light curve CSV files into the same format
as TESS windows (X.npy, y.npy, meta.csv) for domain adaptation.

All Kepler files are confirmed planets (label=1).

Usage:
    python build_kepler_windows.py
        --data_dir "D:/CS_4280_Project_Backup/kepler_test_data/raw"
        --output_dir "data/windows_kepler"
        --seq_len 2048
        --n_windows 3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_flux(flux):
    """
    Apply robust z-score normalization (same as TESS pipeline).

    Args:
        flux: raw flux array

    Returns:
        normalized flux array
    """
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    if mad > 0:
        return (flux - median) / (1.4826 * mad)
    return flux - median


def extract_windows(flux, seq_len=2048, n_windows=3, seed=42):
    """
    Extract multiple sliding windows from a light curve.

    Args:
        flux: Normalized flux array
        seq_len: Window length (default 2048)
        n_windows: Number of windows to extract (default 3)
        seed: Random seed

    Returns:
        List of windows (each seq_len points)
    """
    np.random.seed(seed)

    if len(flux) < seq_len:
        padded = np.pad(flux, (0, seq_len - len(flux)), mode='edge')
        return [padded]

    max_start = len(flux) - seq_len
    if max_start <= 0:
        return [flux[:seq_len]]

    if n_windows == 1:
        start = max_start // 2
        return [flux[start:start + seq_len]]

    windows = []
    step = max(1, max_start // (n_windows + 1))
    for i in range(n_windows):
        start = min(i * step, max_start)
        windows.append(flux[start:start + seq_len])

    return windows


def compute_window_stats(window):
    """
    Compute statistical features for clustering.

    Args:
        window: flux window array

    Returns:
        dict of statistical features
    """
    return {
        'mean': float(np.mean(window)),
        'std': float(np.std(window)),
        'var': float(np.var(window)),
        'skew': float(np.percentile(window, 75) - np.percentile(window, 25)),
        'range': float(np.max(window) - np.min(window)),
        'median': float(np.median(window)),
        'mad': float(np.median(np.abs(window - np.median(window)))),
        'peak_to_peak': float(np.ptp(window))
    }


def extract_kepler_id(filepath):
    """
    Extract Kepler ID from various filename formats.

    Supported formats:
        - "['kplr001026957']_lightcurve.csv" -> "kplr001026957"
        - "Kepler-11_lightcurve.csv" -> "Kepler-11"

    Args:
        filepath: Path object for the CSV file

    Returns:
        string with Kepler ID
    """
    filename = filepath.stem
    # Format 1: "['kplrXXXX']_lightcurve"
    if filename.startswith("['kplr"):
        return filename.replace("['", "").replace("']_lightcurve", "")
    # Format 2: "Kepler-XX_lightcurve"
    if filename.startswith("Kepler-"):
        return filename.replace("_lightcurve", "")
    # Fallback: return as-is
    return filename.replace("_lightcurve", "")


def process_kepler_file(filepath, seq_len, n_windows, seed):
    """
    Process a single Kepler light curve CSV file.

    Args:
        filepath: Path to CSV file
        seq_len: Window length
        n_windows: Number of windows to extract
        seed: Random seed

    Returns:
        windows, labels, metadata lists
    """
    windows = []
    labels = []
    metadata = []

    try:
        kepler_id = extract_kepler_id(filepath)

        # Read CSV with time, flux columns
        df = pd.read_csv(filepath)
        flux = df['flux'].values

        if len(flux) == 0:
            return windows, labels, metadata

        # Normalize flux
        flux = normalize_flux(flux)

        # Extract windows
        window_seed = hash(kepler_id) % (2**31)
        extracted = extract_windows(flux, seq_len, n_windows, window_seed)

        # All Kepler files are confirmed planets
        label = 1

        for i, window in enumerate(extracted):
            windows.append(window)
            labels.append(label)

            meta = {
                'kepler_id': kepler_id,
                'source': 'kepler',
                'window_idx': i,
                'label': label
            }
            meta.update(compute_window_stats(window))
            metadata.append(meta)

    except (ValueError, IOError, KeyError) as err:
        print(f"Error processing {filepath.name}: {err}")

    return windows, labels, metadata


def process_kepler_dataset(data_dirs, output_dir, seq_len=2048,
                           n_windows=3, seed=42):
    """
    Process all Kepler light curves into training windows.

    Args:
        data_dirs: List of directories containing Kepler CSV files
        output_dir: Where to save processed windows
        seq_len: Window length
        n_windows: Windows per light curve
        seed: Random seed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files from all directories (filter for lightcurve files)
    csv_files = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        if data_dir.exists():
            found = [f for f in data_dir.glob('*.csv')
                     if '_lightcurve' in f.name or f.name.startswith("[")]
            print(f"Found {len(found)} lightcurve files in {data_dir}")
            csv_files.extend(found)
    print(f"Total: {len(csv_files)} Kepler light curve files")

    all_windows = []
    all_labels = []
    all_metadata = []

    for filepath in tqdm(csv_files, desc="Processing Kepler"):
        windows, labels, metadata = process_kepler_file(
            filepath, seq_len, n_windows, seed
        )
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_metadata.extend(metadata)

    # Save dataset
    flux_data = np.array(all_windows, dtype=np.float32)
    label_array = np.array(all_labels, dtype=np.int64)
    meta_df = pd.DataFrame(all_metadata)

    print(f"\n{'='*70}")
    print("KEPLER DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total windows: {len(flux_data)}")
    print(f"All are confirmed planets (label=1)")
    print(f"Window shape: {flux_data.shape}")
    print(f"Unique Kepler IDs: {meta_df['kepler_id'].nunique()}")

    np.save(output_dir / 'X.npy', flux_data)
    np.save(output_dir / 'y.npy', label_array)
    meta_df.to_csv(output_dir / 'meta.csv', index=False)

    print(f"\n{'='*70}")
    print(f"Saved to: {output_dir}")
    print(f"  - X.npy: {flux_data.shape}")
    print(f"  - y.npy: {label_array.shape}")
    print(f"  - meta.csv: {len(meta_df)} rows")
    print(f"{'='*70}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build training windows from Kepler light curves'
    )
    parser.add_argument('--data_dirs', type=str, nargs='+',
                        default=[r'D:\kepler_lightcurves',
                                 r'D:\kepler_downloads'],
                        help='Paths to Kepler CSV directories')
    parser.add_argument('--output_dir', type=str,
                        default=r'D:\CS_4280_Project\Code\data\windows_kepler_5pct',
                        help='Output directory for processed windows')
    parser.add_argument('--seq_len', type=int, default=2048,
                        help='Window length in timesteps')
    parser.add_argument('--n_windows', type=int, default=3,
                        help='Number of windows per light curve')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    process_kepler_dataset(
        data_dirs=args.data_dirs,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        n_windows=args.n_windows,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
