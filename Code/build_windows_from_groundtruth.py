"""Build training windows from TESS Sector 1 ground truth dataset."""
import os
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.preprocessing import StandardScaler


def read_light_curve_gz(filepath):
    """Read gzipped light curve file."""
    with gzip.open(filepath, 'rt') as f:
        flux = [float(line.strip()) for line in f if line.strip()]
    return np.array(flux)


def extract_windows(flux, seq_len=2048, n_windows=3, seed=42):
    """Extract multiple sliding windows from a light curve.

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
        # Pad if too short
        padded = np.pad(flux, (0, seq_len - len(flux)), mode='edge')
        return [padded]

    # Extract windows at random positions
    max_start = len(flux) - seq_len
    if max_start <= 0:
        return [flux[:seq_len]]

    windows = []
    if n_windows == 1:
        # Just take from the middle
        start = max_start // 2
        windows.append(flux[start:start + seq_len])
    else:
        # Random windows with no overlap
        step = max(1, max_start // (n_windows + 1))
        for i in range(n_windows):
            start = min(i * step, max_start)
            windows.append(flux[start:start + seq_len])

    return windows


def process_dataset(data_dir, output_dir, seq_len=2048, n_windows_per_curve=3,
                   max_files_per_category=None, seed=42):
    """Process ground truth dataset into training windows.

    Args:
        data_dir: Path to ground-truth directory
        output_dir: Where to save processed windows
        seq_len: Window length
        n_windows_per_curve: Windows per light curve
        max_files_per_category: Limit files per category (for testing)
        seed: Random seed
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read master CSV
    master_csv = data_dir / "tsop301_sector1_groundtruth_master.csv"
    df = pd.read_csv(master_csv)
    print(f"\nLoaded master CSV: {len(df)} entries")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Define categories and their labels
    categories = {
        'planets': {'dir': 'planets', 'label': 1, 'prefix': 'planets_'},
        'stars': {'dir': 'Stars', 'label': 0, 'prefix': 'Stars_'},
        'ebs': {'dir': 'EBs', 'label': 0, 'prefix': 'EBs_'},
        'backebs': {'dir': 'BackEBs', 'label': 0, 'prefix': 'BackEBs_'}
    }

    all_windows = []
    all_labels = []
    all_metadata = []

    np.random.seed(seed)

    # Process each category
    for cat_name, cat_info in categories.items():
        cat_dir = data_dir / cat_info['dir']
        if not cat_dir.exists():
            print(f"Warning: {cat_dir} not found, skipping")
            continue

        # Get all .gz files in this category
        files = list(cat_dir.glob('*.txt.gz'))

        if max_files_per_category:
            files = np.random.choice(files, min(len(files), max_files_per_category),
                                    replace=False).tolist()

        print(f"\nProcessing {cat_name}: {len(files)} files...")

        for filepath in tqdm(files, desc=cat_name):
            try:
                # Extract TIC ID from filename
                tic_id = filepath.stem.replace('.txt', '').replace(cat_info['prefix'], '')

                # Read light curve
                flux = read_light_curve_gz(filepath)

                if len(flux) == 0:
                    continue

                # Normalize (robust z-score)
                median = np.median(flux)
                mad = np.median(np.abs(flux - median))
                if mad > 0:
                    flux = (flux - median) / (1.4826 * mad)

                # Extract windows
                windows = extract_windows(flux, seq_len, n_windows_per_curve,
                                        seed=int(tic_id) if tic_id.isdigit() else seed)

                # Store windows and metadata
                for i, window in enumerate(windows):
                    all_windows.append(window)
                    all_labels.append(cat_info['label'])

                    # Compute statistical features for clustering
                    # (avoiding data leakage by using only light curve statistics)
                    all_metadata.append({
                        'tic_id': tic_id,
                        'category': cat_name,
                        'window_idx': i,
                        'label': cat_info['label'],
                        # Statistical features for clustering
                        'mean': float(np.mean(window)),
                        'std': float(np.std(window)),
                        'var': float(np.var(window)),
                        'skew': float(np.percentile(window, 75) - np.percentile(window, 25)),
                        'range': float(np.max(window) - np.min(window)),
                        'median': float(np.median(window)),
                        'mad': float(np.median(np.abs(window - np.median(window)))),
                        'peak_to_peak': float(np.ptp(window))
                    })

            except Exception as e:
                print(f"Error processing {filepath.name}: {e}")
                continue

    # Convert to arrays
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    meta = pd.DataFrame(all_metadata)

    print(f"\n{'='*70}")
    print(f"DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total windows: {len(X)}")
    print(f"Positive (planets): {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)")
    print(f"Negative (non-planets): {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)")
    print(f"Window shape: {X.shape}")
    print(f"\nCategory breakdown:")
    print(meta['category'].value_counts())

    # Save
    np.save(output_dir / 'X.npy', X)
    np.save(output_dir / 'y.npy', y)
    meta.to_csv(output_dir / 'meta.csv', index=False)

    print(f"\n{'='*70}")
    print(f"Saved to: {output_dir}")
    print(f"  - X.npy: {X.shape}")
    print(f"  - y.npy: {y.shape}")
    print(f"  - meta.csv: {len(meta)} rows")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                       default=r'E:\lilith4_sector-1_groundtruth\sector-1\ground-truth',
                       help='Path to ground-truth directory')
    parser.add_argument('--output_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\data\windows_sector1',
                       help='Output directory for processed windows')
    parser.add_argument('--seq_len', type=int, default=2048,
                       help='Window length in timesteps')
    parser.add_argument('--n_windows', type=int, default=3,
                       help='Number of windows per light curve')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Max files per category (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        n_windows_per_curve=args.n_windows,
        max_files_per_category=args.max_files,
        seed=args.seed
    )
