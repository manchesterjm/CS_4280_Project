# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
"""
Build training windows from TESS Sector 1 ground truth dataset.

SOFA Refactored: December 3, 2025
Pylint Score: 10.00/10

Processes ground truth light curves into standardized training windows
with statistical features for clustering.

Dataset used for final model (December 6, 2025):
- Total: 33,051 windows (26,472 train / 6,579 test)
- Achieved: AUC 0.9261, 100% Recall

Usage:
    python build_windows_from_groundtruth.py
        --data_dir "D:/CS_4280_Project_Backup/sector-1/ground-truth"
        --output_dir data/windows_sector1_full --seq_len 2048 --n_windows 3
"""

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Light Curve Reading
# =============================================================================

def read_light_curve_gz(filepath):
    """
    Read gzipped light curve file.

    Args:
        filepath: path to .txt.gz file

    Returns:
        numpy array of flux values
    """
    with gzip.open(filepath, 'rt') as f:
        flux = [float(line.strip()) for line in f if line.strip()]
    return np.array(flux)


# =============================================================================
# Window Extraction
# =============================================================================

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


def normalize_flux(flux):
    """
    Apply robust z-score normalization.

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


# =============================================================================
# Dataset Processing
# =============================================================================

def get_category_definitions():
    """Return category definitions for ground truth data."""
    return {
        'planets': {'dir': 'planets', 'label': 1, 'prefix': 'planets_'},
        'stars': {'dir': 'Stars', 'label': 0, 'prefix': 'Stars_'},
        'ebs': {'dir': 'EBs', 'label': 0, 'prefix': 'EBs_'},
        'backebs': {'dir': 'BackEBs', 'label': 0, 'prefix': 'BackEBs_'}
    }


def process_category(cat_dir, cat_info, seq_len, n_windows_per_curve,
                     max_files, seed):
    """
    Process all files in a category.

    Args:
        cat_dir: category directory path
        cat_info: category info dict
        seq_len: window length
        n_windows_per_curve: windows per curve
        max_files: max files to process (None for all)
        seed: random seed

    Returns:
        windows, labels, metadata lists
    """
    files = list(cat_dir.glob('*.txt.gz'))

    if max_files:
        files = np.random.choice(
            files, min(len(files), max_files), replace=False
        ).tolist()

    windows = []
    labels = []
    metadata = []

    for filepath in tqdm(files, desc=cat_info['dir']):
        file_windows, file_labels, file_meta = process_single_file(
            filepath, cat_info, seq_len, n_windows_per_curve, seed
        )
        windows.extend(file_windows)
        labels.extend(file_labels)
        metadata.extend(file_meta)

    return windows, labels, metadata


def process_single_file(filepath, cat_info, seq_len, n_windows_per_curve, seed):
    """
    Process a single light curve file.

    Args:
        filepath: path to file
        cat_info: category info dict
        seq_len: window length
        n_windows_per_curve: windows per curve
        seed: random seed

    Returns:
        windows, labels, metadata lists
    """
    windows = []
    labels = []
    metadata = []

    try:
        tic_id = filepath.stem.replace('.txt', '').replace(cat_info['prefix'], '')
        flux = read_light_curve_gz(filepath)

        if len(flux) == 0:
            return windows, labels, metadata

        flux = normalize_flux(flux)
        window_seed = int(tic_id) if tic_id.isdigit() else seed
        extracted = extract_windows(flux, seq_len, n_windows_per_curve, window_seed)

        for i, window in enumerate(extracted):
            windows.append(window)
            labels.append(cat_info['label'])

            meta = {
                'tic_id': tic_id,
                'category': cat_info['dir'],
                'window_idx': i,
                'label': cat_info['label']
            }
            meta.update(compute_window_stats(window))
            metadata.append(meta)

    except (ValueError, IOError) as err:
        print(f"Error processing {filepath.name}: {err}")

    return windows, labels, metadata


def process_dataset(data_dir, output_dir, seq_len=2048, n_windows_per_curve=3,
                    max_files_per_category=None, seed=42):
    """
    Process ground truth dataset into training windows.

    Args:
        data_dir: Path to ground-truth directory
        output_dir: Where to save processed windows
        seq_len: Window length
        n_windows_per_curve: Windows per light curve
        max_files_per_category: Limit files per category
        seed: Random seed
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    master_csv = data_dir / "tsop301_sector1_groundtruth_master.csv"
    master_df = pd.read_csv(master_csv)
    print(f"\nLoaded master CSV: {len(master_df)} entries")
    print(f"Label distribution:\n{master_df['label'].value_counts()}")

    np.random.seed(seed)

    all_windows = []
    all_labels = []
    all_metadata = []

    categories = get_category_definitions()

    for cat_name, cat_info in categories.items():
        cat_dir = data_dir / cat_info['dir']
        if not cat_dir.exists():
            print(f"Warning: {cat_dir} not found, skipping")
            continue

        print(f"\nProcessing {cat_name}...")
        windows, labels, metadata = process_category(
            cat_dir, cat_info, seq_len, n_windows_per_curve,
            max_files_per_category, seed
        )

        all_windows.extend(windows)
        all_labels.extend(labels)
        all_metadata.extend(metadata)

    save_dataset(output_dir, all_windows, all_labels, all_metadata)


def save_dataset(output_dir, windows, labels, metadata):
    """
    Save processed dataset to disk.

    Args:
        output_dir: output directory path
        windows: list of window arrays
        labels: list of labels
        metadata: list of metadata dicts
    """
    flux_data = np.array(windows, dtype=np.float32)
    label_array = np.array(labels, dtype=np.int64)
    meta_df = pd.DataFrame(metadata)

    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total windows: {len(flux_data)}")
    print(f"Positive (planets): {np.sum(label_array == 1)} "
          f"({np.sum(label_array == 1)/len(label_array)*100:.1f}%)")
    print(f"Negative (non-planets): {np.sum(label_array == 0)} "
          f"({np.sum(label_array == 0)/len(label_array)*100:.1f}%)")
    print(f"Window shape: {flux_data.shape}")
    print("\nCategory breakdown:")
    print(meta_df['category'].value_counts())

    np.save(output_dir / 'X.npy', flux_data)
    np.save(output_dir / 'y.npy', label_array)
    meta_df.to_csv(output_dir / 'meta.csv', index=False)

    print(f"\n{'='*70}")
    print(f"Saved to: {output_dir}")
    print(f"  - X.npy: {flux_data.shape}")
    print(f"  - y.npy: {label_array.shape}")
    print(f"  - meta.csv: {len(meta_df)} rows")
    print(f"{'='*70}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build training windows from ground truth data'
    )
    parser.add_argument('--data_dir', type=str,
                        default=r'D:\CS_4280_Project_Backup\sector-1\ground-truth',
                        help='Path to ground-truth directory')
    parser.add_argument('--output_dir', type=str,
                        default=r'D:\CS_4280_Project\Code\data\windows_sector1',
                        help='Output directory for processed windows')
    parser.add_argument('--seq_len', type=int, default=2048,
                        help='Window length in timesteps')
    parser.add_argument('--n_windows', type=int, default=3,
                        help='Number of windows per light curve')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Max files per category (for testing)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    process_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        n_windows_per_curve=args.n_windows,
        max_files_per_category=args.max_files,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
