"""
Build training/test windows from TESS Sector 1 ground truth dataset.
Uses master CSV for label verification and includes 80/20 train/test split.
Optimized with parallel processing and GPU-ready output.
"""
import os

# Limit CPU threads BEFORE importing numpy/sklearn to prevent system overload
# When using parallel workers, each worker should use minimal threads
import multiprocessing as mp
os.environ['OMP_NUM_THREADS'] = '2'  # Each worker uses max 2 threads
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def read_light_curve_gz(filepath):
    """Read gzipped light curve file."""
    with gzip.open(filepath, 'rt') as f:
        flux = [float(line.strip()) for line in f if line.strip()]
    return np.array(flux, dtype=np.float32)


def normalize_flux(flux):
    """Robust z-score normalization using median and MAD."""
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    if mad > 0:
        return (flux - median) / (1.4826 * mad)
    else:
        return flux - median


def compute_statistical_features(window):
    """Compute statistical features for K-means clustering."""
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


def extract_windows(flux, seq_len=2048, n_windows=3, seed=42):
    """Extract multiple windows from a light curve."""
    rng = np.random.RandomState(seed)

    if len(flux) < seq_len:
        padded = np.pad(flux, (0, seq_len - len(flux)), mode='edge')
        return [padded]

    max_start = len(flux) - seq_len
    if max_start <= 0:
        return [flux[:seq_len]]

    windows = []
    if n_windows == 1:
        start = max_start // 2
        windows.append(flux[start:start + seq_len])
    else:
        # Spread windows evenly with some randomness
        step = max(1, max_start // (n_windows + 1))
        for i in range(n_windows):
            base_start = i * step
            jitter = rng.randint(0, max(1, step // 2))
            start = min(base_start + jitter, max_start)
            windows.append(flux[start:start + seq_len])

    return windows


def process_single_file(args):
    """Process a single light curve file (for parallel processing)."""
    filepath, tic_id, label, label_str, seq_len, n_windows = args

    try:
        flux = read_light_curve_gz(filepath)
        if len(flux) == 0:
            return None

        flux = normalize_flux(flux)
        windows = extract_windows(flux, seq_len, n_windows, seed=hash(tic_id) % (2**31))

        results = []
        for i, window in enumerate(windows):
            features = compute_statistical_features(window)
            results.append({
                'window': window,
                'label': label,
                'tic_id': tic_id,
                'category': label_str,
                'window_idx': i,
                **features
            })
        return results
    except Exception as e:
        return None


def build_dataset(data_dir, output_dir, seq_len=2048, n_windows=3,
                  test_size=0.2, n_jobs=-1, seed=42):
    """
    Build train/test datasets from TESS Sector 1 ground truth.

    Args:
        data_dir: Path to ground-truth directory
        output_dir: Output directory for processed data
        seq_len: Window length in timesteps
        n_windows: Windows per light curve
        test_size: Fraction for test set (default 0.2 = 80/20 split)
        n_jobs: Number of parallel workers (-1 = all CPUs)
        seed: Random seed for reproducibility
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of workers (leave 1 core free for system stability)
    if n_jobs == -1:
        n_jobs = max(1, mp.cpu_count() - 1)  # Use 11 of 12 cores

    print(f"\n{'='*70}")
    print(f"TESS SECTOR 1 DATASET BUILDER v2")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Window size: {seq_len}")
    print(f"Windows per curve: {n_windows}")
    print(f"Train/Test split: {int((1-test_size)*100)}/{int(test_size*100)}")
    print(f"Parallel workers: {n_jobs}")
    print(f"{'='*70}\n")

    # Load master CSV for label verification
    master_csv = data_dir / "tsop301_sector1_groundtruth_master.csv"
    if not master_csv.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")

    master_df = pd.read_csv(master_csv)
    master_df['tic_id'] = master_df['tic_id'].astype(str)
    label_map = dict(zip(master_df['tic_id'], master_df['label']))

    print(f"Loaded master CSV: {len(master_df)} entries")
    print(f"Label distribution:")
    for label, count in master_df['label'].value_counts().items():
        print(f"  {label}: {count}")

    # Map string labels to binary
    binary_label_map = {
        'planet': 1,
        'star': 0,
        'eb': 0,
        'beb': 0
    }

    # Collect all files with their labels from master CSV
    categories = {
        'planets': 'planets',
        'Stars': 'Stars',
        'EBs': 'EBs',
        'BackEBs': 'BackEBs'
    }

    file_tasks = []
    skipped = 0

    for cat_name, cat_dir_name in categories.items():
        cat_dir = data_dir / cat_dir_name
        if not cat_dir.exists():
            print(f"Warning: {cat_dir} not found, skipping")
            continue

        files = list(cat_dir.glob('*.txt.gz'))
        print(f"\nFound {len(files)} files in {cat_name}")

        for filepath in files:
            # Extract TIC ID from filename
            filename = filepath.stem.replace('.txt', '')
            # Remove category prefix if present
            for prefix in ['planets_', 'Stars_', 'EBs_', 'BackEBs_']:
                filename = filename.replace(prefix, '')
            tic_id = filename

            # Look up label in master CSV
            if tic_id in label_map:
                label_str = label_map[tic_id]
                label = binary_label_map.get(label_str, 0)
                file_tasks.append((filepath, tic_id, label, label_str, seq_len, n_windows))
            else:
                skipped += 1

    print(f"\nTotal files to process: {len(file_tasks)}")
    print(f"Skipped (not in master CSV): {skipped}")

    # Process files in parallel
    print(f"\nProcessing with {n_jobs} workers...")
    all_results = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(process_single_file, task): task for task in file_tasks}

        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_results.extend(result)
                pbar.update(1)

    print(f"\nTotal windows extracted: {len(all_results)}")

    # Convert to arrays
    X = np.array([r['window'] for r in all_results], dtype=np.float32)
    y = np.array([r['label'] for r in all_results], dtype=np.int64)

    # Build metadata DataFrame
    meta_cols = ['tic_id', 'category', 'window_idx', 'label',
                 'mean', 'std', 'var', 'skew', 'range', 'median', 'mad', 'peak_to_peak']
    meta = pd.DataFrame([{k: r[k] for k in meta_cols} for r in all_results])

    # Stratified train/test split (by TIC ID to prevent data leakage)
    unique_tics = meta['tic_id'].unique()
    tic_labels = meta.groupby('tic_id')['label'].first().values

    train_tics, test_tics = train_test_split(
        unique_tics,
        test_size=test_size,
        stratify=tic_labels,
        random_state=seed
    )

    train_mask = meta['tic_id'].isin(train_tics)
    test_mask = meta['tic_id'].isin(test_tics)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    meta_train, meta_test = meta[train_mask].reset_index(drop=True), meta[test_mask].reset_index(drop=True)

    # Save datasets
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    np.save(train_dir / 'X.npy', X_train)
    np.save(train_dir / 'y.npy', y_train)
    meta_train.to_csv(train_dir / 'meta.csv', index=False)

    np.save(test_dir / 'X.npy', X_test)
    np.save(test_dir / 'y.npy', y_test)
    meta_test.to_csv(test_dir / 'meta.csv', index=False)

    # Also save combined for compatibility
    np.save(output_dir / 'X.npy', X_train)  # Training data as default
    np.save(output_dir / 'y.npy', y_train)
    meta_train.to_csv(output_dir / 'meta.csv', index=False)

    # Print summary
    print(f"\n{'='*70}")
    print(f"DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"\nTRAINING SET ({output_dir / 'train'}):")
    print(f"  Windows: {len(X_train)}")
    print(f"  Positive (planets): {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    print(f"  Negative (non-planets): {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"  Unique TIC IDs: {len(train_tics)}")

    print(f"\nTEST SET ({output_dir / 'test'}):")
    print(f"  Windows: {len(X_test)}")
    print(f"  Positive (planets): {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
    print(f"  Negative (non-planets): {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
    print(f"  Unique TIC IDs: {len(test_tics)}")

    print(f"\nStatistical features for clustering:")
    print(f"  {list(meta_cols[4:])}")

    print(f"\n{'='*70}")
    print(f"Dataset build complete!")
    print(f"{'='*70}")

    return {
        'train_windows': len(X_train),
        'test_windows': len(X_test),
        'train_positive_rate': np.sum(y_train == 1) / len(y_train),
        'test_positive_rate': np.sum(y_test == 1) / len(y_test)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build TESS Sector 1 dataset')
    parser.add_argument('--data_dir', type=str,
                       default=r'E:\lilith4_sector-1_groundtruth\sector-1\ground-truth',
                       help='Path to ground-truth directory')
    parser.add_argument('--output_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\data\sector1_v2',
                       help='Output directory')
    parser.add_argument('--seq_len', type=int, default=2048,
                       help='Window length')
    parser.add_argument('--n_windows', type=int, default=3,
                       help='Windows per light curve')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set fraction (default 0.2 for 80/20 split)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Parallel workers (-1 = all CPUs)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    build_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        n_windows=args.n_windows,
        test_size=args.test_size,
        n_jobs=args.n_jobs,
        seed=args.seed
    )
