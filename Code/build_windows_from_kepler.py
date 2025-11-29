"""
Build training windows from Kepler FLTI (Flux-Level Transit Injection) data.
Similar to build_windows_from_groundtruth.py but for Kepler format.

Usage:
    python build_windows_from_kepler.py --data_dir E:\kepler_data\KSOC-5104 --output_dir data\windows_kepler
"""
import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from astropy.io import fits


def load_kepler_lightcurve(fits_path):
    """Load a Kepler FITS light curve file."""
    try:
        with fits.open(fits_path) as hdul:
            # Kepler light curves typically have data in extension 1
            data = hdul[1].data

            # Common column names in Kepler FITS files
            time = data['TIME']

            # Try different flux column names
            if 'PDCSAP_FLUX' in data.names:
                flux = data['PDCSAP_FLUX']
            elif 'SAP_FLUX' in data.names:
                flux = data['SAP_FLUX']
            elif 'FLUX' in data.names:
                flux = data['FLUX']
            else:
                return None, None

            # Remove NaN values
            mask = ~np.isnan(time) & ~np.isnan(flux)
            time = time[mask]
            flux = flux[mask]

            if len(time) < 100:
                return None, None

            # Normalize flux
            flux = flux / np.nanmedian(flux) - 1.0

            return time, flux
    except Exception as e:
        print(f"Error loading {fits_path}: {e}")
        return None, None


def load_kepler_csv(csv_path):
    """Load a Kepler CSV light curve file."""
    try:
        df = pd.read_csv(csv_path)

        # Find time and flux columns
        time_col = None
        flux_col = None

        for col in df.columns:
            col_lower = col.lower()
            if 'time' in col_lower or col_lower == 't':
                time_col = col
            elif 'flux' in col_lower or col_lower == 'f':
                flux_col = col

        if time_col is None or flux_col is None:
            # Assume first two columns are time and flux
            time_col = df.columns[0]
            flux_col = df.columns[1]

        time = df[time_col].values
        flux = df[flux_col].values

        # Remove NaN values
        mask = ~np.isnan(time) & ~np.isnan(flux)
        time = time[mask]
        flux = flux[mask]

        if len(time) < 100:
            return None, None

        # Normalize flux
        flux = flux / np.nanmedian(flux) - 1.0

        return time, flux
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None


def extract_windows(flux, seq_len=2048, n_windows=3, label=1):
    """Extract windows from a light curve."""
    windows = []

    if len(flux) < seq_len:
        # Pad if too short
        padded = np.zeros(seq_len)
        padded[:len(flux)] = flux
        windows.append(padded)
    else:
        # Extract multiple windows at different positions
        for i in range(n_windows):
            if n_windows == 1:
                start = 0
            else:
                max_start = len(flux) - seq_len
                start = int(i * max_start / (n_windows - 1))

            window = flux[start:start + seq_len]

            # Normalize window
            window = (window - np.mean(window)) / (np.std(window) + 1e-8)

            windows.append(window)

    return windows


def main():
    parser = argparse.ArgumentParser(description='Build windows from Kepler FLTI data')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing Kepler FLTI data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for windows')
    parser.add_argument('--seq_len', type=int, default=2048,
                       help='Window length')
    parser.add_argument('--n_windows', type=int, default=3,
                       help='Windows per light curve')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[scanning] {args.data_dir}")

    # Find all light curve files
    fits_files = glob(os.path.join(args.data_dir, '**', '*.fits'), recursive=True)
    csv_files = glob(os.path.join(args.data_dir, '**', '*.csv'), recursive=True)
    txt_files = glob(os.path.join(args.data_dir, '**', '*.txt'), recursive=True)

    print(f"[found] {len(fits_files)} FITS, {len(csv_files)} CSV, {len(txt_files)} TXT files")

    # Check for injection parameter files
    param_files = glob(os.path.join(args.data_dir, '**', '*param*'), recursive=True)
    param_files += glob(os.path.join(args.data_dir, '**', '*inject*'), recursive=True)
    print(f"[found] {len(param_files)} parameter/injection files")

    all_files = fits_files + csv_files

    if len(all_files) == 0:
        print("[error] No light curve files found!")
        print(f"[hint] Looking in: {args.data_dir}")
        print("[hint] Expected .fits or .csv files")

        # List what's actually there
        print("\n[contents of data_dir]:")
        for item in os.listdir(args.data_dir)[:20]:
            item_path = os.path.join(args.data_dir, item)
            if os.path.isdir(item_path):
                print(f"  [DIR] {item}/")
                # Show contents of subdirectory
                sub_items = os.listdir(item_path)[:5]
                for sub in sub_items:
                    print(f"        {sub}")
                if len(os.listdir(item_path)) > 5:
                    print(f"        ... and {len(os.listdir(item_path)) - 5} more")
            else:
                print(f"  {item}")
        return

    # Process light curves
    all_windows = []
    all_labels = []
    metadata = []

    print(f"\n[processing] {len(all_files)} light curves")

    for file_path in tqdm(all_files, desc="Processing"):
        # Determine file type and load
        if file_path.endswith('.fits'):
            time, flux = load_kepler_lightcurve(file_path)
        else:
            time, flux = load_kepler_csv(file_path)

        if time is None or flux is None:
            continue

        # Extract filename info
        filename = os.path.basename(file_path)

        # Determine label (for injection data, assume planet=1)
        # The FLTI data contains injected transits, so label=1
        label = 1

        # Extract windows
        windows = extract_windows(flux, args.seq_len, args.n_windows, label)

        for i, window in enumerate(windows):
            all_windows.append(window)
            all_labels.append(label)
            metadata.append({
                'filename': filename,
                'window_idx': i,
                'label': label,
                'n_points': len(flux)
            })

    if len(all_windows) == 0:
        print("[error] No windows extracted!")
        return

    # Convert to arrays
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    meta_df = pd.DataFrame(metadata)

    print(f"\n[result] X.shape={X.shape} y.shape={y.shape}")
    print(f"[result] pos={y.sum()} neg={len(y) - y.sum()}")

    # Save
    np.save(os.path.join(args.output_dir, 'X.npy'), X)
    np.save(os.path.join(args.output_dir, 'y.npy'), y)
    meta_df.to_csv(os.path.join(args.output_dir, 'meta.csv'), index=False)

    print(f"\n[saved] {args.output_dir}")
    print(f"  X.npy: {X.nbytes / 1e6:.1f} MB")
    print(f"  y.npy: {y.nbytes / 1e3:.1f} KB")
    print(f"  meta.csv: {len(meta_df)} rows")


if __name__ == '__main__':
    main()
