r"""
Build Test Windows for Real Planet Data (Planet_LightCurve_Data)

Extracts windows with BLS features for clustering-based inference.

Usage:
    python build_planet_test_windows.py --data_dir "C:\CS_4280_Project\Planet_LightCurve_Data\processed" --output_dir "C:\CS_4280_Project\Code\data\windows_planet_test"
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from astropy.timeseries import BoxLeastSquares


def robust_detrend(time, flux):
    """Remove polynomial trend using robust fit"""
    mask = np.isfinite(flux)
    if mask.sum() < 10:
        return flux

    try:
        # Fit 2nd order polynomial
        coeffs = np.polyfit(time[mask], flux[mask], 2)
        trend = np.polyval(coeffs, time)
        detrended = flux - trend + np.median(flux[mask])
        return detrended
    except:
        return flux


def normalize(flux):
    """Normalize to median=1, then z-score"""
    mask = np.isfinite(flux)
    if mask.sum() == 0:
        return np.zeros_like(flux, dtype=np.float32)

    # Median normalization
    med = np.median(flux[mask])
    flux_norm = flux / med if med > 0 else flux

    # Z-score
    mean = np.mean(flux_norm[mask])
    std = np.std(flux_norm[mask]) + 1e-8
    return ((flux_norm - mean) / std).astype(np.float32)


def run_bls(time, flux, min_period=0.5, max_period=20.0, duration_grid_step=0.05):
    """Run Box Least Squares period finding"""
    mask = np.isfinite(time) & np.isfinite(flux)
    if mask.sum() < 100:
        return None, None

    try:
        # Create BLS model
        model = BoxLeastSquares(time[mask], flux[mask])

        # Compute periodogram
        periods = np.linspace(min_period, max_period, 10000)
        periodogram = model.autopower(
            duration=np.arange(0.05, 0.5, duration_grid_step),
            frequency_factor=5.0
        )

        # Get best period
        best_idx = np.argmax(periodogram.power)
        period = periodogram.period[best_idx]
        t0 = periodogram.transit_time[best_idx]
        depth = periodogram.depth[best_idx]
        duration = periodogram.duration[best_idx]
        power = periodogram.power[best_idx]

        stats = {
            'period': float(period),
            't0': float(t0),
            'depth': float(depth),
            'duration': float(duration),
            'bls_power': float(power)
        }

        return stats, periodogram

    except Exception as e:
        print(f"    BLS failed: {e}")
        return None, None


def phase_fold(time, flux, period, t0):
    """Phase fold light curve"""
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1  # Center at phase 0
    return phase


def extract_window_at_phase(phase, flux, target_phase, window_size=2048, phase_width=0.15):
    """Extract window centered at target phase"""
    # Find points near target phase
    mask = np.abs(phase - target_phase) < phase_width
    if mask.sum() < window_size // 2:
        return None

    # Sort by phase
    idx = np.argsort(phase[mask])
    phase_sorted = phase[mask][idx]
    flux_sorted = flux[mask][idx]

    # Sample uniformly in phase space
    if len(flux_sorted) >= window_size:
        indices = np.linspace(0, len(flux_sorted) - 1, window_size, dtype=int)
        window = flux_sorted[indices]
    else:
        # Pad if not enough points
        pad_len = window_size - len(flux_sorted)
        window = np.pad(flux_sorted, (0, pad_len), mode='edge')

    return window


def process_light_curve(csv_file, window_size=2048):
    """Process a single light curve and extract windows"""
    try:
        # Load data
        df = pd.read_csv(csv_file)
        time = df['time'].values
        flux = df['flux'].values

        # Basic cleaning
        mask = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
        if mask.sum() < 100:
            return [], []

        time = time[mask]
        flux = flux[mask]

        # Detrend
        flux = robust_detrend(time, flux)

        # Run BLS
        bls_stats, periodogram = run_bls(time, flux)
        if bls_stats is None:
            print(f"    No BLS solution found")
            return [], []

        # Phase fold
        period = bls_stats['period']
        t0 = bls_stats['t0']
        phase = phase_fold(time, flux, period, t0)

        # Extract TIC ID
        tic_id = csv_file.stem.replace('_lightcurve', '')

        # Extract windows at different phases
        windows = []
        meta_list = []

        # Window 1: Transit center (phase 0)
        window = extract_window_at_phase(phase, flux, 0.0, window_size)
        if window is not None:
            windows.append(normalize(window))
            meta_list.append({
                'tic_id': tic_id,
                'period': bls_stats['period'],
                'duration': bls_stats['duration'],
                'depth': bls_stats['depth'],
                'bls_power': bls_stats['bls_power'],
                't0': bls_stats['t0'],
                'phase_center': 0.0
            })

        # Window 2: Out of transit (phase 0.25)
        window = extract_window_at_phase(phase, flux, 0.25, window_size)
        if window is not None:
            windows.append(normalize(window))
            meta_list.append({
                'tic_id': tic_id,
                'period': bls_stats['period'],
                'duration': bls_stats['duration'],
                'depth': bls_stats['depth'],
                'bls_power': bls_stats['bls_power'],
                't0': bls_stats['t0'],
                'phase_center': 0.25
            })

        # Window 3: Out of transit (phase -0.25)
        window = extract_window_at_phase(phase, flux, -0.25, window_size)
        if window is not None:
            windows.append(normalize(window))
            meta_list.append({
                'tic_id': tic_id,
                'period': bls_stats['period'],
                'duration': bls_stats['duration'],
                'depth': bls_stats['depth'],
                'bls_power': bls_stats['bls_power'],
                't0': bls_stats['t0'],
                'phase_center': -0.25
            })

        return windows, meta_list

    except Exception as e:
        print(f"    Error: {e}")
        return [], []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                       default=r'C:\CS_4280_Project\Planet_LightCurve_Data\processed',
                       help='Directory with planet light curves')
    parser.add_argument('--output_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\data\windows_planet_test',
                       help='Output directory')
    parser.add_argument('--window_size', type=int, default=2048,
                       help='Window size (must match training)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(" BUILDING TEST WINDOWS FOR REAL PLANET DATA")
    print("="*70)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Window size: {args.window_size}\n")

    # Find CSV files
    csv_files = sorted(data_dir.glob("*_lightcurve.csv"))

    if not csv_files:
        print(f"\n❌ No CSV files found in {data_dir}")
        return

    print(f"Found {len(csv_files)} light curves\n")

    all_X = []
    all_meta = []

    for csv_file in tqdm(csv_files, desc="Processing"):
        print(f"\n  {csv_file.stem}")
        windows, meta_list = process_light_curve(csv_file, args.window_size)

        if windows:
            all_X.extend(windows)
            all_meta.extend(meta_list)
            print(f"    ✓ {len(windows)} windows extracted")
        else:
            print(f"    ✗ Failed to extract windows")

    if not all_X:
        print("\n❌ No windows extracted!")
        return

    # Convert to numpy arrays
    X = np.array(all_X, dtype=np.float32)
    meta_df = pd.DataFrame(all_meta)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total windows: {len(X)}")
    print(f"Unique stars: {meta_df['tic_id'].nunique()}")
    print(f"Window shape: {X.shape}")
    print(f"{'='*70}\n")

    # Save
    np.save(output_dir / 'X.npy', X)
    meta_df.to_csv(output_dir / 'meta.csv', index=False)

    print(f"✓ Saved to {output_dir}")
    print(f"  - X.npy: {X.shape}")
    print(f"  - meta.csv: {len(meta_df)} rows")

    # Print feature statistics
    print(f"\nFeature Statistics:")
    print(meta_df[['period', 'duration', 'depth', 'bls_power']].describe())


if __name__ == '__main__':
    main()
