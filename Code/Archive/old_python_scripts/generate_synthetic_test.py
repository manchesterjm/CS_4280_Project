"""
Generate synthetic light curves to test transit detection.

Creates light curves with known injected transits to verify
the model learned transit patterns, not just memorized training data.

Usage:
    python generate_synthetic_test.py --output_dir data/windows_synthetic_test --n_planets 100 --n_noise 100
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def generate_transit_signal(n_points, period, depth, duration, t0=0, noise_level=0.001):
    """
    Generate a light curve with a transit signal.

    Args:
        n_points: Number of data points
        period: Orbital period (in data point units)
        depth: Transit depth (fractional, e.g., 0.01 = 1%)
        duration: Transit duration (in data point units)
        t0: Phase offset
        noise_level: Gaussian noise standard deviation

    Returns:
        flux: Normalized flux array with transit
    """
    time = np.arange(n_points)
    flux = np.ones(n_points)

    # Add transits at each period
    for transit_center in np.arange(t0, n_points, period):
        transit_center = int(transit_center)
        half_dur = int(duration / 2)

        start = max(0, transit_center - half_dur)
        end = min(n_points, transit_center + half_dur)

        # Box-shaped transit (simple model)
        flux[start:end] -= depth

    # Add Gaussian noise
    flux += np.random.normal(0, noise_level, n_points)

    # Normalize to zero mean, unit variance
    flux = (flux - np.mean(flux)) / (np.std(flux) + 1e-8)

    return flux


def generate_noise_only(n_points, noise_level=0.001, add_variability=True):
    """
    Generate a light curve with only noise (no transit).

    Args:
        n_points: Number of data points
        noise_level: Gaussian noise standard deviation
        add_variability: Add slow stellar variability

    Returns:
        flux: Normalized flux array without transit
    """
    flux = np.ones(n_points)

    # Add Gaussian noise
    flux += np.random.normal(0, noise_level, n_points)

    # Optionally add slow stellar variability (sinusoidal)
    if add_variability:
        period = np.random.uniform(500, 2000)
        amplitude = np.random.uniform(0.001, 0.005)
        phase = np.random.uniform(0, 2 * np.pi)
        time = np.arange(n_points)
        flux += amplitude * np.sin(2 * np.pi * time / period + phase)

    # Normalize
    flux = (flux - np.mean(flux)) / (np.std(flux) + 1e-8)

    return flux


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic test light curves')
    parser.add_argument('--output_dir', type=str, default='data/windows_synthetic_test')
    parser.add_argument('--n_planets', type=int, default=100, help='Number of planet light curves')
    parser.add_argument('--n_noise', type=int, default=100, help='Number of noise-only light curves')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GENERATING SYNTHETIC TEST LIGHT CURVES")
    print("=" * 60)
    print(f"Planets (with transit): {args.n_planets}")
    print(f"Noise only: {args.n_noise}")
    print(f"Sequence length: {args.seq_len}")

    all_windows = []
    all_labels = []
    all_metadata = []

    # Generate planet light curves (with transits)
    print("\n[generating] Planet light curves with transits...")
    for i in tqdm(range(args.n_planets), desc="Planets"):
        # Random transit parameters
        period = np.random.uniform(100, 500)  # Transit every 100-500 points
        depth = np.random.uniform(0.005, 0.03)  # 0.5% to 3% depth
        duration = np.random.uniform(20, 80)  # Transit duration
        t0 = np.random.uniform(0, period)  # Phase offset
        noise = np.random.uniform(0.0005, 0.003)  # Noise level

        flux = generate_transit_signal(
            args.seq_len, period, depth, duration, t0, noise
        )

        # Clip and sanitize
        flux = np.clip(flux, -10, 10)
        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)

        all_windows.append(flux)
        all_labels.append(1)
        all_metadata.append({
            'id': f'synthetic_planet_{i}',
            'label': 1,
            'period': period,
            'depth': depth,
            'duration': duration,
            'noise': noise,
            'source': 'synthetic'
        })

    # Generate noise-only light curves
    print("[generating] Noise-only light curves...")
    for i in tqdm(range(args.n_noise), desc="Noise"):
        noise = np.random.uniform(0.0005, 0.003)
        add_var = np.random.random() > 0.3  # 70% have variability

        flux = generate_noise_only(args.seq_len, noise, add_var)

        # Clip and sanitize
        flux = np.clip(flux, -10, 10)
        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)

        all_windows.append(flux)
        all_labels.append(0)
        all_metadata.append({
            'id': f'synthetic_noise_{i}',
            'label': 0,
            'period': 0,
            'depth': 0,
            'duration': 0,
            'noise': noise,
            'source': 'synthetic'
        })

    # Convert to arrays
    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    meta_df = pd.DataFrame(all_metadata)

    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]
    meta_df = meta_df.iloc[idx].reset_index(drop=True)

    print(f"\n[result] X.shape={X.shape} y.shape={y.shape}")
    print(f"[result] Planets: {y.sum()}, Noise: {len(y) - y.sum()}")

    # Save
    np.save(output_dir / 'X.npy', X)
    np.save(output_dir / 'y.npy', y)
    meta_df.to_csv(output_dir / 'meta.csv', index=False)

    print(f"\n[saved] {output_dir}")
    print(f"  X.npy: {X.nbytes / 1e6:.1f} MB")
    print(f"  y.npy: {y.nbytes / 1e3:.1f} KB")
    print(f"  meta.csv: {len(meta_df)} rows")

    print("\n" + "=" * 60)
    print("DONE - Test with:")
    print(f"  python evaluate_simple.py --model_path runs/combined_model/best.pt --data_dir {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
