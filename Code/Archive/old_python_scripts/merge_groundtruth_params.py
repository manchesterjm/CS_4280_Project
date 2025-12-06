"""
Merge LILITH-4 ground truth parameters into training metadata for better clustering.

Uses actual injected transit parameters (Period, Depth, Duration, Teff) instead of
noisy flux-derived statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
GROUND_TRUTH_DIR = Path(r"E:\lilith4_sector-1_groundtruth\sector-1\ground-truth")
WINDOWS_DIR = Path(r"C:\CS_4280_Project\Code\data\windows_sector1_full")

def parse_planet_data(filepath):
    """Parse planet ground truth file."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 16:
                records.append({
                    'tic_id': int(parts[0]),
                    'planet_num': int(parts[1]),
                    'period': float(parts[2]),
                    'epoch': float(parts[3]),
                    'rp_rstar': float(parts[4]),
                    'rp_earth': float(parts[5]),
                    'impact': float(parts[6]),
                    'a_rstar': float(parts[7]),
                    'duration': float(parts[8]),
                    'depth_ppm': float(parts[9]),
                    'insolation': float(parts[10]),
                    'rstar': float(parts[11]),
                    'mstar': float(parts[12]),
                    'teff': float(parts[13]),
                    'logg': float(parts[14]),
                    'feh': float(parts[15]),
                    'category': 'planet'
                })
    return pd.DataFrame(records)

def parse_eb_data(filepath):
    """Parse eclipsing binary ground truth file."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 14:
                records.append({
                    'tic_id': int(parts[0]),
                    'planet_num': int(parts[1]),
                    'period': float(parts[2]),
                    'epoch': float(parts[3]),
                    'depth_ppm': float(parts[7]),  # Primary depth
                    'teff': float(parts[12]),
                    'logg': float(parts[13]),
                    'rstar': float(parts[10]),
                    'duration': 0.0,  # Not provided for EBs
                    'category': 'eb'
                })
    return pd.DataFrame(records)

def parse_star_data(filepath):
    """Parse stellar variability file - get Teff, logg for stars."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 5:
                records.append({
                    'tic_id': int(parts[0]),
                    'teff': float(parts[3]) if len(parts) > 3 else 5500.0,
                    'logg': float(parts[4]) if len(parts) > 4 else 4.5,
                    'period': 0.0,
                    'depth_ppm': 0.0,
                    'duration': 0.0,
                    'category': 'star'
                })
    return pd.DataFrame(records)

def parse_backeb_data(filepath):
    """Parse background EB file."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 10:
                records.append({
                    'tic_id': int(parts[0]),
                    'period': float(parts[2]) if len(parts) > 2 else 0.0,
                    'depth_ppm': float(parts[7]) if len(parts) > 7 else 0.0,
                    'teff': 5500.0,  # Default
                    'logg': 4.5,
                    'duration': 0.0,
                    'category': 'backeb'
                })
    return pd.DataFrame(records)

def main():
    print("Loading ground truth parameter files...")

    # Parse all ground truth files
    planet_df = parse_planet_data(GROUND_TRUTH_DIR / "tsop301_planet_data.txt")
    print(f"  Planets: {len(planet_df)} records")

    eb_df = parse_eb_data(GROUND_TRUTH_DIR / "tsop301_eb_data.txt")
    print(f"  EBs: {len(eb_df)} records")

    star_df = parse_star_data(GROUND_TRUTH_DIR / "tsop301_star_data.txt")
    print(f"  Stars: {len(star_df)} records")

    backeb_df = parse_backeb_data(GROUND_TRUTH_DIR / "tsop301_backeb_data.txt")
    print(f"  BackEBs: {len(backeb_df)} records")

    # For planets with multiple entries, take the first (strongest signal)
    planet_df = planet_df.sort_values('depth_ppm', ascending=False).drop_duplicates('tic_id', keep='first')

    # Combine all ground truth
    gt_cols = ['tic_id', 'period', 'depth_ppm', 'duration', 'teff', 'logg', 'category']

    all_gt = pd.concat([
        planet_df[gt_cols] if all(c in planet_df.columns for c in gt_cols) else planet_df,
        eb_df[['tic_id', 'period', 'depth_ppm', 'duration', 'teff', 'logg', 'category']] if 'logg' in eb_df.columns else eb_df,
        star_df[['tic_id', 'period', 'depth_ppm', 'duration', 'teff', 'logg', 'category']] if 'logg' in star_df.columns else star_df,
        backeb_df[['tic_id', 'period', 'depth_ppm', 'duration', 'teff', 'category']] if 'teff' in backeb_df.columns else backeb_df
    ], ignore_index=True)

    # Fill missing values
    all_gt['logg'] = all_gt['logg'].fillna(4.5)
    all_gt['duration'] = all_gt['duration'].fillna(0.0)

    print(f"\nTotal ground truth records: {len(all_gt)}")
    print(f"Unique TIC IDs: {all_gt['tic_id'].nunique()}")

    # Process train and test sets
    for split in ['train', 'test']:
        meta_path = WINDOWS_DIR / split / 'meta.csv'
        if not meta_path.exists():
            print(f"Skipping {split} - not found")
            continue

        print(f"\nProcessing {split}...")
        meta_df = pd.read_csv(meta_path)
        original_cols = list(meta_df.columns)

        # Merge ground truth parameters
        meta_df = meta_df.merge(
            all_gt[['tic_id', 'period', 'depth_ppm', 'duration', 'teff', 'logg']],
            on='tic_id',
            how='left'
        )

        # Fill missing values with defaults
        meta_df['period'] = meta_df['period'].fillna(0.0)
        meta_df['depth_ppm'] = meta_df['depth_ppm'].fillna(0.0)
        meta_df['duration'] = meta_df['duration'].fillna(0.0)
        meta_df['teff'] = meta_df['teff'].fillna(5500.0)
        meta_df['logg'] = meta_df['logg'].fillna(4.5)

        # Log transform period for better clustering (avoid log(0))
        meta_df['log_period'] = np.log10(meta_df['period'] + 0.1)

        # Log transform depth for better clustering
        meta_df['log_depth'] = np.log10(meta_df['depth_ppm'] + 1.0)

        # Save updated metadata
        meta_df.to_csv(meta_path, index=False)

        # Stats
        matched = (meta_df['period'] > 0).sum()
        print(f"  Windows: {len(meta_df)}")
        print(f"  Matched with GT: {matched} ({100*matched/len(meta_df):.1f}%)")
        print(f"  New columns: period, depth_ppm, duration, teff, logg, log_period, log_depth")

        # Show sample
        print(f"\n  Sample (planets):")
        planets = meta_df[meta_df['category'] == 'planet'].head(3)
        if len(planets) > 0:
            print(planets[['tic_id', 'category', 'period', 'depth_ppm', 'duration', 'teff']].to_string(index=False))

if __name__ == "__main__":
    main()
    print("\nDone! Metadata updated with ground truth parameters.")
