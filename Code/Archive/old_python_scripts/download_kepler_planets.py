"""
Download Kepler Confirmed Planet Light Curves for Cross-Mission Testing

This script downloads light curves from Kepler confirmed exoplanet systems
to test if our TESS-trained model generalizes across missions.

Usage:
    python download_kepler_planets.py --output_dir "C:\CS_4280_Project\kepler_test_data\raw" --n_targets 50
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import lightkurve as lk
except ImportError:
    print("ERROR: lightkurve not installed. Install with: pip install lightkurve")
    exit(1)


def get_kepler_confirmed_planets(n_targets=50):
    """
    Get list of confirmed Kepler planets from NASA Exoplanet Archive

    Returns:
        DataFrame with KOI IDs and planet names
    """
    print(f"Fetching confirmed Kepler planets from NASA Exoplanet Archive...")

    # Query NASA Exoplanet Archive for Kepler confirmed planets
    from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

    # Get confirmed planets discovered by Kepler
    planets = NasaExoplanetArchive.query_criteria(
        table="ps",  # Planetary Systems table
        select="pl_name,hostname,sy_kepmag,pl_orbper,pl_trandep,pl_trandur",
        where="disc_facility like 'Kepler%' and tran_flag=1"
    )

    # Convert to pandas
    df = planets.to_pandas()

    # Filter for good targets
    df = df[df['sy_kepmag'] < 15.0]  # Bright enough for good SNR
    df = df[df['pl_orbper'] < 30.0]  # Short periods (observable in one quarter)
    df = df.dropna(subset=['pl_trandep', 'pl_trandur'])  # Must have transit info

    # Sample n_targets
    if len(df) > n_targets:
        df = df.sample(n=n_targets, random_state=42).reset_index(drop=True)

    print(f"Selected {len(df)} Kepler confirmed planet systems")
    print(f"Magnitude range: {df['sy_kepmag'].min():.1f} - {df['sy_kepmag'].max():.1f}")
    print(f"Period range: {df['pl_orbper'].min():.1f} - {df['pl_orbper'].max():.1f} days")

    return df


def download_kepler_lightcurve(hostname, output_dir):
    """
    Download Kepler light curve for a given star

    Args:
        hostname: Star name (e.g., 'Kepler-10')
        output_dir: Directory to save light curve

    Returns:
        Success status (True/False)
    """
    try:
        # Search for light curves
        search = lk.search_lightcurve(hostname, mission='Kepler', cadence='long')

        if len(search) == 0:
            print(f"  No data found for {hostname}")
            return False

        # Download all quarters and stitch
        lc_collection = search.download_all()

        if lc_collection is None or len(lc_collection) == 0:
            print(f"  Download failed for {hostname}")
            return False

        # Stitch quarters together
        lc = lc_collection.stitch()

        # Remove NaNs and outliers
        lc = lc.remove_nans().remove_outliers(sigma=5)

        if len(lc.time) < 1000:  # Need enough data
            print(f"  Insufficient data for {hostname} ({len(lc.time)} points)")
            return False

        # Extract KIC/KOI ID for filename
        target_id = str(search[0].target_name).replace(' ', '_')

        # Save to CSV
        output_path = output_dir / f"{target_id}_lightcurve.csv"
        df = pd.DataFrame({
            'time': lc.time.value,
            'flux': lc.flux.value
        })
        df.to_csv(output_path, index=False)

        print(f"  ✓ Saved {hostname} ({len(df)} points) → {target_id}")
        return True

    except Exception as e:
        print(f"  Error downloading {hostname}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str,
                       default=r'C:\CS_4280_Project\kepler_test_data\raw',
                       help='Output directory for raw light curves')
    parser.add_argument('--n_targets', type=int, default=50,
                       help='Number of Kepler planet systems to download')
    parser.add_argument('--save_list', action='store_true',
                       help='Save list of targets to CSV')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(" DOWNLOADING KEPLER CONFIRMED PLANET LIGHT CURVES")
    print("="*70)
    print()

    # Get confirmed planet list
    planets_df = get_kepler_confirmed_planets(args.n_targets)

    if args.save_list:
        list_path = output_dir.parent / 'kepler_planet_list.csv'
        planets_df.to_csv(list_path, index=False)
        print(f"\n✓ Saved planet list to {list_path}")

    # Download light curves
    print(f"\nDownloading {len(planets_df)} light curves...")
    print("(This may take 10-20 minutes depending on connection)\n")

    success_count = 0
    fail_count = 0

    for idx, row in tqdm(planets_df.iterrows(), total=len(planets_df)):
        hostname = row['hostname']
        success = download_kepler_lightcurve(hostname, output_dir)

        if success:
            success_count += 1
        else:
            fail_count += 1

    print()
    print("="*70)
    print(" DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Process Kepler data: python process_kepler_for_testing.py")
    print("  2. Build test windows: python build_simple_windows.py")
    print("  3. Run inference: python inference_cluster_model.py")
    print("="*70)


if __name__ == '__main__':
    main()
