"""
Download TESS Light Curves using Lightkurve
Creates test dataset in same format as your existing data

Installation:
    conda install -c conda-forge lightkurve

Usage:
    python download_tess_lightcurves.py --tic_list tic_ids.txt --output_dir "C:\CS_4280_Project\test_dataset_v2\raw"

TIC ID sources:
    - ExoFOP-TESS TOIs: https://exofop.ipac.caltech.edu/tess/
    - NASA Exoplanet Archive: https://exoplanetarchive.ipac.caltech.edu/
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import lightkurve as lk
except ImportError:
    print("ERROR: lightkurve not installed!")
    print("Install with: conda install -c conda-forge lightkurve")
    exit(1)


def download_lightcurve(tic_id, output_dir, mission='TESS', author='SPOC'):
    """
    Download and save a TESS light curve
    
    Args:
        tic_id: TIC identifier (e.g., 100229772)
        output_dir: Where to save the light curve
        mission: 'TESS' or 'Kepler'
        author: 'SPOC' (TESS) or 'Kepler' (Kepler)
    
    Returns:
        dict: Metadata about the downloaded light curve, or None if failed
    """
    
    try:
        # Search for light curve
        search_result = lk.search_lightcurve(
            f'TIC {tic_id}',
            mission=mission,
            author=author
        )
        
        if len(search_result) == 0:
            print(f"  ❌ No data found for TIC {tic_id}")
            return None
        
        print(f"  Found {len(search_result)} sectors")
        
        # Download all sectors
        lc_collection = search_result.download_all()
        
        if lc_collection is None or len(lc_collection) == 0:
            print(f"  ❌ Download failed for TIC {tic_id}")
            return None
        
        # Stitch sectors together
        lc = lc_collection.stitch()
        
        # Remove NaNs
        lc = lc.remove_nans()
        
        # Normalize (SAP flux)
        lc = lc.normalize()
        
        # Remove outliers (3-sigma clip)
        lc = lc.remove_outliers(sigma=3)
        
        # Extract data (convert to regular numpy arrays)
        time = np.asarray(lc.time.value)  # Days
        flux = np.asarray(lc.flux.value)  # Normalized flux
        
        # Save as NPY files (same format as your existing data)
        output_path = Path(output_dir) / f"TIC_{tic_id}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / "time.npy", time)
        np.save(output_path / "flux.npy", flux)
        
        # Create metadata
        metadata = {
            'tic_id': tic_id,
            'n_points': len(time),
            'duration_days': time[-1] - time[0],
            'sectors': len(search_result),
            'mission': mission,
            'author': author
        }
        
        print(f"  ✓ Saved {len(time)} points ({metadata['duration_days']:.1f} days)")
        
        return metadata
        
    except Exception as e:
        print(f"  ❌ Error downloading TIC {tic_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download TESS light curves')
    
    parser.add_argument('--tic_list', type=str, required=True,
                       help='Text file with TIC IDs (one per line)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for light curves')
    parser.add_argument('--mission', type=str, default='TESS',
                       choices=['TESS', 'Kepler'],
                       help='Mission to download from')
    parser.add_argument('--author', type=str, default='SPOC',
                       help='Data author (SPOC for TESS, Kepler for Kepler)')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" TESS/Kepler Light Curve Downloader")
    print("="*70)
    
    # Read TIC IDs
    with open(args.tic_list, 'r') as f:
        tic_ids = [line.strip() for line in f 
                   if line.strip() and not line.strip().startswith('#')]
    
    print(f"\nFound {len(tic_ids)} TIC IDs to download")
    print(f"Mission: {args.mission}")
    print(f"Author: {args.author}")
    print(f"Output: {args.output_dir}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download each light curve
    metadata_list = []
    successful = 0
    failed = 0
    
    for tic_id in tqdm(tic_ids, desc="Downloading"):
        print(f"\nTIC {tic_id}:")
        metadata = download_lightcurve(
            tic_id,
            args.output_dir,
            mission=args.mission,
            author=args.author
        )
        
        if metadata is not None:
            metadata_list.append(metadata)
            successful += 1
        else:
            failed += 1
    
    # Save manifest
    if metadata_list:
        manifest_df = pd.DataFrame(metadata_list)
        manifest_path = Path(args.output_dir).parent / 'manifest.csv'
        manifest_df.to_csv(manifest_path, index=False)
        print(f"\n✓ Saved manifest: {manifest_path}")
    
    # Summary
    print("\n" + "="*70)
    print(" DOWNLOAD COMPLETE")
    print("="*70)
    print(f"\nSuccessful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(tic_ids)}")
    
    if successful > 0:
        print(f"\nLight curves saved to: {args.output_dir}")
        print(f"Manifest saved to: {manifest_path}")


if __name__ == '__main__':
    main()
