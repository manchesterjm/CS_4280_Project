"""
Process Downloaded TESS Light Curves for Testing
Prepares data in same format as training data for inference

Usage:
    python process_tess_for_testing.py --raw_dir "C:\CS_4280_Project\test_dataset_v2\raw" --output_dir "C:\CS_4280_Project\test_dataset_v2\processed"
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.signal import medfilt
import warnings
warnings.filterwarnings('ignore')


def process_lightcurve(time, flux):
    """
    Process light curve: flatten, normalize, remove outliers
    Same processing as your training data
    """
    
    # Remove NaNs
    mask = ~(np.isnan(time) | np.isnan(flux))
    time = time[mask]
    flux = flux[mask]
    
    if len(time) < 100:
        return None, None
    
    # Flatten using median filter (remove long-term trends)
    window_length = min(101, len(flux) // 10 * 2 + 1)  # Must be odd
    trend = medfilt(flux, kernel_size=window_length)
    flux_flattened = flux / trend
    
    # Normalize to mean=1, then convert to relative flux
    flux_normalized = flux_flattened / np.median(flux_flattened)
    
    # Remove extreme outliers (beyond 5 sigma)
    median = np.median(flux_normalized)
    mad = np.median(np.abs(flux_normalized - median))
    std_approx = 1.4826 * mad
    
    outlier_mask = np.abs(flux_normalized - median) < 5 * std_approx
    time = time[outlier_mask]
    flux_normalized = flux_normalized[outlier_mask]
    
    if len(time) < 100:
        return None, None
    
    return time, flux_normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True,
                       help='Directory with raw downloaded light curves')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed light curves')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" PROCESSING TESS DATA FOR TESTING")
    print("="*70)
    
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all TIC directories
    tic_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith('TIC_')]
    
    print(f"\nFound {len(tic_dirs)} light curves to process\n")
    
    processed_count = 0
    failed_count = 0
    metadata_list = []
    
    for tic_dir in tqdm(tic_dirs, desc="Processing"):
        tic_id = tic_dir.name.replace('TIC_', '')
        
        try:
            # Load raw data
            time = np.load(tic_dir / 'time.npy')
            flux = np.load(tic_dir / 'flux.npy')
            
            print(f"\n{tic_dir.name}:")
            print(f"  Raw: {len(time)} points")
            
            # Process
            time_proc, flux_proc = process_lightcurve(time, flux)
            
            if time_proc is None:
                print(f"  ❌ Processing failed (too few points)")
                failed_count += 1
                continue
            
            print(f"  Processed: {len(time_proc)} points")
            
            # Save processed data
            output_tic_dir = output_dir / f"TIC_{tic_id}"
            output_tic_dir.mkdir(exist_ok=True)
            
            np.save(output_tic_dir / 'time.npy', time_proc)
            np.save(output_tic_dir / 'flux.npy', flux_proc)
            
            # Metadata
            metadata_list.append({
                'tic_id': int(tic_id),
                'n_points': len(time_proc),
                'duration_days': float(time_proc[-1] - time_proc[0]),
                'has_planet': 0,  # Unknown - you'll need to label these manually
                'notes': 'Downloaded from TESS'
            })
            
            processed_count += 1
            print(f"  ✓ Saved")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed_count += 1
            continue
    
    # Save manifest
    if metadata_list:
        manifest_df = pd.DataFrame(metadata_list)
        manifest_path = output_dir.parent / 'manifest.csv'
        manifest_df.to_csv(manifest_path, index=False)
        print(f"\n✓ Saved manifest: {manifest_path}")
    
    # Summary
    print("\n" + "="*70)
    print(" PROCESSING COMPLETE")
    print("="*70)
    print(f"\nProcessed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total: {len(tic_dirs)}")
    
    if processed_count > 0:
        print(f"\nProcessed light curves: {output_dir}")
        print(f"Manifest: {manifest_path}")
        
        print("\n" + "="*70)
        print(" NEXT STEPS")
        print("="*70)
        print("\n1. Label the stars (which have confirmed planets):")
        print(f"   - Edit {manifest_path}")
        print("   - Set has_planet=1 for planet hosts")
        print("   - Check ExoFOP-TESS: https://exofop.ipac.caltech.edu/tess/")
        
        print("\n2. Build inference windows:")
        print(f"   python build_windows_infer_v2.py --manifest {manifest_path}")
        
        print("\n3. Run inference with your trained model:")
        print("   python inference_rnn.py --model runs/bilstm_cluster/best.pt")
        print()


if __name__ == '__main__':
    main()
