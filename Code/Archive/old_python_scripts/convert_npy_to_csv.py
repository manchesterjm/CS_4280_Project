"""
Convert processed NPY files to CSV format for build_windows_infer_v2.py
Downsamples long light curves to prevent memory issues

Usage:
    python convert_npy_to_csv.py --input_dir "C:\CS_4280_Project\test_dataset_v2\processed" --output_dir "C:\CS_4280_Project\test_dataset_v2\processed_csv" --max_points 50000
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def downsample_lightcurve(time, flux, max_points=50000):
    """Downsample if light curve is too long"""
    if len(time) <= max_points:
        return time, flux
    
    # Uniform sampling
    indices = np.linspace(0, len(time) - 1, max_points, dtype=int)
    return time[indices], flux[indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory with NPY files (TIC_*/time.npy, flux.npy)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for CSV files')
    parser.add_argument('--max_points', type=int, default=50000,
                       help='Maximum points per light curve (prevents memory issues)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(" CONVERTING NPY TO CSV")
    print("="*70)
    print(f"\nMax points per light curve: {args.max_points}")
    
    # Find all TIC directories
    tic_dirs = sorted([d for d in input_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('TIC_')])
    
    print(f"Found {len(tic_dirs)} light curves\n")
    
    converted = 0
    for tic_dir in tqdm(tic_dirs, desc="Converting"):
        tic_id = tic_dir.name.replace('TIC_', '')
        
        try:
            # Load NPY files
            time = np.load(tic_dir / 'time.npy')
            flux = np.load(tic_dir / 'flux.npy')
            
            original_len = len(time)
            
            # Downsample if needed
            time, flux = downsample_lightcurve(time, flux, args.max_points)
            
            if len(time) < original_len:
                print(f"\n  TIC {tic_id}: {original_len} → {len(time)} points")
            
            # Create DataFrame
            df = pd.DataFrame({
                'time': time,
                'flux': flux
            })
            
            # Save as CSV
            output_file = output_dir / f"{tic_id}_lightcurve.csv"
            df.to_csv(output_file, index=False)
            
            converted += 1
            
        except Exception as e:
            print(f"\n  ❌ Error converting TIC {tic_id}: {e}")
            continue
    
    print(f"\n✓ Converted {converted}/{len(tic_dirs)} light curves")
    print(f"✓ CSV files saved to: {output_dir}")
    
    print("\n" + "="*70)
    print(" NEXT STEP")
    print("="*70)
    print(f"\nBuild windows:")
    print(f'python build_windows_infer_v2.py --processed_dir "{output_dir}" --out_dir "C:\\CS_4280_Project\\Code\\data\\windows_test" --n_jobs -1')
    print()


if __name__ == '__main__':
    main()
