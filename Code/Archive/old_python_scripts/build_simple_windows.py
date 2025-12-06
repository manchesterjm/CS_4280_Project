"""
Simple sliding window builder for testing (no BLS, no memory issues)
Just creates windows from the light curves without phase folding

Usage:
    python build_simple_windows.py --data_dir "C:\CS_4280_Project\test_dataset_v2\processed_csv" --output_dir "C:\CS_4280_Project\Code\data\windows_test" --window_size 2048 --stride 1024
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def zscore(x):
    """Normalize to mean=0, std=1"""
    m = np.isfinite(x)
    if m.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)
    med = np.nanmedian(x[m])
    mad = np.nanmedian(np.abs(x[m] - med)) + 1e-8
    return ((x - med) / (1.4826 * mad)).astype(np.float32)


def extract_windows(flux, window_size=2048, stride=1024):
    """Extract sliding windows"""
    n = len(flux)
    
    if n < window_size:
        # Pad if too short
        pad_len = window_size - n
        flux_padded = np.pad(flux, (0, pad_len), mode='edge')
        return [flux_padded]
    
    windows = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(flux[start:end])
    
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--window_size', type=int, default=2048,
                       help='Window size (must match training)')
    parser.add_argument('--stride', type=int, default=1024,
                       help='Stride between windows')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print(" BUILDING SIMPLE SLIDING WINDOWS")
    print("="*70)
    print(f"\nWindow size: {args.window_size}")
    print(f"Stride: {args.stride}")
    
    # Find CSV files
    csv_files = sorted(data_dir.glob("*_lightcurve.csv"))
    
    if not csv_files:
        print(f"\n❌ No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} light curves\n")
    
    all_X = []
    all_meta = []
    
    for csv_file in tqdm(csv_files, desc="Processing"):
        try:
            # Load light curve
            df = pd.read_csv(csv_file)
            flux = df['flux'].values
            
            # Extract TIC ID
            tic_id = csv_file.stem.replace('_lightcurve', '')
            
            # Extract windows
            windows = extract_windows(flux, args.window_size, args.stride)
            
            print(f"  {tic_id}: {len(windows)} windows")
            
            # Normalize each window and add to dataset
            for i, window in enumerate(windows):
                window_norm = zscore(window)
                all_X.append(window_norm)
                
                all_meta.append({
                    'tic_id': tic_id,
                    'window_idx': i,
                    'n_windows': len(windows)
                })
        
        except Exception as e:
            print(f"  ❌ Error processing {csv_file.name}: {e}")
            continue
    
    if not all_X:
        print("\n❌ No windows created")
        return
    
    # Stack into arrays
    X = np.stack(all_X)
    meta_df = pd.DataFrame(all_meta)
    
    # Save
    np.save(output_dir / 'X.npy', X)
    meta_df.to_csv(output_dir / 'meta.csv', index=False)
    
    print(f"\n✓ Created {len(X)} windows")
    print(f"✓ Saved to {output_dir}")
    print(f"\n  X.npy: {X.shape}")
    print(f"  meta.csv: {len(meta_df)} rows")
    
    print("\n" + "="*70)
    print(" NEXT STEP")
    print("="*70)
    print("\nRun inference:")
    print('python inference_rnn.py --model_path "C:\\CS_4280_Project\\Code\\runs\\bilstm_cluster\\best.pt" --windows_dir "C:\\CS_4280_Project\\Code\\data\\windows_test" --output_dir "C:\\CS_4280_Project\\Code\\reports\\test_results"')
    print()


if __name__ == '__main__':
    main()
