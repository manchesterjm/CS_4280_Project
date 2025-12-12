"""Quick check of D:\kepler_lightcurves files."""
import os
import pandas as pd

data_dir = r'D:\kepler_lightcurves'
files = os.listdir(data_dir)
print(f"Found {len(files)} files")

first_file = os.path.join(data_dir, files[0])
print(f"\nFirst file: {files[0]}")
df = pd.read_csv(first_file)
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
