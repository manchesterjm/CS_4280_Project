"""Quick script to check Kepler data format."""
import pandas as pd
import os

raw_dir = r'D:\CS_4280_Project_Backup\kepler_test_data\raw'
files = os.listdir(raw_dir)
print(f"Found {len(files)} Kepler files")

first_file = os.path.join(raw_dir, files[0])
print(f"\nFirst file: {files[0]}")
df = pd.read_csv(first_file)
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
