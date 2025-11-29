import pandas as pd

meta = pd.read_csv('data/windows_sector1_full/train/meta.csv')
zeros = (meta['std'] == 0).sum()
total = len(meta)
print(f"Windows with zero std: {zeros}/{total} ({100*zeros/total:.1f}%)")
print(f"\nLabel distribution:")
print(meta['label'].value_counts())
print(f"\nCategory distribution:")
print(meta['category'].value_counts())
