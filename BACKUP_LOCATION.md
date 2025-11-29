# Data Backup Location

**Date:** November 28, 2025

Large data files have been moved to `E:\CS_4280_Project_Backup` to free up space on C: drive.

## Backup Location: `E:\CS_4280_Project_Backup\`

### Moved Folders

| Original Location | Backup Location | Contents |
|-------------------|-----------------|----------|
| `C:\CS_4280_Project\Code\data\` | `E:\CS_4280_Project_Backup\Code\data\` | Training windows (17 subdirs with .npy files) |
| `C:\CS_4280_Project\Code\runs\` | `E:\CS_4280_Project_Backup\Code\runs\` | Model checkpoints (19 subdirs with .pt files) |
| `C:\CS_4280_Project\runs\` | `E:\CS_4280_Project_Backup\runs_toplevel\` | Top-level runs (sector1_experiments) |
| `C:\CS_4280_Project\kepler_test_data\` | `E:\CS_4280_Project_Backup\kepler_test_data\` | Kepler light curve test data |
| `C:\CS_4280_Project\Planet_LightCurve_Data\` | `E:\CS_4280_Project_Backup\Planet_LightCurve_Data\` | 100 confirmed exoplanet light curves |
| `C:\CS_4280_Project\synthetic_balanced_dataset\` | `E:\CS_4280_Project_Backup\synthetic_balanced_dataset\` | Synthetic balanced training data |
| `C:\CS_4280_Project\synthetic_dataset_400\` | `E:\CS_4280_Project_Backup\synthetic_dataset_400\` | 400 synthetic light curves |
| `C:\CS_4280_Project\test_dataset\` | `E:\CS_4280_Project_Backup\test_dataset\` | Test dataset (simulated) |
| `C:\CS_4280_Project\test_dataset_v2\` | `E:\CS_4280_Project_Backup\test_dataset_v2\` | Test dataset v2 (TESS downloads) |

## Data Directory Contents (Code/data)

- `windows_sector1_full/` - TESS Sector 1 ground truth (26K train + 6K test windows)
- `windows_train/` - Original training windows (655 windows)
- `windows_kepler/` - Kepler confirmed planets
- `windows_combined/` - TESS + Kepler combined dataset
- `windows_smote_true/` - SMOTE-balanced dataset
- `windows_balanced/` - Balanced sampling dataset
- `windows_hybrid_75/`, `windows_hybrid_90/` - Hybrid real+synthetic mixes
- `windows_synthetic_test/` - Synthetic test set
- `windows_planet_test/` - 100 confirmed planet test windows

## Model Checkpoints (Code/runs)

Key models:
- `bilstm_cluster_optimized/` - Optuna-optimized model (AUC 0.7572)
- `bilstm_cluster_sector1/` - Sector 1 trained model
- `combined_model/` - TESS+Kepler combined (AUC 0.9111)
- `bilstm_cluster_smote_true/` - SMOTE model (AUC 0.8175)

## To Restore

To use the data/runs again, either:

1. **Copy back to C:** (if space allows)
   ```powershell
   Copy-Item -Path "E:\CS_4280_Project_Backup\Code\data" -Destination "C:\CS_4280_Project\Code\" -Recurse
   Copy-Item -Path "E:\CS_4280_Project_Backup\Code\runs" -Destination "C:\CS_4280_Project\Code\" -Recurse
   ```

2. **Create symbolic links** (recommended - keeps data on E:)
   ```powershell
   # Run as Administrator
   New-Item -ItemType SymbolicLink -Path "C:\CS_4280_Project\Code\data" -Target "E:\CS_4280_Project_Backup\Code\data"
   New-Item -ItemType SymbolicLink -Path "C:\CS_4280_Project\Code\runs" -Target "E:\CS_4280_Project_Backup\Code\runs"
   ```

3. **Update script paths** to point directly to E: drive

## Important Notes

- The TESS Sector 1 ground truth source data is still at: `E:\lilith4_sector-1_groundtruth\sector-1\ground-truth`
- The backup on E: drive contains processed/derived data (can be regenerated from source)
- Model checkpoints in runs/ contain trained weights (cannot be easily regenerated - keep these!)
