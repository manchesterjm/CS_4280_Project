# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

We are software engineers with 40 years of experience.  We are professionals and strive to write the best code possible
Even though this is a project for college, I am 50 years old and I know what I am doing.  What we present needs to reflect that
All code will be written with these in mind
  - SOFA principles of coding
    -- Short: Functions should be short and do one thing. This makes them easier to read, test, and reuse.
    -- One Thing: Each function should have a single, clear responsibility. This aligns with the Single Responsibility Principle from SOLID.
    -- Few Arguments: Functions should have a minimal number of arguments. Functions with too many arguments can indicate a design issue and become difficult to use.
    -- Abstraction level consistency: All functions at a given level of the system should be at the same level of abstraction. This helps maintain a clear structure and prevents mixing high-level logic with low-level implementation details
  - If we find a problem with the code, it is not a matter for later, we should fix it now
    -- example: if we run pylint and there are issues raised, we fix them regardless of criticality
  

## Project Overview

This is an exoplanet detection project using deep learning (BiLSTM with K-means clustering) to identify planetary transits in stellar light curve data from NASA's TESS/Kepler missions. The project has achieved **AUC 0.7572** (+9.0% improvement) after Optuna hyperparameter optimization, up from the baseline AUC 0.6947. Successfully tested on 100 confirmed exoplanet systems with 16/300 windows correctly identified as planet candidates.

**UPDATE November 11, 2025**: Balanced synthetic approach **FAILED** due to domain shift. Pure synthetic training achieved AUC 1.0 in training but AUC 0.45 (worse than random!) on real TESS data. Root cause: Synthetic transit depth 8× shallower than real data.

**UPDATE November 14, 2025 (5:28 AM)**: **MAJOR PIVOT** - Transitioning to TESS Sector 1 Ground Truth Dataset
- Team alignment: All partners using same Sector 1 ground truth data
- Dataset: 13,541 light curves → **40,623 training windows** (62× larger!)
- Categories: 3,146 planets + 8,624 stars + 900 EBs + 871 BackEBs
- Class balance: 23.2% positive (9,438 planets vs 31,185 non-planets)
- Location: `D:\CS_4280_Project_Backup\sector-1\ground-truth`
- See `PROGRESS_LOG_NOV_14_2025.md` for full details

**UPDATE November 27, 2025**: **DATASET BUILT & TRAINING IN PROGRESS**
- Dataset successfully built: 26,472 training + 6,579 test windows (80/20 split)
- Training running in background with corrected hyperparameters
- **pos_weight corrected**: 7.41 (was 3.367, now based on actual ratio 23325/3147)
- **Clustering fix applied**: Percentile clipping to handle outliers (for future runs)
- **CPU limits added**: Prevents system crash during parallel processing
- See `PROGRESS_LOG_NOV_27_2025.md` for full session details

**UPDATE November 29, 2025**: **BATCH SIZE BENCHMARK COMPLETED**
- Tested batch sizes: 8, 16, 32, 64, 128
- **Optimal batch size: 64** (3.14 min/epoch) - 7.8× faster than batch 128
- Batch 128 is slowest (24.46 min/epoch) due to GPU memory bottleneck
- See `Code/BATCH_SIZE_BENCHMARK_RESULTS.md` and `PROGRESS_LOG_NOV_29_2025.md`

**UPDATE December 2, 2025 (Morning)**: **NEW SYSTEM ARRIVING**
- **Hardware**: iBuyPower RDY Y70 B01
  - Intel i9-14900KF (24 cores)
  - **RTX 5070 Ti (16 GB VRAM)** - 2× previous VRAM
  - 32 GB DDR5-6000
  - 2 TB NVMe + **Crucial T710 4 TB Gen5 NVMe** (~12,400 MB/s)
- **Expected improvements**: 2-3× faster training, batch size 128-192 optimal
- **Data layout**: Code on C:, Data on D: (T710)
- **Priority**: Run Optuna on Sector 1 dataset, then final training
- **Deadline**: Presentations Dec 9-11, Final submission Dec 18
- See `NEXT_SESSION_QUICKSTART.md` for full setup guide

**UPDATE December 2, 2025 (Afternoon)**: **FINAL PAPER PREPARATION**
- **Code README created**: `Code/README.md` with full run instructions for TA
- **RNN Related Work condensed**: 6,700 → 1,036 words (85% reduction per TA feedback)
- **MiKTeX installed**: Can now compile LaTeX papers locally
- **Paper compiled**: 18 pages (target 15-18 for team of 3)
- **RNN Methodology updating**: Converting from 655-window to Sector 1 dataset
- **Working paper**: `term_project_files/Merged_Proposal_CONDENSED_RNN_12.2.2025.tex`
- **See**: `PROGRESS_LOG_DEC_2_2025.md` for detailed session log

**Environment**: Windows 11, NVIDIA GeForce RTX 5070 Ti (new) / RTX 3060 Ti (old), conda environment `exo-lstm-gpu`
**Current Status**: Updating RNN Methodology section in paper, waiting for new PC delivery

## Key Commands

### Setup
```powershell
conda activate exo-lstm-gpu
cd D:\CS_4280_Project\Code
```

### TESS Sector 1 Ground Truth Dataset (CURRENT PRIORITY - November 14, 2025)

**Dataset Location**: `D:\CS_4280_Project_Backup\sector-1\ground-truth`

#### Step 1: Build Sector 1 Training Windows
```powershell
# Process ground truth data into training windows with statistical features
python build_windows_from_groundtruth.py `
  --data_dir "D:\CS_4280_Project_Backup\sector-1\ground-truth" `
  --output_dir "data\windows_sector1_full" `
  --seq_len 2048 `
  --n_windows 3 `
  --seed 42

# Or use batch script
.\build_sector1_dataset.bat
```

**Output**:
- 40,623 training windows (3 per light curve × 13,541 files)
- Class distribution: 9,438 planets (23.2%) vs 31,185 non-planets (76.8%)
- Metadata with statistical features for clustering: mean, std, var, skew, range, median, mad, peak_to_peak
- Processing time: ~5-10 minutes

**Verify statistical features**:
```python
import pandas as pd
meta = pd.read_csv('data/windows_sector1_full/meta.csv')
print(meta.columns)  # Should include: mean, std, var, skew, range, median, mad, peak_to_peak
```

#### Step 2: Train on Sector 1 Dataset
```powershell
python train_bilstm_cluster.py `
  --windows_dir "D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train" `
  --n_clusters 5 `
  --epochs 60 `
  --batch_size 64 `
  --lr 0.0001 `
  --hidden 256 `
  --layers 4 `
  --dropout 0.311 `
  --save_dir "runs\sector1_batch64" `
  --amp_dtype fp16 `
  --pos_weight 7.41 `
  --num_workers 0 `
  --seed 42
```

**IMPORTANT**:
- Use `data\windows_sector1_full\train` (not just `data\windows_sector1_full`) - dataset has train/test split
- pos_weight should be **7.41** (calculated from 23325/3147 = actual class ratio)
- **Use batch_size 64** (optimal) - NOT 128 (too slow due to GPU memory)
- **Use lr 0.0001** (stable) - NOT 0.000225 (caused NaN crash)

**Expected Training Time**: ~3.14 hours (60 epochs × 3.14 min/epoch with batch size 64)
**Expected Performance**: AUC 0.65-0.75 (target: >0.70)

### Batch Size Benchmark Results (November 29, 2025)

| Batch Size | Batches/Epoch | Time/Batch | Est. Epoch Time |
|------------|---------------|------------|-----------------|
| 8 | 3309 | 0.356 sec | 19.61 min |
| 16 | 1655 | 0.304 sec | 8.38 min |
| 32 | 828 | 0.362 sec | 5.00 min |
| **64** | **414** | **0.455 sec** | **3.14 min** |
| 128 | 207 | 7.089 sec | 24.46 min |

**Key Finding**: Batch size 64 is optimal. Batch size 128 causes GPU memory pressure (7 sec/batch vs 0.455 sec).

### GPU-Specific Settings (IMPORTANT)

For Optuna optimization, edit `optuna_optimize.py` line 214:

```python
# RTX 3060 Ti (8 GB) - Current:
HIGH_VRAM_GPU = False

# RTX 5070 Ti (16 GB) - New system (Dec 2, 2025):
HIGH_VRAM_GPU = True
```

| Setting | 8 GB VRAM (3060 Ti) | 16 GB VRAM (5070 Ti) |
|---------|---------------------|----------------------|
| hidden_size | [128, 256] | [128, 256, 512] |
| batch_size | [32, 64, 128] | [64, 128, 192, 256] |

**Key Improvements**:
- 62× more training data than previous approach (655 → 40,623 windows)
- Real TESS ground truth data (no synthetic domain shift)
- Better class diversity (4 categories: planets, stars, EBs, BackEBs)
- Statistical features prevent data leakage in clustering

---

### Training the BiLSTM+Clustering Model (Legacy - 655 Windows)
```powershell
python train_bilstm_cluster.py `
  --windows_dir "D:\CS_4280_Project\Code\data\windows_train" `
  --n_clusters 5 `
  --epochs 80 `
  --batch_size 64 `
  --lr 1e-4 `
  --hidden 256 `
  --layers 3 `
  --dropout 0.4 `
  --save_dir "D:\CS_4280_Project\Code\runs\bilstm_cluster" `
  --amp_dtype fp16 `
  --pos_weight 3.367 `
  --num_workers 0
```

**Critical**: `--num_workers 0` must be used on Windows to avoid multiprocessing crashes

### Generating Balanced Synthetic Dataset (Recommended Approach)
```powershell
# 1. Generate 400 light curves (200 planets + 200 non-planets)
python generate_synthetic_dataset.py `
  --output_dir "D:\CS_4280_Project\synthetic_dataset_400" `
  --n_planets 200 `
  --n_non_planets 200 `
  --noise_ppm 200 `
  --seed 42

# 2. Build training windows from synthetic data
python build_windows_from_synthetic.py `
  --data_dir "D:\CS_4280_Project\synthetic_dataset_400" `
  --output_dir "D:\CS_4280_Project\Code\data\windows_train_400" `
  --seq_len 2048 `
  --n_windows 3 `
  --seed 42
```

**Output**:
- 400 light curves (50/50 balance)
- ~1,500 training windows (30% positive rate)
- Transit detection via scipy peak finding
- TESS-realistic noise (100-1000 ppm)

**Non-planet types**: Stellar flares, eclipsing binaries, pure noise, background events

**IMPORTANT**: Pure synthetic training failed (AUC 0.45 on real data). Use hybrid approach instead.

### Hybrid Training (Real + Synthetic Mix) - RECOMMENDED

```powershell
# 1. Build hybrid dataset (90% real, 10% synthetic)
python build_hybrid_dataset.py `
  --real_dir "D:\CS_4280_Project\Code\data\windows_train" `
  --synthetic_dir "D:\CS_4280_Project\Code\data\windows_train_400" `
  --mix_ratio 0.90 `
  --output_dir "D:\CS_4280_Project\Code\data\windows_hybrid_90"

# 2. Train on hybrid dataset
python train_bilstm_cluster.py `
  --windows_dir "D:\CS_4280_Project\Code\data\windows_hybrid_90" `
  --n_clusters 5 `
  --epochs 80 `
  --batch_size 128 `
  --lr 0.000225 `
  --hidden 256 `
  --layers 4 `
  --dropout 0.311 `
  --save_dir "D:\CS_4280_Project\Code\runs\bilstm_cluster_hybrid_90" `
  --amp_dtype fp16 `
  --pos_weight 3.0 `
  --num_workers 0

# 3. Or use the batch script to train both 90% and 75% models
.\train_hybrid_models.bat
```

**Why Hybrid?**
- Real data provides domain fidelity (prevents domain shift)
- Synthetic data improves class balance (22.9% → 24.2% positive)
- More training examples (655 → 727 windows for 90% mix)

**Expected Performance**:
- Hybrid 90: AUC 0.79-0.82 (matches or beats pure real)
- Hybrid 75: AUC 0.75-0.80 (more synthetic noise)

### Building Training Windows (Legacy Approach)
```powershell
python build_windows_parallel_v6.py `
  --processed_dir "D:\CS_4280_Project\test_dataset\simulated_dataset\processed" `
  --out_dir "D:\CS_4280_Project\Code\data\windows_train" `
  --seq_len 2048 `
  --neg_per_pos 5 `
  --n_jobs -1 `
  --seed 42 `
  --manifest "D:\CS_4280_Project\test_dataset\simulated_dataset\manifest.csv"
```

### Testing on New TESS Data (Complete Pipeline)
```powershell
# 1. Download TESS light curves
python download_tess_lightcurves.py --tic_list sample_tic_ids.txt --output_dir "D:\CS_4280_Project\test_dataset_v2\raw"

# 2. Process downloaded data
python process_tess_for_testing.py --raw_dir "D:\CS_4280_Project\test_dataset_v2\raw" --output_dir "D:\CS_4280_Project\test_dataset_v2\processed"

# 3. Convert to CSV format
python convert_npy_to_csv.py --input_dir "D:\CS_4280_Project\test_dataset_v2\processed" --output_dir "D:\CS_4280_Project\test_dataset_v2\processed_csv" --max_points 50000

# 4. Build test windows
python build_simple_windows.py --data_dir "D:\CS_4280_Project\test_dataset_v2\processed_csv" --output_dir "D:\CS_4280_Project\Code\data\windows_test"

# 5. Run inference
python inference_cluster_model.py --model_path "D:\CS_4280_Project\Code\runs\bilstm_cluster\best.pt" --windows_dir "D:\CS_4280_Project\Code\data\windows_test" --output_file "D:\CS_4280_Project\Code\reports\test_predictions.csv"
```

## Architecture Overview

### Data Pipeline
1. **Raw light curves** → CSV files with `time` and `flux` columns
2. **Window builder** (`build_windows_parallel_v6.py`) → Extracts 2048-point sliding windows
   - Uses Box Least Squares (BLS) to detect periodicity
   - Creates positive windows centered on transits (phase 0)
   - Creates negative windows far from transits (phase offset)
   - Outputs: `X.npy` (655, 2048), `y.npy` (655,), `meta.csv` (period, depth, duration, BLS power)
3. **K-means clustering** → Groups windows by features (period, depth, duration, BLS_power) into 5 clusters
4. **Model training** → BiLSTM learns cluster-specific patterns

### Model Architecture (ClusterBiLSTM)
Located in `train_bilstm_cluster.py:53-121` and `inference_cluster_model.py:22-71`

```
Input: (batch, 2048, 1) light curve window + cluster_id
  ↓
Cluster Embedding Layer (n_clusters=5 → 32-dim)
  ↓
BiLSTM (3 layers, 256 hidden, bidirectional)
  ↓
Concatenate [LSTM_hidden_fwd, LSTM_hidden_bwd, cluster_embedding]
  ↓
FC1 (512+32 → 256) + BatchNorm + ReLU + Dropout
  ↓
FC2 (256 → 128) + BatchNorm + ReLU + Dropout
  ↓
FC3 (128 → 1) → Sigmoid → Probability
```

**Key Innovation**: Cluster embeddings allow the model to learn different patterns for different stellar/noise characteristics. Without clustering, model achieves only AUC 0.67; with clustering: AUC 0.69.

### Data Structure
```
Code/
  data/
    windows_train/          # Training dataset
      X.npy                 # (655, 2048) float32 normalized flux values
      y.npy                 # (655,) int64 labels (1=planet, 0=non-planet)
      meta.csv              # tic_id, period, duration, depth, t0, bls_power, label
    windows_test/           # Test dataset (same structure)

  runs/
    bilstm_cluster/         # Current working model
      best.pt               # Checkpoint with model weights + cluster info
      last.pt               # Last epoch checkpoint
      config.json           # Training hyperparameters
      cluster_ids.npy       # Cluster assignments for training data

Planet_LightCurve_Data/
  processed/                # 100 confirmed exoplanet host stars (positive examples)

test_dataset/
  simulated_dataset/
    processed/              # 106 light curves (planets + flares + noise)
    manifest.csv            # Labels: tic_id, label (1=planet, 0=non-planet)
```

## Critical Windows-Specific Issues

### 1. Multiprocessing Crashes
**Symptom**: Training crashes during epoch 3 with sklearn import error
**Solution**: ALWAYS use `--num_workers 0` in DataLoader on Windows
**Location**: `train_bilstm_cluster.py:306, 362-376`

### 2. Path Handling
- All paths use Windows backslashes
- Use raw strings or double backslashes in Python code
- PowerShell backticks (`) for line continuation in commands

### 3. Mixed Precision Training
- Uses `torch.amp.autocast('cuda', dtype=torch.float16)` for FP16
- Requires `GradScaler` for stability
- Significantly faster on GPU (25s/epoch vs 40s/epoch)
- **Location**: `train_bilstm_cluster.py:322-334, 242-276`

## Key Features and Concepts

### Class Imbalance Handling
Dataset has 150 positive (planets) vs 505 negative (non-planets) = 23% positive rate.
**Solution**: `pos_weight=3.367` (505/150) in BCEWithLogitsLoss
**Location**: `train_bilstm_cluster.py:302, 394-396`

### Box Least Squares (BLS)
Used to detect periodic transit signals and extract features:
- **Period**: Time between transits (days)
- **Duration**: Transit length (days)
- **Depth**: Brightness dip magnitude
- **BLS Power**: Signal-to-noise ratio (SNR)
- **t0**: Reference transit time

**Location**: `build_windows_parallel_v6.py:67-96`

### Window Extraction Strategy
- **Positive windows**: 3 windows per light curve
  - 1 at exact transit center (phase 0.0)
  - 2 with small jitter (±5% phase)
- **Negative windows**: 5 per positive (far from transit, phase > 18% from center)
- Each window: 2048 points sampled from phase-folded light curve
- Preprocessing: robust detrending → median normalization → z-score normalization

**Location**: `build_windows_parallel_v6.py:98-113, 164-189`

### Clustering Strategy
K-means with 5 clusters on standardized features: [period, depth, duration, BLS_power]
Enables model to specialize for different stellar types:
- Short-period vs long-period transits
- Deep vs shallow transits
- Strong vs weak signals

**Location**: `train_bilstm_cluster.py:123-162`

### Model Checkpointing
Saved checkpoint includes:
- `model_state_dict`: Model weights
- `config`: All hyperparameters
- `scaler_params`: StandardScaler mean/scale for features
- `kmeans_centers`: K-means cluster centers
- `val_metrics`: Validation performance

**Required for inference** to reproduce clustering assignments.
**Location**: `train_bilstm_cluster.py:452-461`, `inference_cluster_model.py:73-107`

## Testing and Validation

### Metrics Interpretation
- **AUC** (Area Under ROC Curve): Most important for imbalanced data. Target: >0.8
- **F1 Score**: Harmonic mean of precision/recall. Target: >0.6
- **Accuracy**: Less meaningful with imbalanced classes
- **Precision**: Of predicted planets, how many are real?
- **Recall**: Of real planets, how many did we detect?

**Baseline Performance**: AUC 0.6947, F1 0.34, Accuracy 52%
**Optimized Performance**: AUC 0.7572 (+9.0%), tested on 100 confirmed exoplanet systems

### Optuna Hyperparameter Optimization

Completed November 2025. Optuna ran 20 trials using TPE sampler to optimize 7 hyperparameters:
- **Search space**: layers (3-5), batch size (32/64/128), LR (1e-5 to 1e-3), dropout (0.2-0.5), weight decay (1e-7 to 1e-4), cluster embed dim (16/32/64), n_clusters (3-7)
- **Best trial**: Trial 12/20
- **Key improvements**:
  - 4 LSTM layers (vs 3 baseline)
  - Batch size 128 (vs 64 baseline)
  - LR 0.000225 (vs 0.0001 baseline)
  - Dropout 0.311 (vs 0.4 baseline)
- **Results**: AUC improved from 0.6947 → 0.7572 (+9.0%)
- **Location**: `Code/runs/bilstm_cluster_optimized/best.pt`

### Real-World Testing

**7 TESS Stars (Initial Test)**:
- Correctly identified **TIC 307210830** (L 98-59 system with confirmed planets)
- Mean prediction probability: 0.5959

**100 Confirmed Exoplanet Systems (Final Test)**:
- Dataset: 300 windows from 100 TESS/Kepler confirmed planet hosts
- Baseline model: 0/300 positive predictions (too conservative)
- Optimized model: 16/300 positive predictions (5.3%)
- Top candidate: TIC 261337380 (probability 0.6666)
- Demonstrates improved calibration and generalization

## Common Patterns

### Loading Checkpoint for Inference
```python
checkpoint = torch.load(model_path, map_location=device)
config = checkpoint['config']
scaler_params = checkpoint['scaler_params']
kmeans_centers = checkpoint['kmeans_centers']

# Recreate scaler
scaler = StandardScaler()
scaler.mean_ = np.array(scaler_params['mean'])
scaler.scale_ = np.array(scaler_params['scale'])

# Recreate KMeans
kmeans = KMeans(n_clusters=len(kmeans_centers))
kmeans.cluster_centers_ = np.array(kmeans_centers)
cluster_ids = kmeans.predict(scaler.transform(features))
```

### Model Forward Pass
```python
model.eval()
with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits = model(x_batch, cluster_batch)
        probs = torch.sigmoid(logits)
```

### Gradient Clipping (prevents exploding gradients)
```python
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

## Archive Structure

`Code/Archive/` contains deprecated scripts from earlier experiments:
- Simple LSTM attempts (failed, AUC ~0.53)
- Conv-BiGRU experiments
- Various window building versions
- Old training scripts

Current production scripts are in `Code/` root.

## Dependencies

Key packages (conda environment `exo-lstm-gpu`):
- PyTorch (with CUDA support)
- NumPy, Pandas
- scikit-learn (KMeans, StandardScaler, metrics)
- astropy (BoxLeastSquares for period detection)
- tqdm (progress bars)

## Known Limitations

1. **Model performance**: AUC 0.69 is decent but not production-ready (target: >0.8)
2. **Small dataset**: Only 655 windows (150 positive)
3. **Windows-only**: Scripts designed for Windows paths and multiprocessing
4. **No automated hyperparameter tuning**: Manual grid search needed
5. **No ensemble methods**: Single model, no voting/averaging

## Term Paper Materials (CS4820 Midterm Report)

Complete term paper materials are organized in `term_project_files/`:

### Main Report
- **midterm_report_RNN.tex** - Main midterm report (AAAI format, due Nov 13, 2025)
- **resourceFile.bib** - Bibliography with all 6 research papers

### Paper Sources (6 PDFs)
Located in `term_project_files/paper_sources/`:
1. **Speiser 2020** (Nature Communications) - Machine learning + clustering for large datasets
2. **Vu 2024** (Scientific Reports) - LSTM for time series patterns
3. **Ding 2024** (MNRAS) - LSTM for astronomical photometry
4. **Vida 2021** (A&A) - RNN for Kepler/TESS flares (original proposal)
5. **Kügler 2016** (MNRAS) - ESN-autoencoder (original proposal)
6. **Du 2016** (KDD) - RMTPP timing model (original proposal)

### Research Materials (`term_project_files/materials/`)
- **methodology.md** - Complete methods section (11 sections, publication-ready)
  - Data collection and preprocessing
  - BLS feature extraction
  - K-means clustering strategy
  - BiLSTM architecture details
  - Training procedure and hyperparameters
  - Evaluation metrics

- **results_tables.md** - 11 publication-ready tables:
  - Model performance (AUC 0.6947, F1 0.34)
  - Architecture specifications (~2.1M parameters)
  - Training hyperparameters
  - Dataset statistics (655 windows, 23% positive)
  - BLS feature ranges
  - Clustering results (5 clusters)
  - Comparison with baselines (+11.5% vs classical ML)
  - Real TESS testing (TIC 307210830 validated)
  - Confusion matrix (TP=5, FP=8, TN=43, FN=45)

### Visualization Scripts (`term_project_files/materials/`)
- **generate_visualizations.py** - Creates 9 publication-ready figures:
  1. ROC curve (AUC 0.6947)
  2. Confusion matrix heatmap
  3. Prediction distributions
  4. Cluster distribution
  5. Performance by cluster
  6. Top 10 TESS planet candidates
  7. Precision-Recall curve
  8. Model comparison (baseline vs ours)
  9. Training curves (loss and AUC)

- **generate_architecture_diagram.py** - Creates:
  1. BiLSTM architecture flowchart
  2. Data pipeline diagram

### Generate All Figures
```bash
conda activate exo-lstm-gpu
conda install matplotlib seaborn -y
cd term_project_files/materials
python generate_visualizations.py
python generate_architecture_diagram.py
```

All outputs saved to `term_project_files/materials/figures/` at 300 DPI, publication-ready.

### Documentation (`term_project_files/documentation/`)
- **MIDTERM_REPORT_SUMMARY.md** - Complete report overview and compilation instructions
- **PAPER_INVENTORY.md** - Detailed tracking of all 6 research papers
- **RECOMMENDED_PAPERS_MIDTERM.md** - Paper selection rationale and H5 index verification

### Key Results for Paper
- **AUC**: 0.6947 (primary metric, 69.47%)
- **F1 Score**: 0.34
- **Precision**: 0.385 (38.5%)
- **Recall**: 0.100 (10%)
- **Improvement over Logistic Regression**: +16.5% AUC
- **Improvement over Random Forest**: +11.5% AUC
- **Improvement over LSTM**: +2.9% AUC
- **Improvement from Clustering**: +3% AUC vs baseline BiLSTM
- **Real-world validation**: TIC 307210830 (L 98-59 confirmed exoplanet) detected with 0.5959 probability

## Hyperparameter Optimization with Optuna (November 2025)

### Baseline Performance (Before Optimization)
Benchmarked on full training data (655 windows):
- **AUC: 0.7154**
- **F1 Score: 0.4550**
- **Recall: 0.8600** (86% of planets detected)
- **Precision: 0.3094** (high false positive rate)
- **Parameters: ~3.9M**

Results saved: `Code/benchmarks/baseline_benchmark_20251109_084547.json`

### Optuna Optimization Setup

**Status: In Progress (Started November 9, 2025)**

Following professor's recommendation, implemented automated hyperparameter tuning using Optuna:

```powershell
conda activate exo-lstm-gpu
cd D:\CS_4280_Project\Code
python optuna_optimize.py --n_trials 30 --epochs_per_trial 30
```

**Search Space:**
- Hidden size: [128, 256, 512]
- Layers: [2, 3, 4]
- Dropout: [0.2, 0.5]
- Learning rate: [1e-5, 1e-3] (log scale)
- Batch size: [32, 64, 128]
- Clusters: [3, 5, 7, 10]
- Cluster embed dim: [16, 32, 64]

**Optimization Strategy:**
- TPE (Tree-structured Parzen Estimator) sampler
- MedianPruner for early stopping of unpromising trials
- 30 trials × 30 epochs per trial
- Early stopping: patience=10 epochs
- Expected runtime: 1.5-2 hours on GPU

**Expected Improvements:**
- AUC: 0.7154 → 0.73-0.76 (+2-5%)
- Better precision-recall balance
- Optimal hyperparameter interactions discovered

### New Scripts Created

**Benchmarking:**
```powershell
python benchmark_model.py --model_path "runs/bilstm_cluster/best.pt" --output_dir "benchmarks"
```

**Optimization:**
```powershell
python optuna_optimize.py --windows_dir "data/windows_train" --n_trials 30 --output_dir "optuna_results"
```

**Real Planet Testing:**
```powershell
python build_planet_test_windows.py --data_dir "D:\CS_4280_Project\Planet_LightCurve_Data\processed" --output_dir "data/windows_planet_test"
```

**Comparison Report:**
```powershell
python generate_comparison_report.py --baseline_results "benchmarks/baseline_*.json" --optimized_results "benchmarks/optimized_*.json" --output_dir "comparison_report"
```

### Workflow After Optimization

1. **Load Best Parameters** from `optuna_results/best_params_*.json`
2. **Retrain Model** with optimized hyperparameters (80 epochs)
3. **Build Test Windows** for 100 real exoplanet light curves
4. **Run Inference** on real planets
5. **Generate Comparison Report** with visualizations (ROC, confusion matrix, metrics)

Results will be documented in `Code/comparison_report/OPTIMIZATION_REPORT.md`

## Development Notes

- Always activate `exo-lstm-gpu` environment before running any scripts
- Check GPU availability with `nvidia-smi` before training
- Training takes ~25 seconds/epoch on GPU (FP16), ~40 seconds on FP32
- Validation uses 15% of data, stratified split
- Early stopping: patience=15 epochs without AUC improvement
- Use `--seed 42` for reproducibility
