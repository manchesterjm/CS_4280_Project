# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an exoplanet detection project using deep learning (BiLSTM with K-means clustering) to identify planetary transits in stellar light curve data from NASA's TESS/Kepler missions. The project has achieved **AUC 0.6947** on validation data and successfully tested on real TESS light curves.

**Environment**: Windows 11, CUDA-enabled GPU, conda environment `exo-lstm-gpu`

## Key Commands

### Setup
```powershell
conda activate exo-lstm-gpu
cd C:\CS_4280_Project\Code
```

### Training the BiLSTM+Clustering Model (Current Working Model)
```powershell
python train_bilstm_cluster.py `
  --windows_dir "C:\CS_4280_Project\Code\data\windows_train" `
  --n_clusters 5 `
  --epochs 80 `
  --batch_size 64 `
  --lr 1e-4 `
  --hidden 256 `
  --layers 3 `
  --dropout 0.4 `
  --save_dir "C:\CS_4280_Project\Code\runs\bilstm_cluster" `
  --amp_dtype fp16 `
  --pos_weight 3.367 `
  --num_workers 0
```

**Critical**: `--num_workers 0` must be used on Windows to avoid multiprocessing crashes

### Building Training Windows
```powershell
python build_windows_parallel_v6.py `
  --processed_dir "C:\CS_4280_Project\test_dataset\simulated_dataset\processed" `
  --out_dir "C:\CS_4280_Project\Code\data\windows_train" `
  --seq_len 2048 `
  --neg_per_pos 5 `
  --n_jobs -1 `
  --seed 42 `
  --manifest "C:\CS_4280_Project\test_dataset\simulated_dataset\manifest.csv"
```

### Testing on New TESS Data (Complete Pipeline)
```powershell
# 1. Download TESS light curves
python download_tess_lightcurves.py --tic_list sample_tic_ids.txt --output_dir "C:\CS_4280_Project\test_dataset_v2\raw"

# 2. Process downloaded data
python process_tess_for_testing.py --raw_dir "C:\CS_4280_Project\test_dataset_v2\raw" --output_dir "C:\CS_4280_Project\test_dataset_v2\processed"

# 3. Convert to CSV format
python convert_npy_to_csv.py --input_dir "C:\CS_4280_Project\test_dataset_v2\processed" --output_dir "C:\CS_4280_Project\test_dataset_v2\processed_csv" --max_points 50000

# 4. Build test windows
python build_simple_windows.py --data_dir "C:\CS_4280_Project\test_dataset_v2\processed_csv" --output_dir "C:\CS_4280_Project\Code\data\windows_test"

# 5. Run inference
python inference_cluster_model.py --model_path "C:\CS_4280_Project\Code\runs\bilstm_cluster\best.pt" --windows_dir "C:\CS_4280_Project\Code\data\windows_test" --output_file "C:\CS_4280_Project\Code\reports\test_predictions.csv"
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

**Current Performance**: AUC 0.6947, F1 0.34, Accuracy 52%

### Real-World Testing
Successfully tested on 7 TESS stars, correctly identified **TIC 307210830** (L 98-59 system with confirmed planets).
Mean prediction probability: 0.5959

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

## Research Paper Materials

Complete research paper materials are available in `research_paper/`:

### Documentation Files
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

- **paper_template.tex** - Full LaTeX paper template with:
  - Abstract, Introduction, Related Work
  - Methodology (references methodology.md)
  - Results (references all tables and figures)
  - Discussion, Conclusion, Bibliography

### Visualization Scripts
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
cd research_paper
python generate_visualizations.py
python generate_architecture_diagram.py
```

All outputs saved to `research_paper/figures/` at 300 DPI, publication-ready.

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
cd C:\CS_4280_Project\Code
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
python build_planet_test_windows.py --data_dir "C:\CS_4280_Project\Planet_LightCurve_Data\processed" --output_dir "data/windows_planet_test"
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
