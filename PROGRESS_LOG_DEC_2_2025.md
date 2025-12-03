# Progress Log - December 2, 2025

## Session Overview

**Date**: December 2, 2025
**Context**: Preparing final submission for CS 4280/5820 AI course term project
**Deadlines**:
- Presentations: Dec 9-11, 2025
- Final Submission: Dec 18, 2025

**Team**: 3 members (Josh Manchester = RNN, Tristan = CNN, Brianne = Transformer)
**Josh's Component**: BiLSTM + K-means Clustering for exoplanet detection

---

## What We Did Today

### 1. Created Code/README.md ✅
- Location: `C:\CS_4280_Project\Code\README.md`
- Purpose: Clear instructions for TA to run the code
- Contents:
  - Environment setup (conda, PyTorch, dependencies)
  - Data location and how to obtain it
  - Training command with all hyperparameters
  - Model architecture description
  - Expected results

### 2. Condensed RNN Related Work Section ✅
**Why**: TA feedback said "Related work section is too long. Condensing it to focus on only the most relevant studies will improve clarity."

**Original**: ~6,700 words across 9 papers (lines 147-262 in original)
**Condensed**: ~1,036 words (85% reduction)

**Strategy Used**:
- First 6 papers (from midterm): Brief "what + why I used it" format
- 3 new papers (Becker, Schanche, Malik): Detailed coverage
- Synthesis section: Bulleted list of design decisions

**Files Created**:
| File | Purpose |
|------|---------|
| `Merged_Proposal_AAAI24.11.22.2025.1743.BACKUP_ORIGINAL.tex` | Backup of original paper |
| `Merged_Proposal_CONDENSED_RNN_12.2.2025.tex` | Working version with condensed RNN |
| `RNN_RELATED_WORK_CONDENSED_v2.tex` | Standalone condensed RNN section |

### 3. Installed MiKTeX LaTeX ✅
- Installed via: `winget install MiKTeX.MiKTeX`
- Copied `aaai24.sty` to term_project_files folder
- Successfully compiled paper to PDF

### 4. Verified Page Count ✅
- **Current page count**: 18 pages
- **Target**: 15-18 pages total (for entire team of 3)
- **Status**: At upper bound, but teammates will also condense their sections

### 5. Updated RNN Methodology Section ✅ COMPLETE
**Why**: Current methodology describes old 655-window dataset, needs update for Sector 1

**All Changes Made**:

| Section | Old Value | New Value | Status |
|---------|-----------|-----------|--------|
| Data source | 100 TESS + 106 simulated | TESS Sector 1 Ground Truth (13,541 light curves) | ✅ Done |
| Total windows | 655 | 33,051 (26,472 train + 6,579 test) | ✅ Done |
| Clustering features | BLS (period, depth, duration, power) | Statistical (mean, std, var, skew, range, median, MAD, peak-to-peak) | ✅ Done |
| BiLSTM layers | 3 | 4 | ✅ Done |
| Dropout | 0.4 | 0.311 (Optuna optimized) | ✅ Done |
| Parameters | 2.1M | 3.9M | ✅ Done |
| pos_weight | 3.367 | 7.41 | ✅ Done |
| Class ratio | 23% positive | 11.9% positive | ✅ Done |
| Data split | 85/15 train/val | 80/20 train/test | ✅ Done |
| Epochs | 80 | 60 | ✅ Done |
| Batch size rationale | N/A | Added (64 optimal from benchmark) | ✅ Done |

**PDF recompiled: 17 pages** (down from 18, before conclusion update)

### 6. Updated Conclusion Section ✅
**Why**: Conclusion was in proposal format ("we plan to..."), needed final format

**Before** (proposal style):
> "In conclusion, through this research, we plan to solve two problems..."

**After** (final report style):
- Opening paragraph: summarizes what we investigated (3 architectures on TESS Sector 1)
- **RNN Findings**: BiLSTM + clustering results with placeholder [AUC TO BE UPDATED]
- **CNN Findings**: placeholder for teammate
- **Transformer Findings**: placeholder for teammate
- **Comparative Analysis**: placeholder for combined results
- **Contributions**: 3 key contributions listed

**Placeholders to update later**:
- `[AUC TO BE UPDATED]` - after training on new PC
- `[TO BE COMPLETED BY TEAMMATE]` - CNN and Transformer sections
- Comparative Analysis section

**PDF recompiled: 18 pages** (conclusion added ~1 page of content)

### 7. Updated RNN Results Tables for Sector 1 ✅
Updated all tables that can be fixed NOW (before training on new PC):

| Table | What Changed |
|-------|--------------|
| **Dataset Stats** | 655 → 33,051 windows, 23% → 11.9% positive, 106 → 13,541 light curves |
| **Architecture** | Added BiLSTM Layer 4, 2.1M → 3.9M parameters |
| **Hyperparameters** | Epochs 80→60, dropout 0.4→0.311, layers 3→4, pos_weight 3.367→7.41 |
| **Clustering** | Changed from BLS features (period, depth) to statistical features (mean, std, skew, etc.) |

**Updated image paths for Overleaf structure:**
- `bilstm_smote_roc_curve.png` → `Images/RNN/bilstm_smote_roc_curve.png`
- `bilstm_class_imbalance_comparison.png` → `Images/RNN/bilstm_class_imbalance_comparison.png`
- `confusion_matrix.png` → `Images/RNN/confusion_matrix.png`

**Tables/Figures that still need updating AFTER training:**
- Performance table (AUC 0.6947 → new results)
- TESS testing table (7 stars)
- Optuna optimization results table
- Planet test results table (100 confirmed exoplanets)
- Top 10 candidates table
- SMOTE comparison table
- All 3 RNN figures (ROC, confusion matrix, class imbalance comparison)

**PDF recompiled: 18 pages** (still within 15-18 target)

---

## What Still Needs To Be Done

### Immediate (Today/This Session)

1. ~~**Finish RNN Methodology Update**~~ ✅ COMPLETE
   - ~~Update pos_weight from 3.367 to 7.41~~
   - ~~Update class ratio from 23% to 11.9%~~
   - ~~Update data split description from 85/15 to 80/20~~
   - ~~Update training epochs, batch size, learning rate~~
   - ~~Location: Lines 383-391 in condensed tex file~~

2. ~~**Write Tentative Conclusion**~~ ✅ COMPLETE
   - ~~Needs to be a placeholder that can be updated with teammate results~~
   - ~~Should summarize RNN findings~~
   - ~~Will be updated when CNN and Transformer results available~~

3. ~~**Update RNN Results Section**~~ ✅ COMPLETE
   - ~~Tables that can be updated now: Dataset, Architecture, Hyperparams, Clustering~~
   - ~~Performance table updated with Sector 1 results (AUC 0.893)~~
   - ~~Image paths updated for Overleaf structure~~
   - ~~New figures generated (ROC curve, confusion matrix)~~

4. **Upload to Overleaf** (NEXT)
   - Upload `Merged_Proposal_CONDENSED_RNN_12.2.2025.tex`
   - Verify image paths work with `Images/RNN/` folder

### Before Presentation (Dec 9-11)

1. **Train final model on new PC** (arriving today Dec 2)
   - Run batch size benchmark
   - Run Optuna optimization on Sector 1
   - Train final model with optimized hyperparameters
   - Expected AUC: 0.91+

2. **Create original figures**
   - ROC curve
   - Confusion matrix
   - Training curves
   - Architecture diagram

3. **Record 20-second demo video**

4. **Prepare presentation slides**

### Before Final Submission (Dec 18)

1. **Get teammate results** (CNN, Transformer)
2. **Update Conclusion** with comparison of all 3 methods
3. **Final proofreading**

---

## Key Technical Details

### Dataset: TESS Sector 1 Ground Truth

| Category | Count | Percentage |
|----------|-------|------------|
| Planets (confirmed/candidate) | 3,146 | 23.2% |
| Stars (non-planet) | 8,624 | 63.7% |
| Eclipsing Binaries | 900 | 6.6% |
| Background EBs | 871 | 6.4% |
| **Total** | **13,541** | 100% |

**Windows Generated**: 33,051 (3 per light curve)
**Train/Test Split**: 80/20 → 26,472 train, 6,579 test
**Class Balance in Training**: 3,147 planets (11.9%) vs 23,325 non-planets (88.1%)
**pos_weight**: 23,325 / 3,147 = **7.41**

### Optimal Hyperparameters (from Optuna + benchmarking)

| Parameter | Value | Source |
|-----------|-------|--------|
| BiLSTM layers | 4 | Optuna |
| Hidden size | 256 | Optuna |
| Dropout | 0.311 | Optuna |
| Learning rate | 0.0001 | Manual (0.000225 caused NaN) |
| Batch size | 64 | Benchmark (128 too slow) |
| pos_weight | 7.41 | Calculated |
| Epochs | 60 | Standard |
| num_workers | 0 | Windows requirement |

### Training Command
```powershell
python train_bilstm_cluster.py `
  --windows_dir "E:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train" `
  --n_clusters 5 `
  --epochs 60 `
  --batch_size 64 `
  --lr 0.0001 `
  --hidden 256 `
  --layers 4 `
  --dropout 0.311 `
  --pos_weight 7.41 `
  --save_dir "runs\sector1_batch64" `
  --amp_dtype fp16 `
  --num_workers 0 `
  --seed 42
```

---

## File Locations

### Paper Files
```
C:\CS_4280_Project\term_project_files\
├── Merged_Proposal_AAAI24.11.22.2025.1743.tex          # Original (DO NOT EDIT)
├── Merged_Proposal_AAAI24.11.22.2025.1743.BACKUP_ORIGINAL.tex  # Backup
├── Merged_Proposal_CONDENSED_RNN_12.2.2025.tex         # WORKING VERSION
├── Merged_Proposal_CONDENSED_RNN_12.2.2025.pdf         # Compiled PDF (18 pages)
├── RNN_RELATED_WORK_CONDENSED_v2.tex                   # Standalone RNN section
├── aaai24.sty                                          # AAAI style file
└── resourceFile.bib                                    # Bibliography (23 papers)
```

### Code Files
```
C:\CS_4280_Project\Code\
├── README.md                           # NEW - Created today
├── train_bilstm_cluster.py             # Main training script
├── build_windows_from_groundtruth.py   # Build Sector 1 windows
├── inference_cluster_model.py          # Run inference
├── evaluate_test.py                    # Evaluate on test set
├── optuna_optimize.py                  # Hyperparameter optimization
└── benchmark_batch_sizes.py            # Find optimal batch size
```

### Data Locations
```
E:\CS_4280_Project_Backup\Code\data\windows_sector1_full\
├── train\
│   ├── X.npy      # (26472, 2048) float32
│   ├── y.npy      # (26472,) int64
│   └── meta.csv   # Metadata with statistical features
└── test\
    ├── X.npy      # (6579, 2048) float32
    ├── y.npy      # (6579,) int64
    └── meta.csv
```

---

## Syllabus Requirements Checklist

### Final Report (25% of grade)
- [x] AAAI format
- [x] 9 references per student (Josh has 9 RNN papers)
- [ ] 15-18 pages total (currently 18, teammates need to condense)
- [ ] 10 required sections (most present, need Problem Statement check)

### Required 10 Sections
1. [x] Project Title
2. [x] Authors
3. [x] Abstract
4. [x] Introduction
5. [x] Background & Related Work
6. [ ] Problem Statement (may be embedded in Intro)
7. [x] Methodology
8. [x] Results
9. [ ] Discussion & Conclusion (needs expansion)
10. [x] References

### Demo Requirements
- [ ] 20-second demo video
- [ ] Show model running on test data
- [ ] Display predictions/results

---

## TA Feedback to Address

From Ali Al Shami (Nov 30):

1. **Introduction** - "reads more like a related work section"
   - Needs: engaging opening → problem → motivation → contributions
   - Status: SHARED SECTION (coordinate with team)

2. **Related Work** - "too long, condense to focus on most relevant studies"
   - Status: ✅ DONE for RNN section (85% reduction)

3. **Methodology, Dataset, Results** - "clearly presented and well done"
   - Status: Keep format, update data for Sector 1

---

## New PC Setup (When It Arrives)

### Hardware
- CPU: Intel i9-14900KF (24 cores)
- GPU: RTX 5070 Ti (16 GB VRAM)
- RAM: 32 GB DDR5-6000
- Storage: 2TB NVMe + Crucial T710 4TB Gen5 NVMe

### Setup Steps
1. Install Miniconda
2. Create exo-lstm-gpu environment
3. Copy data from backup drive
4. Run benchmark_batch_sizes.py (expect batch 128-192 optimal)
5. Set HIGH_VRAM_GPU = True in optuna_optimize.py
6. Run Optuna on Sector 1 dataset
7. Train final model

---

## Resume Instructions

To continue this work:

1. **Open the working tex file**:
   `C:\CS_4280_Project\term_project_files\Merged_Proposal_CONDENSED_RNN_12.2.2025.tex`

2. **Go to line 385** - Training Procedure section

3. **Update these values**:
   - pos_weight: 3.367 → 7.41
   - Class ratio: 23% → 11.9%
   - Data split: 85/15 → 80/20
   - Batch size: 64 → 64 (mention benchmark)
   - Learning rate: 1e-4 → 0.0001

4. **After Methodology, move to**:
   - Write tentative Conclusion
   - Update Results section (or add placeholders)

5. **Recompile PDF** to check page count:
   ```powershell
   cd "C:\CS_4280_Project\term_project_files"
   "C:\Users\manch\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode Merged_Proposal_CONDENSED_RNN_12.2.2025.tex
   ```

---

## Questions to Ask Professor/TA

1. Is the 15-18 page requirement for the TOTAL paper or per student?
   - We believe: TOTAL (which aligns with TA's "too long" feedback)

2. Should Related Work be condensed even though syllabus says add 2 pages per submission?
   - We believe: Yes, TA feedback overrides (he's grading it)

3. Is Problem Statement a separate section or can it be embedded in Introduction?

---

## Session 2: Afternoon - Final Results Update

### 8. Updated Performance Table with Actual Sector 1 Results ✅

Found actual test results from `sector1_test_predictions.csv`:

| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.893** |
| F1 Score | 0.580 |
| Accuracy | 83.9% |
| Precision | 41.8% |
| Recall | 94.6% |
| True Positives | 729 |
| False Positives | 1,015 |
| True Negatives | 4,793 |
| False Negatives | 42 |

**Test set**: 6,579 windows (771 planets, 5,808 non-planets)

### 9. Generated New Figures ✅

Created `generate_sector1_figures.py` and generated:
- `term_project_files/Images/RNN/bilstm_sector1_roc_curve.png` - ROC curve showing AUC 0.893
- `term_project_files/Images/RNN/confusion_matrix.png` - Confusion matrix with all metrics

### 10. Updated Conclusion ✅

Changed from placeholder `[AUC TO BE UPDATED]` to actual results:
- AUC 0.893
- 94.6% recall
- 729 of 771 planets detected
- 83.9% overall accuracy

### 11. Kepler Data Status

**Clarified**: Kepler-only testing was not fully completed. The approach was:
1. Tested TESS model on Kepler data → failed (domain shift)
2. Created combined dataset (TESS + Kepler) for training
3. Main reportable results are from **TESS Sector 1 test set**

Kepler data exists at `E:\CS_4280_Project_Backup\Code\data\windows_kepler\`:
- 279 windows, all positives (confirmed exoplanets)
- Used for training augmentation, not separate testing

---

## Final Deliverables for Upload to Overleaf

### Files to Upload:
1. `term_project_files/Merged_Proposal_CONDENSED_RNN_12.2.2025.tex` - Updated paper
2. `term_project_files/Images/RNN/bilstm_sector1_roc_curve.png` - New ROC curve
3. `term_project_files/Images/RNN/confusion_matrix.png` - New confusion matrix

### Key Results Summary (RNN Component):
| Dataset | Windows | AUC | Recall | Accuracy |
|---------|---------|-----|--------|----------|
| TESS Sector 1 Test | 6,579 | **0.893** | 94.6% | 83.9% |

---

## Session 3: New PC Setup (December 2, 2025 - Evening)

### 12. New System Arrived and Setup ✅

**Hardware Verified:**
- **GPU**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **Driver**: 576.88

### 13. Environment Setup ✅

1. **Miniconda installed** via `winget install Anaconda.Miniconda3`
2. **ToS accepted** for Anaconda channels
3. **Created conda environment**: `exo-lstm-gpu` with Python 3.11
4. **PyTorch nightly installed** (required for Blackwell architecture sm_120):
   - Version: `2.10.0.dev20251202+cu128`
   - Standard PyTorch 2.6.0 does NOT support RTX 5070 Ti
   - Used: `pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128`
5. **All dependencies installed**: numpy, pandas, scikit-learn, optuna, matplotlib, seaborn, astropy, scipy, tqdm

### 14. GPU Verification ✅

```
PyTorch: 2.10.0.dev20251202+cu128
CUDA available: True
GPU: NVIDIA GeForce RTX 5070 Ti
VRAM: 15.9 GB
```

### 15. Data Located ✅

Data already present on D: drive from old system:
- **Training**: `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train\`
  - 26,472 windows (3,147 planets = 11.9%)
- **Test**: `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\test\`
  - 6,579 windows (732 planets = 11.1%)

### 16. Batch Size Benchmark ✅

**Coarse benchmark (powers of 2):**

| Batch Size | Est. Epoch Time | Notes |
|------------|-----------------|-------|
| 8 | 21.48 min | Too many iterations |
| 16 | 11.15 min | |
| 32 | 5.77 min | |
| 64 | 3.05 min | |
| 128 | 1.77 min | Near optimal |
| 256 | 15.02 min | Memory pressure |
| 512 | OOM | Crash |

**Fine-grained benchmark (112-144):**

| Batch Size | Est. Epoch Time | Notes |
|------------|-----------------|-------|
| 112 | 2.00 min | |
| 120 | 1.86 min | |
| 128 | 1.78 min | |
| **136** | **1.75 min** | **OPTIMAL** |
| 144 | 1.78 min | |

**Key Finding**: Batch size **136** is optimal (1.75 min/epoch)
- **1.8× faster** than old RTX 3060 Ti (was 3.14 min/epoch with batch 64)
- 60 epochs = **1.75 hours** training time

### 17. Optuna Config Updated ✅

Changed `optuna_optimize.py` line 214:
```python
HIGH_VRAM_GPU = True  # Was False
```

This enables larger search space:
- hidden_size: [128, 256, 512]
- batch_size: [64, 128, 192, 256]

---

## Next Steps (When Resuming)

1. **Run Optuna optimization** (~10-15 hours, run overnight):
   ```powershell
   C:\Users\manch\miniconda3\envs\exo-lstm-gpu\python.exe optuna_optimize.py `
     --windows_dir "D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train" `
     --n_trials 30 `
     --epochs_per_trial 30 `
     --output_dir "optuna_results_sector1_5070ti"
   ```

2. **Train final model** with optimized hyperparameters (~1.75 hours)

3. **Evaluate on test set** and update paper with final results

4. **Generate figures** (ROC curve, confusion matrix)

5. **Record 20-second demo video**

6. **Prepare presentation slides**

---

## Key Technical Notes

### PyTorch Blackwell Support
- RTX 5070 Ti uses **Blackwell architecture (sm_120)**
- Standard PyTorch 2.6.0 only supports up to sm_90 (Ada Lovelace)
- **Must use PyTorch nightly** with CUDA 12.8 for Blackwell support
- Install command: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`

### Python Executable Path
Since conda isn't in PATH on new system, use full path:
```
C:\Users\manch\miniconda3\envs\exo-lstm-gpu\python.exe
```

---

**Last Updated**: December 2, 2025, evening session
**Status**: New PC setup complete, ready for Optuna run
