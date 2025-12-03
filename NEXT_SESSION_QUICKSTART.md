# Next Session Quick Start Guide

## Last Session: December 3, 2025 (Afternoon)

### Status: RNN PAPER SECTIONS COMPLETE - WAITING FOR TEAMMATES

**What Was Done Today**:
- Expanded RNN Related Work to 2 full pages (added ~1,350 words to Becker, Schanche, Malik sections)
- Converted paper from proposal format to final submission format
- Removed proposal artifacts (Experimental Plan, Risks sections)
- Added cross-mission generalization analysis (TESS → Kepler)
- Updated AI disclosure for Opus 4.5

**Key Finding from Earlier**: Optuna Trial 0 achieved **AUC 0.916** (best ever!)

---

## What's Complete (RNN Component)

- [x] RNN Related Work - 2 full pages (6 foundational + 3 new papers)
- [x] RNN Results section - condensed with key tables
- [x] Cross-mission generalization analysis (TESS → Kepler)
- [x] SOFA refactoring - all Python files at pylint 10.00/10
- [x] Code README.md for TA

## What's Waiting On

- [ ] **Teammates (Tristan/Bree)** to update their CNN/Transformer sections
- [ ] Resume Optuna optimization OR train final model
- [ ] Generate final figures (ROC, confusion matrix)
- [ ] Create 20-second demo video
- [ ] Prepare presentation slides

---

## When You Return: Priority Options

### Option A: Resume Optuna (~9 hours)

If you have overnight time available:

```powershell
conda activate exo-lstm-gpu
cd D:\CS_4280_Project\Code

python optuna_optimize.py `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train `
  --n_trials 15 `
  --epochs_per_trial 20 `
  --output_dir optuna_results_sector1_5070ti
```

### Option B: Train Final Model (~1.75 hours)

If you need results quickly:

```powershell
conda activate exo-lstm-gpu
cd D:\CS_4280_Project\Code

python train_bilstm_cluster.py `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train `
  --n_clusters 5 `
  --epochs 60 `
  --batch_size 136 `
  --lr 0.0001 `
  --hidden 256 `
  --layers 4 `
  --dropout 0.311 `
  --pos_weight 7.41 `
  --save_dir runs\sector1_final `
  --amp_dtype fp16 `
  --num_workers 0 `
  --seed 42
```

### Option C: Wait for Teammates

If paper updates from teammates are coming soon, wait before making more changes to avoid merge conflicts.

---

## Key Parameters (VERIFIED)

| Parameter | Value | Source |
|-----------|-------|--------|
| **batch_size** | **136** | Benchmarked (optimal for RTX 5070 Ti) |
| lr | 0.0001 | Stable (0.000225 caused NaN crash) |
| pos_weight | 7.41 | Calculated (23325/3147) |
| hidden | 256 | Previous Optuna |
| layers | 4 | Previous Optuna |
| dropout | 0.311 | Previous Optuna |
| epochs | 60 | Standard |

---

## Data Locations

| Data | Location |
|------|----------|
| Training windows | `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train\` |
| Test windows | `D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\test\` |
| Code | `D:\CS_4280_Project\Code\` |
| Working paper | `D:\CS_4280_Project\term_project_files\Merged_Proposal_as_of_12.2.2025.tex` |

**Training set**: 26,472 windows (3,147 planets = 11.9%)
**Test set**: 6,579 windows

---

## After Training: Generate Figures

### Step 1: Evaluate on Test Set

```powershell
python inference_cluster_model.py `
  --model_path runs\sector1_final\best.pt `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\test `
  --output_file reports\sector1_final_predictions.csv
```

### Step 2: Generate Figures for Paper

```powershell
python generate_sector1_figures.py
```

Creates ROC curve and confusion matrix in `term_project_files/Images/RNN/`

---

## Best Results So Far

| Dataset | Model | AUC | Status |
|---------|-------|-----|--------|
| Old (655 windows) | BiLSTM+Cluster | 0.7572 | Archived |
| Sector 1 (trained Nov) | BiLSTM+Cluster | 0.893 | Current best saved |
| **Sector 1 (Optuna Trial 0)** | BiLSTM+Cluster | **0.916** | Not saved (interrupted) |

---

## Timeline to Final Submission

| Date | Task | Status |
|------|------|--------|
| ~~Dec 2~~ | New PC setup + benchmark | ✅ Done |
| ~~Dec 2-3~~ | Optuna overnight | ⚠️ Interrupted |
| ~~Dec 3~~ | Paper finalization (RNN sections) | ✅ Done |
| **Dec 4-5** | Resume Optuna OR train final model | **NEXT** |
| Dec 5-6 | Generate figures + finalize paper | Pending |
| Dec 6-7 | Create slides + record 20s demo | Pending |
| Dec 7-8 | Practice presentation | Buffer |
| **Dec 9-11** | **Presentations** | |
| **Dec 18** | **Final submission deadline** | |

---

## Files Changed This Session

| File | Change |
|------|--------|
| `term_project_files/Merged_Proposal_as_of_12.2.2025.tex` | Expanded RNN Related Work, removed proposal sections |
| `CLAUDE.md` | Added Dec 3 afternoon update |
| `PROGRESS_LOG_DEC_3_2025.md` | Added afternoon session details |
| `NEXT_SESSION_QUICKSTART.md` | This file - updated |
| `Code/TRAINING_STATUS.md` | Updated to reflect current state |

---

**Last Updated**: December 3, 2025, afternoon
**Status**: RNN paper sections complete, waiting for teammates
