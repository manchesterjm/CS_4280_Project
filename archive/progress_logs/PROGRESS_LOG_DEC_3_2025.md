# Progress Log - December 3, 2025

## Session Overview

**Date**: December 3, 2025
**Session**: Overnight Optuna run + morning check-in
**GPU**: RTX 5070 Ti (16 GB VRAM)

---

## What Happened

### 1. Updated Optuna Batch Size Search Space

**Problem**: The original Optuna search space had batch sizes `[64, 128, 192, 256]`, but our benchmark showed:
- Batch 256 causes **memory pressure** (8.67 sec/batch vs 0.54 sec for batch 136)
- Optimal batch size is **136** (1.75 min/epoch)

**Fix Applied**: Updated `optuna_optimize.py` line 238-239:
```python
# Before:
batch_size = trial.suggest_categorical('batch_size', [64, 128, 192, 256])

# After:
batch_size = trial.suggest_categorical('batch_size', [112, 128, 136, 144])
```

### 2. Started Optuna Optimization (~11:01 PM Dec 2)

**Command**:
```powershell
python optuna_optimize.py `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train `
  --n_trials 30 `
  --epochs_per_trial 30 `
  --output_dir optuna_results_sector1_5070ti
```

**Configuration**:
- 30 trials × 30 epochs per trial
- Batch sizes: [112, 128, 136, 144]
- Hidden sizes: [128, 256, 512]
- Learning rate: 1e-5 to 1e-3 (log scale)
- Dropout: 0.2 to 0.5
- Layers: 2-4
- Clusters: [3, 5, 7, 10]

### 3. Morning Status Check (~7:00 AM Dec 3)

**Results**:
- **Trial 0 completed**: AUC = **0.9160** (excellent!)
- **Trial 1 in progress** (at ~7 hours elapsed)
- GPU utilization: 100%
- GPU memory: 15.6 GB / 16.3 GB (96% used)

**Problem Identified**: Each trial takes ~52 minutes (30 epochs × 1.75 min/epoch). At this rate:
- 30 trials = ~26 hours total
- Only ~8 trials would complete overnight

### 4. Stopped Optuna Run (~7:00 AM Dec 3)

**Reason**: User needed GPU for other work.

**What Was Lost**:
- Optuna results were not saved (saves only at end of optimization)
- Trial 0's hyperparameters were not captured
- However, we know Trial 0 achieved **AUC 0.916**

---

## Key Finding

**AUC 0.916 on Trial 0** is a major improvement over previous results:
- Previous best (old dataset): AUC 0.7572
- Previous Sector 1 test: AUC 0.893
- **New Optuna Trial 0: AUC 0.916** (+2.3% over previous Sector 1 result)

This suggests the Sector 1 dataset with proper hyperparameters can achieve very strong results.

---

## Current State

### Files Modified
| File | Change |
|------|--------|
| `Code/optuna_optimize.py` | Updated batch_size search to [112, 128, 136, 144] |

### Results Folder
- `Code/optuna_results_sector1_5070ti/` - **Empty** (no results saved)

### GPU Status
- **Available**: GPU freed after stopping Optuna
- Utilization: ~1%
- Memory: ~1.2 GB (system overhead only)

---

## What Still Needs To Be Done

### Option A: Resume Optuna (Faster Settings)
Run with fewer trials and epochs to complete faster:

```powershell
cd D:\CS_4280_Project\Code
conda activate exo-lstm-gpu
python optuna_optimize.py `
  --windows_dir D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train `
  --n_trials 15 `
  --epochs_per_trial 20 `
  --output_dir optuna_results_sector1_5070ti
```

**Estimated time**: 15 trials × ~35 min = **~9 hours**

### Option B: Skip Optuna, Train Final Model
Since Trial 0 already achieved AUC 0.916, we could train a final model with reasonable hyperparameters:

```powershell
cd D:\CS_4280_Project\Code
conda activate exo-lstm-gpu
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

**Estimated time**: ~1.75 hours (60 epochs × 1.75 min/epoch)

### After Training
1. Evaluate on test set
2. Generate figures (ROC curve, confusion matrix)
3. Update paper with final results
4. Record 20-second demo video
5. Prepare presentation slides

---

## Timeline

| Date | Task | Status |
|------|------|--------|
| Dec 2 (PM) | New PC setup + benchmark | ✅ Done |
| Dec 2-3 (overnight) | Optuna run | ⚠️ Interrupted (1/30 trials) |
| **Dec 3** | Resume Optuna OR train final model | **Pending** |
| Dec 4-5 | Generate figures + update paper | Pending |
| Dec 5-6 | Create slides + record demo | Pending |
| Dec 7-8 | Practice presentation | Buffer |
| **Dec 9-11** | **Presentations** | |
| **Dec 18** | **Final submission deadline** | |

---

## Benchmark Reference (RTX 5070 Ti)

| Batch Size | Est. Epoch Time | Notes |
|------------|-----------------|-------|
| 112 | 2.00 min | |
| 128 | 1.78 min | |
| **136** | **1.75 min** | **OPTIMAL** |
| 144 | 1.78 min | |
| 256 | 15.02 min | Memory pressure - AVOID |

---

---

## Afternoon Session (December 3, 2025)

### Paper Finalization Work

#### 1. RNN Related Work Section Expanded

**Requirement**: Each teammate needs 2 full pages of Related Work content.

**What Was Done**:
- Expanded the three "New Reference" subsections (Becker, Schanche, Malik) from ~500 words to ~1,350 words
- Added detailed methodology, results, and relevance analysis for each paper
- First 6 foundational papers kept as brief summaries (per professor guidance)

**Content Added**:
- **Becker et al. (CNN-LSTM)**: Added architecture variants comparison, detailed class imbalance handling, cross-survey generalization evidence
- **Schanche et al. (Ground-Based)**: Added "False Positive Problem" section, methodology comparison (RF vs CNN), quantitative feature importance
- **Malik et al. (Gradient Boosting)**: Added TSFRESH feature categories, feature selection methodology, philosophical comparison of approaches

#### 2. Paper Converted from Proposal to Final Format

**Changes Made**:
- Changed `\section*{Preliminary Results}` → `\section*{Results}`
- Updated text removing "preliminary" language
- Removed "Experimental Plan & Milestones" section (proposal artifact)
- Removed "Overall Project Risks & Mitigations" section (proposal artifact)
- Updated AI Use Disclosure to include both Claude Sonnet 4.5 and Claude Opus 4.5
- Added teammate names: Tristan (CNN), Bree (Transformer)

#### 3. Cross-Mission Generalization Section

**Added content for paper** (in `COPY_PASTE_THIS.txt` and `RNN_FINDINGS_CROSSMISSION.tex`):
- TESS vs Kepler observational differences table
- Cross-mission test setup description
- Results showing 100% recall but unknown precision on Kepler data
- Discussion of why complete cross-mission testing was not achieved
- Implications for future work

### Files Modified

| File | Change |
|------|--------|
| `term_project_files/Merged_Proposal_as_of_12.2.2025.tex` | Expanded RNN Related Work, removed proposal sections |
| `CLAUDE.md` | Added Dec 3 afternoon update |
| `PROGRESS_LOG_DEC_3_2025.md` | Added afternoon session details |
| `NEXT_SESSION_QUICKSTART.md` | Updated for next session |
| `Code/TRAINING_STATUS.md` | Updated to reflect current state |

### Backup Created

- `D:\CS_4280_Project_Backup\Merged_Proposal_as_of_12.2.2025_BACKUP_before_RNN_expansion.tex`

---

## Current State Summary

### What's Complete (RNN Component)
- [x] RNN Related Work - 2 full pages (6 foundational + 3 new papers)
- [x] RNN Results section - condensed with key tables
- [x] Cross-mission generalization analysis
- [x] SOFA refactoring - all Python files at pylint 10.00/10
- [x] Code README for TA

### What's Waiting On
- [ ] Teammates (Tristan/Bree) to update their sections
- [ ] Final Optuna optimization (Trial 0 hit AUC 0.916)
- [ ] Final model training with optimized hyperparameters
- [ ] Generate final figures for paper
- [ ] Create 20-second demo video
- [ ] Prepare presentation slides

### Best Results So Far

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
| ~~Dec 2-3~~ | Optuna overnight | ⚠️ Interrupted (1/30 trials) |
| ~~Dec 3~~ | Paper finalization (RNN sections) | ✅ Done |
| Dec 4-5 | Resume Optuna OR train final model | **NEXT** |
| Dec 5-6 | Generate figures + finalize paper | Pending |
| Dec 6-7 | Create slides + record 20s demo | Pending |
| Dec 7-8 | Practice presentation | Buffer |
| **Dec 9-11** | **Presentations** | |
| **Dec 18** | **Final submission deadline** | |

---

**Last Updated**: December 3, 2025, afternoon
**Status**: RNN paper sections complete, waiting for teammates, ready to resume training/optimization
