# Progress Log - December 18, 2025 (Final Submission Day)

## Session Summary

Final paper preparation session before submission deadline (tonight).

---

## Changes Made

### 1. Updated RNN Results with Final Model Performance

**Before:** Abstract and results showed outdated AUC 0.6947 (baseline) or 0.893 (intermediate)

**After:** Updated throughout with final results:
- **Test AUC: 0.9261** (92.61%)
- **Recall: 100%** (732/732 planets detected, zero false negatives)
- **Precision: 39.93%**
- **F1 Score: 0.5708**
- **Accuracy: 83.26%**
- **Improvement: +33.3%** over baseline

### 2. Added Kepler Fine-Tuning Experiment Results

Added new table and discussion showing:
- Fine-tuning on mixed TESS+Kepler data made performance **worse**
- Original model: 3.7% Kepler detection → Fine-tuned: 0%
- Demonstrates fundamental domain shift (2-min TESS vs 30-min Kepler cadence)

### 3. Updated Hyperparameters Throughout

| Parameter | Old Value | Corrected Value |
|-----------|-----------|-----------------|
| Hidden Units | 256 | **192** |
| K-means Clusters | 5 | **7** |
| Cluster Embed Dim | 32 | **64** |
| Parameters | 3.9M | **3.1M** |
| Dropout | 0.311 | **0.334** |
| Batch Size | 64 | **128** |

### 4. Added Comparative Analysis Table

New Table 13 comparing all three architectures:
| Model | AUC | Recall | Precision | F1 | Params |
|-------|-----|--------|-----------|-----|--------|
| BiLSTM+Cluster | **.926** | **100%** | 40% | .57 | 3.1M |
| CNN (dual) | .861 | 81% | 78% | .76 | 0.9M |
| Transformer | .677 | 67% | 64% | .66 | 0.1M |

### 5. Updated Datasets Table

Replaced outdated "Planned datasets" with actual datasets used:
- TESS Sector 1 Ground Truth (33,051 windows) - RNN
- Kepler MAST (795 LCs) - CNN
- TESS MAST (1,105 LCs) - CNN, Transformer
- TESS-Lilith simulated (2,821 LCs) - CNN
- Kepler cross-test (1,473 windows) - RNN

### 6. Updated Conclusion Section

- Replaced "[AUC TO BE UPDATED]" placeholder with actual 0.9261
- Added cross-mission limitation findings
- Completed comparative analysis text

### 7. File Renamed

`Merged_Proposal_as_of_12.2.2025.tex` → `Merged_Final_as_of_12.18.2025.tex`

### 8. Fixed Table 13 Width

Made comparative table fit column by:
- Added `\small` font
- Abbreviated headers (Rec., Prec.)
- Shortened model names

---

## Files Modified

- `term_project_files/Merged_Final_as_of_12.18.2025.tex` (main)
- `term_project_files/CS4280_Term_Paper_current/Merged_Final_as_of_12.18.2025.tex` (synced)
- `term_project_files/resourceFile.bib` (synced from team's version)
- `term_project_files/Merged_Final_as_of_12.18.2025.pdf` (compiled, 20 pages)

---

## Brianne's Transformer Section - Status Update

### Discovery (December 18, 2025)

Upon reviewing the paper and bib file, **Brianne appears to have already made her final updates** without notifying the team (Josh or Tristan). Evidence:

**Related Work Section (lines 243-294):**
- 10 papers cited with substantial discussion
- Vaswani, Morvan, Salinas (2023 & 2025), Wen, Astroconformer, Chen, BitNet, Pyraformer, etc.
- Bib file header updated to "Brianne Transformer section: COMPLETE with 10 papers"

**Methodology Section (lines 490-536):**
- Dataset: 2,100 train / 450 val / 450 test samples
- Architecture: 2M parameters, 4 transformer layers, 4 attention heads
- Training procedure with Optuna optimization
- Notes "dataset is in process of being increased to 3000"

**Results Section (lines 860-940):**
- Real metrics: **63.3% accuracy, 0.677 AUC, 0.658 F1**
- Per-class performance tables
- Scaling experiments (1.5M parameter model didn't improve)
- Zero-shot Kepler generalization (~50% accuracy)

**Conclusion (line 951):**
- Full paragraph summarizing Transformer findings

### Issue Found

**Wrong confusion matrix image path** (line 913):
```latex
\includegraphics[width=1.0\linewidth]{Images/RNN/confusion_matrix.png}
```
This points to Josh's RNN confusion matrix, not a Transformer-specific one. Need to verify if `Images/Transformer/confusion_matrix.png` exists or if she needs to generate one.

### Bib File Inconsistency

- **Header (line 5-7)**: Says "28 papers" and "Brianne Transformer section: COMPLETE with 10 papers"
- **Summary (line 402-410)**: Says "23 papers" and "Still needed: Transformer +2" (STALE - never updated)
- **Actual count**: 29 papers total (10 Transformer + 9 CNN + 9 RNN + 1 shared)

### Action Items

- [ ] Confirm with Brianne via Discord that she is done
- [ ] Verify if Transformer confusion matrix image exists or needs to be added
- [ ] Update bib file summary comments to match actual counts (cosmetic)

---

## Waiting On

### Brianne Confirmation

Waiting for Brianne to confirm via Discord that her sections are finalized. If she confirms:

---

## Final Checklist (for after Brianne's updates)

- [ ] All three architectures have final results
- [ ] Abstract reflects all final results
- [ ] Conclusion summarizes all findings
- [ ] Comparative Analysis table is complete
- [ ] No TBD or placeholder text remains
- [ ] All tables fit within column widths
- [ ] Bibliography compiles without errors
- [ ] PDF is 15-20 pages (currently 20)
- [ ] AI Use Disclosure is complete

---

## Current Paper Status

| Section | Status |
|---------|--------|
| Abstract | ✅ Updated with RNN final results |
| Introduction | ✅ Complete |
| Related Work | ✅ Complete (all 3 sections) |
| Datasets | ✅ Updated with actual data |
| Methodology - CNN | ✅ Complete |
| Methodology - RNN | ✅ Updated with final hyperparameters |
| Methodology - Transformer | ✅ Appears complete (awaiting confirmation) |
| Results - CNN | ✅ Complete |
| Results - RNN | ✅ Updated with final results |
| Results - Transformer | ⚠️ Complete but uses wrong confusion matrix image |
| Conclusion | ✅ Complete (all 3 architectures) |
| AI Disclosure | ✅ Complete |
| Bibliography | ✅ Compiles (29 papers - exceeds 27 target) |

---

## Deadline

**Tonight (December 18, 2025)** - Final submission

---

## FINAL SUBMISSION COMPLETE ✅

**Submitted**: December 18, 2025, 10:15 AM

### Final Deliverables

| File | Size | Description |
|------|------|-------------|
| `CS4280_Final_Report.pdf` | 1.2 MB | Team final report (AAAI format) |
| `CS4280_Final_Presentation.pptx` | 18.9 MB | Team presentation with embedded demo |
| `Code/README.md` | 15.6 KB | Comprehensive run instructions |
| `Code/models/best.pt` | 35.2 MB | Pre-trained BiLSTM model |
| `Code/*.py` | 97 KB | 7 Python scripts |

**Total submission**: `CS4280_Final_Submission.zip` (50.1 MB)

### Final Actions Taken

1. ✅ Updated README.md with comprehensive step-by-step instructions
2. ✅ Removed all personal names from submission files (team submission)
3. ✅ Replaced outdated report with final team report (`CS4280_Term_Paper-3.pdf`)
4. ✅ Added final presentation slides
5. ✅ Verified all professor requirements met
6. ✅ Uploaded to Canvas

### Professor Requirements Checklist

| Requirement | Status |
|-------------|--------|
| Final Report (AAAI PDF) | ✅ |
| Presentation Slides | ✅ |
| Demo Video (~20 sec) | ✅ (embedded in slides) |
| Project Code | ✅ |
| Run Instructions | ✅ |

---

## PROJECT COMPLETE

**Project Duration**: October 2025 - December 2025 (~3 months)
**Final Model AUC**: 0.9261 (92.61%)
**Key Achievement**: 100% recall - all planets detected

---

*Last updated: December 18, 2025, 10:20 AM*
*Status: SUBMITTED TO CANVAS - PROJECT COMPLETE*
