# Session Summary - November 12, 2025 (Continuation)
## RNN Midterm Presentation Final Updates

**Session Time**: November 12, 2025 (Evening Session)
**Status**: ✅ COMPLETE - Presentation ready for midterm

---

## Overview

This session continued from the earlier November 12 session to finalize the RNN midterm presentation slides. Major updates included fixing language (our → my), adding class imbalance discovery, and correcting the balanced data narrative.

---

## Changes Made

### 1. Language Correction: "Our" → "My"

**Issue**: Presentation used team language ("our approach", "the team") when this is Josh's individual work.

**Fix**: Updated all references to first-person singular:
- Changed "Our Approach" to "My Approach" on Slide 2
- Updated both markdown and PowerPoint generation script

**Files Modified**:
- `term_project_files/RNN_MIDTERM_SLIDES_CONDENSED.md`
- `term_project_files/create_rnn_presentation.py`

### 2. Added Class Imbalance Discovery to Slide 6

**Issue**: Slide 6 "Learning from Failure" didn't mention the critical discovery that imbalanced training data caused biased results.

**Fix**: Added section showing:
- Imbalanced data problem: 150 planets vs 505 non-planets (23% positive)
- High recall (88.67%) but low precision (38.27%)
- Too many false positives → Model biased toward negatives

**Before**:
```markdown
**Solution: Add Non-Planets**
✓ 100 planets + 300 non-planets
✓ AUC: 0.69 → Model learned to distinguish
```

**After**:
```markdown
**Solution: Add Non-Planets**
✓ 100 planets + 300 non-planets → AUC: 0.69
✓ Real TESS: Identified TIC 307210830 (prob: 0.5959)

**But: Imbalanced Data Problem**
❌ 150 planets vs 505 non-planets (23% positive)
❌ High recall (88.67%) but low precision (38.27%)
❌ Too many false positives → Model biased toward negatives
```

### 3. Corrected Balanced Data Definition and Failure on Slide 9

**Issue**: Misunderstood "balanced data" - it means 50/50 split (equal planets and non-planets), not just "better balance".

**Clarification**: Balanced data = 200 planets + 200 non-planets (50% positive), NOT 23% → 24.2%

**Fix**: Updated Slide 9 to show:
- Attempted true balanced synthetic data (50/50 split)
- **Failed badly**: AUC dropped to 0.45 on real TESS data (domain shift)
- Solution: Hybrid training (90% real + 10% synthetic) as compromise

**Before**:
```markdown
**Hybrid Training (Real + Synthetic)**
• 90% real TESS + 10% synthetic data
• Improves balance (23% → 24.2% positive)
• Maintains domain fidelity (avoids synthetic-only failure)
```

**After**:
```markdown
**Balanced Data Attempt Failed**
• Tried 50/50 balanced synthetic data (200 planets + 200 non-planets)
• AUC dropped to 0.45 on real TESS data (domain shift)

**Solution: Hybrid Training**
• Mix real TESS + synthetic (90/10 ratio)
• Better balance than 23% but maintains domain fidelity
```

---

## Final Slide Content

### Complete 9-Slide Presentation

**Slide 1: Related Work - Three NEW Papers**
- Full APA citations with H5 indices (all >100)
- Speiser 2020 (H5: 399), Vu 2024 (H5: 234), Ding 2024 (H5: 151)

**Slide 2: Why BiLSTM + Clustering?**
- From the Papers: Key insights from each paper
- **My Approach**: BiLSTM + K-means clustering on BLS features

**Slide 3: Methodology**
- Data: 655 windows (150 planets, 505 non-planets)
- Features: Period, depth, duration, BLS power
- Architecture: K-means (5 clusters), 4-layer BiLSTM, cluster embeddings
- Image: preprocessing_pipeline.png

**Slide 4: BiLSTM Architecture**
- Architecture diagram
- Input → BiLSTM → Cluster Embedding → FC → Output
- ~2.1M parameters

**Slide 5: Results**
- AUC: 75.72%
- Recall: 88.67% | Precision: 38.27%
- Tested on 100 confirmed exoplanet systems
- Images: metrics_bar_chart.png, confusion_matrix.png

**Slide 6: Learning from Failure**
- 100 planets only → Failed (100% false positives)
- 100 planets + 300 non-planets → AUC 0.69
- Real TESS validation → Identified TIC 307210830
- **Discovery**: Imbalanced data (150 vs 505) causes bias

**Slide 7: Optuna Optimization**
- Parameter changes: 4 layers, batch 128, LR 2.25e-4
- Result: AUC 0.69 → 0.76 (+9%)
- Image: model_progression.png

**Slide 8: Demo**
- Video: demo_video.mp4
- Model identifying TIC 307210830 (L 98-59 multi-planet system)

**Slide 9: What's Next?**
- Balanced data (50/50) attempt failed → AUC 0.45
- Solution: Hybrid training (90% real + 10% synthetic)
- Cross-mission testing: TESS → Kepler

---

## Files Modified

### Presentation Files
1. **term_project_files/RNN_MIDTERM_SLIDES_CONDENSED.md**
   - Updated Slide 2: "My Approach" (not "Our Approach")
   - Updated Slide 6: Added imbalanced data discovery
   - Updated Slide 9: Corrected balanced data definition and failure

2. **term_project_files/create_rnn_presentation.py**
   - Updated all three slides to match markdown
   - Adjusted font sizes to fit additional content
   - Regenerated PowerPoint 3 times during session

3. **term_project_files/RNN_MIDTERM_PRESENTATION.pptx**
   - Final PowerPoint with all corrections
   - All 9 slides with embedded images
   - Ready to present or copy to SharePoint

---

## Key Learnings Documented

### Scientific Narrative
The presentation now tells a complete story of iterative scientific discovery:

1. **Initial Failure**: 100 planets only → predicted everything as planet
2. **Baseline Success**: Added non-planets → AUC 0.69
3. **Real-World Validation**: Identified confirmed exoplanet TIC 307210830
4. **Problem Discovery**: Imbalanced data causes bias (high recall, low precision)
5. **Failed Solution**: Balanced synthetic data → AUC 0.45 (domain shift)
6. **Optimization**: Optuna improved to AUC 0.76
7. **Proposed Solution**: Hybrid training (90/10) to balance domain fidelity and class balance
8. **Future Work**: Cross-mission testing to verify generalization

### Balanced Data Definition
- **Balanced**: Equal numbers of planets and non-planets (50/50 split)
- Example: 200 planets + 200 non-planets = 400 total (50% positive)
- **Imbalanced**: Unequal distribution
- Example: 150 planets + 505 non-planets = 655 total (23% positive)

### Domain Shift Problem
- Pure synthetic balanced data achieved AUC 1.0 in training
- But AUC 0.45 on real TESS data (worse than random!)
- Root cause: Synthetic transit depth 8× shallower than real data
- Solution: Hybrid approach maintains real data characteristics while improving balance

---

## Individual Paper Slides

The 3 individual paper slides (PAPER_SLIDES_TO_ADD.pptx) were created in the earlier session:
- Slide 2: Machine Learning for Cluster Analysis (Speiser 2020)
- Slide 3: LSTM for Time Series Patterns (Vu 2024)
- Slide 4: LSTM for Astronomical Photometry (Ding 2024)

Each has:
- Key Innovation box (light gray)
- Key Takeaway box (dark blue)
- Architecture and Results sections
- Full APA citation

---

## Presentation Materials Checklist

✅ **RNN_MIDTERM_PRESENTATION.pptx** - Main 9-slide presentation
✅ **PAPER_SLIDES_TO_ADD.pptx** - 3 individual paper slides
✅ **RNN_SPEAKING_SCRIPT.md** - 7-minute speaking script
✅ **demo_video.mp4** - 20-second demonstration video
✅ **5 visualizations** at 300 DPI:
   - metrics_bar_chart.png
   - confusion_matrix.png
   - model_progression.png
   - preprocessing_pipeline.png
   - bilstm_architecture.png

---

## Next Steps (Weekend Work)

User mentioned continuing this weekend. Potential tasks:
1. Practice presentation with speaking script (7 minutes)
2. Test hybrid training models (90/10 and 75/25 ratios)
3. Begin cross-mission testing (TESS → Kepler)
4. Prepare for midterm presentation (November 13, 2025)

---

## Technical Details

### Model Performance Summary
- **Baseline (imbalanced)**: AUC 0.69, F1 0.34
- **Optimized (imbalanced)**: AUC 0.76, Recall 0.89, Precision 0.38
- **Balanced synthetic**: AUC 0.45 on real data (FAILED)
- **Hybrid (planned)**: Expected AUC 0.79-0.82

### Dataset Details
- **Current training**: 655 windows (150 planets, 505 non-planets, 23% positive)
- **Balanced synthetic**: 400 light curves (200 planets, 200 non-planets, 50% positive)
- **Hybrid 90**: 727 windows (~24% positive, maintains domain fidelity)
- **Hybrid 75**: More synthetic, better balance but more domain shift risk

---

## Time Investment

**Session Duration**: ~45 minutes
**PowerPoint Regenerations**: 3
**Files Modified**: 3
**Documentation Created**: 1 (this file)

---

## Status: Ready for Midterm

All presentation materials are complete, corrected, and ready for the November 13, 2025 midterm presentation. The narrative clearly shows scientific problem-solving and iterative improvement.

**Next Session**: Weekend work on hybrid training and cross-mission testing

---

*Generated: November 12, 2025 (Evening)*
*Session Focus: Presentation finalization and corrections*
*Status: ✅ COMPLETE*
