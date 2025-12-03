# Documentation Standard

**Purpose**: This document defines what Claude Code must document when the user says "document everything we did, where we are at, and what are our future plans."

---

## Required Documentation

When asked to document a session, Claude Code MUST create/update these files:

### 1. Progress Log (REQUIRED)
**File**: `PROGRESS_LOG_[DATE].md` (e.g., `PROGRESS_LOG_NOV_29_2025.md`)

Must include:

#### A. Session Summary
- Date/Time
- Overall status (one line)
- Key finding/accomplishment (one line)

#### B. What We Did Today
For EACH task completed:
- Task name and status (Completed/In Progress/Failed)
- Command(s) used (copy-pasteable)
- Results with actual numbers (not estimates)
- Any errors encountered and how they were resolved

#### C. Current State
- Dataset location and size
- Best model location and metrics (AUC, F1, etc.)
- Any running processes

#### D. Future Plans
- Immediate (this session or next)
- Short-term (next few sessions)
- Long-term (project goals)

#### E. Files Created/Modified
- List ALL files created or modified during the session
- Include full paths

---

### 2. Quick Start Guide (REQUIRED)
**File**: `NEXT_SESSION_QUICKSTART.md`

Must include:
- Last session date
- Key findings summary table
- Current state (dataset, best model)
- Recommended next command (copy-pasteable)
- Key parameters with verified values
- Data locations

---

### 3. CLAUDE.md Updates (REQUIRED)
**File**: `CLAUDE.md`

Must update:
- Add dated UPDATE entry at top with key findings
- Update any commands that changed
- Update expected training times if they changed
- Add new benchmark results if applicable

---

## What Counts as "Results with Actual Numbers"

### DO Document:
- Benchmark results: "Batch size 64: 3.14 min/epoch"
- Model metrics: "AUC: 0.9159, F1: 0.5673"
- Dataset sizes: "26,472 training windows"
- Training times: "Completed in 3.2 hours"
- Error messages: exact error text

### DON'T Document:
- Vague estimates: "training takes a while"
- Unverified claims: "should be faster"
- Assumptions without testing

---

## Example: Minimum Viable Documentation

For a session where we ran batch size benchmarks:

```markdown
# Progress Log - November 29, 2025

## Session Summary
**Date**: 2025-11-29
**Status**: Batch size benchmark completed
**Key Finding**: Batch size 64 optimal (3.14 min/epoch)

## What We Did

### 1. Batch Size Benchmark (Completed)
**Command**:
```powershell
python benchmark_batch_sizes.py
```

**Results**:
| Batch Size | Time/Epoch |
|------------|------------|
| 64 | 3.14 min |
| 128 | 24.46 min |

## Current State
- Dataset: E:\...\windows_sector1_full\ (26,472 train)
- Best model: runs\sector1_groundtruth_overnight\ (AUC 0.9159)

## Future Plans
- Immediate: Retrain with batch size 64
- Short-term: Test on held-out set

## Files Created
- Code/benchmark_batch_sizes.py
- Code/BATCH_SIZE_BENCHMARK_RESULTS.md
```

---

## Checklist Before Ending Session

When the user says "document everything", verify:

- [ ] Created `PROGRESS_LOG_[DATE].md` with all sections
- [ ] Updated `NEXT_SESSION_QUICKSTART.md` with current state
- [ ] Added UPDATE entry to `CLAUDE.md`
- [ ] All commands are copy-pasteable
- [ ] All results have actual numbers (not estimates)
- [ ] All file paths are correct
- [ ] Future plans are specific and actionable

---

## Files That Should NOT Be Modified

Unless explicitly requested:
- Term paper files in `term_project_files/`
- Archive files in `Code/Archive/`
- Bibliography files (`.bib`)
- Git configuration

---

**Created**: November 29, 2025
**Purpose**: Ensure consistent, complete documentation across sessions
