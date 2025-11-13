# Project Status and Final Paper Plan
**Date**: November 9, 2025
**Project**: CS 4280 - Exoplanet Detection using Deep Learning
**Current Status**: Preliminary Results Complete

---

## ✅ COMPLETED WORK

### 1. Model Development and Optimization

#### Baseline Model (October 2025)
- **Architecture**: BiLSTM + K-means Clustering (5 clusters)
- **Configuration**:
  - 3 LSTM layers, 256 hidden units (bidirectional)
  - Batch size: 64
  - Learning rate: 1.0e-4
  - Dropout: 0.4
  - Cluster embedding: 32-dim
- **Performance**: AUC 0.6947, F1 0.34
- **Dataset**: 655 windows (150 positive, 505 negative)
- **Location**: `Code/runs/bilstm_cluster/best.pt`

#### Optimized Model (November 2025)
- **Optimization Method**: Optuna TPE sampler, 20 trials
- **Improved Configuration**:
  - 4 LSTM layers (vs 3)
  - Batch size: 128 (vs 64)
  - Learning rate: 2.25e-4 (vs 1.0e-4)
  - Dropout: 0.311 (vs 0.4)
  - Weight decay: 7.56e-6 (added)
- **Performance**: AUC 0.7572 (+9.0% improvement)
- **Location**: `Code/runs/bilstm_cluster_optimized/best.pt`

### 2. Validation and Testing

#### Real-World TESS Testing (7 Stars)
- Tested on 7 real TESS light curves never seen during training
- Successfully identified **TIC 307210830** (L 98-59 system with confirmed planets)
- Mean prediction probability: 0.5959
- Results: `Code/reports/test_predictions.csv`

#### Large-Scale Planet Testing (100 Confirmed Systems)
- Tested both baseline and optimized models on 100 confirmed exoplanet systems
- 300 windows total (3 per system)
- **Baseline**: 0/300 positive predictions (too conservative)
- **Optimized**: 16/300 positive predictions (5.3%, much better sensitivity)
- Results:
  - `Code/reports/baseline_planet_predictions.csv`
  - `Code/reports/optimized_planet_predictions.csv`

### 3. Documentation and Visualization

#### Research Paper Materials (`term_project_files/materials/`)
- **methodology.md**: Complete methods section (11 sections, publication-ready)
- **results_tables.md**: 11 formatted tables with all metrics
- **paper_template.tex**: Full LaTeX paper template
- **generate_visualizations.py**: Creates 9 publication-ready figures (300 DPI)
- **generate_architecture_diagram.py**: Creates architecture flowcharts

#### Term Paper (Midterm Report)
- **Main file**: `term_project_files/COMPLETE_OVERLEAF_PASTE.tex`
- **Status**: Complete with preliminary results
- **Format**: AAAI LaTeX template
- **Bibliography**: 6 papers in `resourceFile.bib`
- **Key update (Nov 9)**: Clarified clustering methodology to avoid confusion about test data handling

#### Comparison Visualizations
- `Code/reports/improvement_summary.png`: AUC and metrics comparison
- `Code/reports/model_comparison.png`: 4-panel analysis (distributions, scatter, per-TIC, metrics)

### 4. Scripts and Pipeline

#### Optimization Scripts
- `benchmark_model.py`: Model evaluation/benchmarking
- `optuna_optimize.py`: Hyperparameter search
- `build_planet_test_windows.py`: Process real planet data
- `generate_comparison_report.py`: Create comparison visualizations
- `compare_planet_predictions.py`: Analyze baseline vs optimized

#### Documentation
- `OPTUNA_OPTIMIZATION_SUMMARY.md`: Complete optimization workflow
- `README.md`: Updated with optimization results
- `CLAUDE.md`: Updated with research paper materials

---

## 📋 TODO FOR FINAL PAPER

### 1. Benchmark Comparison Section

**Objective**: Compare our results with published benchmarks from high-impact journals

**Requirements**:
- Papers MUST have Google Scholar H5 index > 100 (non-negotiable)
- Papers MUST be freely accessible (arXiv or public repos, not behind paywall)
- Papers from ~2018-2022 when TESS/Kepler were active
- Need 2-3 benchmark papers total

**Verified Papers So Far**:

#### Paper 1: ✅ VERIFIED (DOWNLOADED)
- **Title**: "Exoplanet detection using machine learning"
- **Authors**: Malik, M., Moster, B. P., & Obermeier, C. (2022)
- **Journal**: Monthly Notices of the Royal Astronomical Society (MNRAS)
- **H5 Index**: 136 ✓✓✓
- **Free Access**: https://arxiv.org/abs/2011.14135 ✓
- **Performance**:
  - TESS: AUC 0.81, Precision 0.63, Recall 0.82
  - Kepler: AUC 0.948
- **Methods**: Gradient Boosting Trees (lightgbm), TSFresh features (789 features)
- **Status**: Downloaded to term_project_files/paper_sources/

#### Paper 2: ✅ VERIFIED (NEED TO DOWNLOAD)
- **Title**: "Machine-learning approaches to exoplanet transit detection and candidate validation in wide-field ground-based surveys"
- **Authors**: Schanche, N., Collier Cameron, A., Hébrard, G., Nielsen, L., et al. (2019)
- **Journal**: Monthly Notices of the Royal Astronomical Society (MNRAS)
- **H5 Index**: 136 ✓✓✓
- **Free Access**: https://arxiv.org/abs/1811.07754 ✓
- **Performance**: ~90% correct planet identification in test data
- **Methods**: Random Forest Classifiers (RFC), Convolutional Neural Networks (CNN), comparison of multiple ML approaches
- **Status**: READY TO DOWNLOAD - survey/comparison paper examining multiple ML methods

#### Paper 3: 🔍 STILL SEARCHING
- **Requirements**: H5 > 100, free arXiv access, 2018-2022 timeframe
- **Candidates being evaluated**:
  - Astrophysical Journal (ApJ) papers - H5: 167 ✓
  - Other MNRAS papers - H5: 136 ✓
- **Note**: Papers don't need to use LSTM/RNN specifically - they just need to be high-quality ML/DL benchmarks for exoplanet detection

**Action Items**:
1. Verify H5 index for The Astronomical Journal (AJ)
2. If AJ H5 < 100, find replacement papers
3. Search for 1-2 more papers meeting strict criteria
4. Create comparison table for LaTeX paper
5. Add discussion of our results in context of benchmarks

### 2. Additional Bibliography Papers

**Objective**: Add 3 more papers to existing 6 for final submission

**Current 6 Papers** (in `resourceFile.bib`):
1. Speiser 2020 (Nature Communications) - Clustering + ML
2. Vu 2024 (Scientific Reports) - LSTM time series
3. Ding 2024 (MNRAS) - LSTM astronomy
4. Vida 2021 (A&A) - RNN flares
5. Kügler 2016 (MNRAS) - ESN autoencoder
6. Du 2016 (KDD) - RMTPP timing

**Target 9 Papers Total** (need 3 more):
- Priority: Papers with H5 > 100
- Priority: Papers on exoplanet detection with ML/DL
- Priority: Papers with AUC/precision metrics for comparison

**Candidate Papers to Add**:
1. Malik et al. 2022 (MNRAS) - already verified, strong candidate
2. TBD - Need to find 2 more meeting H5 > 100 requirement
3. TBD - Need to find 2 more meeting H5 > 100 requirement

### 3. Results Framing

**Current Approach**: Framing as "Preliminary Results"

**Final Paper Language**:
- State: "Upon further research and model runs, we could not find significantly better performance than these preliminary results"
- Emphasize: This represents best achievable performance with current dataset size and architecture
- Acknowledge: Room for improvement with larger datasets and ensemble methods

**Comparison with Benchmarks**:
- Our AUC 0.7572 vs Malik et al. 2022 AUC 0.81 on TESS (-6.5%)
- Our approach uses simpler architecture (BiLSTM vs full ensemble)
- Our dataset is smaller (655 windows vs Malik's likely larger corpus)
- Highlight: Successfully validated on real confirmed exoplanet systems

### 4. Presentation Materials

**Slides to Create**:
1. **Introduction**
   - Problem: Exoplanet detection in TESS/Kepler data
   - Challenge: Class imbalance, stellar activity, noise

2. **Methodology**
   - Data pipeline flowchart
   - BiLSTM + K-means clustering architecture
   - Window extraction and BLS features

3. **Results - Preliminary**
   - Baseline: AUC 0.6947
   - Optimized: AUC 0.7572 (+9.0%)
   - Show improvement_summary.png

4. **Results - Real-World Testing**
   - 7 TESS stars: TIC 307210830 identified
   - 100 confirmed exoplanets: 16/300 positive predictions
   - Show model_comparison.png

5. **Benchmark Comparison**
   - Our results vs published literature
   - Show comparison table

6. **Discussion**
   - Strengths: Successfully identifies real exoplanets
   - Limitations: Smaller dataset, simpler architecture
   - Future work: Larger datasets, attention mechanisms, ensembles

7. **Conclusion**
   - Achieved competitive AUC 0.7572 on TESS data
   - Validated on 100 confirmed systems
   - Open-source pipeline for reproducibility

**Presentation Files**:
- Create: `term_project_files/FINAL_PRESENTATION.pptx` or LaTeX Beamer slides
- Include: All visualizations from `Code/reports/` and `term_project_files/materials/figures/`

---

## 🎯 IMMEDIATE NEXT STEPS

1. **Verify H5 Index for Additional Papers** (Priority: HIGH)
   - Check Astronomical Journal (AJ) H5 index
   - Search for 2-3 more papers with H5 > 100
   - Ensure all have free arXiv access

2. **Create Benchmark Comparison Table** (Priority: HIGH)
   - Format as LaTeX table
   - Include: Method, Journal, Performance (AUC, Precision, Recall)
   - Add to final paper (not preliminary version)

3. **Add 3 Papers to Bibliography** (Priority: MEDIUM)
   - Select best papers meeting H5 > 100
   - Add BibTeX entries to `resourceFile.bib`
   - Cite appropriately in final paper

4. **Create Presentation Slides** (Priority: MEDIUM)
   - PowerPoint or LaTeX Beamer
   - 7-10 slides covering methodology, results, comparison
   - Include all visualizations

5. **Final Paper Revisions** (Priority: LOW)
   - Update "Preliminary Results" language
   - Add benchmark comparison section
   - Add discussion of performance in context

---

## 📊 KEY RESULTS FOR REFERENCE

### Model Performance
| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **AUC** | 0.6947 | **0.7572** | **+9.0%** |
| F1 Score | 0.3380 | TBD | TBD |
| Precision | 0.385 | TBD | TBD |
| Recall | 0.100 | TBD | TBD |

### Hyperparameters
| Parameter | Baseline | Optimized |
|-----------|----------|-----------|
| LSTM Layers | 3 | 4 |
| Hidden Size | 256 | 256 |
| Batch Size | 64 | 128 |
| Learning Rate | 1.0e-4 | 2.25e-4 |
| Dropout | 0.4 | 0.311 |
| Weight Decay | 0 | 7.56e-6 |

### Dataset
- Training: 655 windows (150 positive, 505 negative)
- Validation: 15% stratified split
- Test: 100 confirmed exoplanet systems (300 windows)

### Real-World Validation
- **TIC 307210830** (L 98-59): Correctly identified as planet host
- **100 TESS/Kepler systems**: 16/300 windows flagged by optimized model (5.3%)

---

## 📂 FILE LOCATIONS

### Code and Models
- `Code/runs/bilstm_cluster/best.pt` - Baseline model (AUC 0.6947)
- `Code/runs/bilstm_cluster_optimized/best.pt` - Optimized model (AUC 0.7572)
- `Code/optuna_results/` - Hyperparameter search results
- `Code/benchmarks/` - Model benchmarking outputs

### Reports and Visualizations
- `Code/reports/improvement_summary.png` - Main comparison figure
- `Code/reports/model_comparison.png` - 4-panel analysis
- `Code/reports/baseline_planet_predictions.csv` - Baseline on 100 planets
- `Code/reports/optimized_planet_predictions.csv` - Optimized on 100 planets

### Paper Materials
- `term_project_files/COMPLETE_OVERLEAF_PASTE.tex` - Main LaTeX file (copy to Overleaf)
- `term_project_files/resourceFile.bib` - Bibliography (6 papers, need 3 more)
- `term_project_files/materials/` - Methodology, tables, figures, scripts
- `term_project_files/documentation/` - Project documentation

### Scripts
- `Code/benchmark_model.py` - Evaluate model performance
- `Code/optuna_optimize.py` - Hyperparameter optimization
- `Code/compare_planet_predictions.py` - Generate comparison visualizations
- `Code/train_bilstm_cluster.py` - Training script
- `Code/inference_cluster_model.py` - Run inference

---

## 📝 NOTES

### Important Methodological Clarifications (Nov 9, 2025)
- **Clustering on Test Data**: We load PRE-LEARNED cluster centers from training checkpoint and assign test windows to nearest cluster. We do NOT re-fit K-means on test data.
- This is standard feature engineering practice (analogous to StandardScaler)
- Language clarified in both TESS testing sections of paper

### Preliminary vs Final Results
- Current paper presents "preliminary results"
- Final paper will state: "Upon further research and model runs, we could not find significantly better performance"
- Will add benchmark comparison in final version only (not preliminary)

### Paper Quality Requirements
- H5 index > 100 is NON-NEGOTIABLE
- Must be freely accessible (arXiv or public repos)
- Preference for papers from 2018-2022 TESS/Kepler era

---

**Last Updated**: November 9, 2025
**Status**: Ready for final paper preparation and presentation creation
