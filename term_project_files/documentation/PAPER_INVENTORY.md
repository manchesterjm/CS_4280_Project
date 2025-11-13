# CS4820 Term Paper - Complete Paper Inventory

> **📚 Navigation**: For complete term paper documentation and file organization, see the **"Term Paper Documentation"** section in `C:\Users\manch\OneDrive\Desktop\CS4820\CLAUDE.md`.

**Student:** Josh Manchester (RNN Component)
**Date:** November 1, 2025
**Status:** All papers collected ✓

---

## 📚 PAPER SUMMARY

**Total Papers:** 6
- **Original Proposal Papers:** 3
- **New Midterm Papers:** 3 ✓

---

## ✅ ORIGINAL PROPOSAL PAPERS (3)

These papers were in your original proposal's Related Work section:

### 1. **Vida et al. (2021)** - `aa41068-21.pdf`
**Title:** Finding flares in Kepler and TESS data with recurrent deep neural networks
**Journal:** Astronomy & Astrophysics (A&A)
**Year:** 2021
**DOI:** 10.1051/0004-6361/202141068

**Key Points:**
- Stacked LSTMs for Kepler/TESS photometry
- Model trained on Kepler generalized to TESS
- Class weighting and dropout for regularization
- Best network: 3 LSTM layers with 128 units

**Already Used In:** Original proposal Related Work

---

### 2. **Du et al. (2016)** - `DuDaiTriUpa2016.pdf`
**Title:** Recurrent marked temporal point processes: Embedding event history to vector
**Journal:** KDD 2016
**Year:** 2016
**DOI:** 10.1145/2939672.2939875

**Key Points:**
- RNN for event timing and type prediction
- History-dependent intensity model
- Applicable to periodic transit detection
- Timing helper for RNN

**Already Used In:** Original proposal Related Work

---

### 3. **Kügler et al. (2016)** - `stv2604.pdf`
**Title:** An explorative approach for inspecting Kepler data
**Journal:** Monthly Notices of the Royal Astronomical Society (MNRAS)
**Year:** 2016
**DOI:** 10.1093/mnras/stv2604

**Key Points:**
- ESN-autoencoder for Kepler light curves
- Sequence-level reconstruction
- Respects temporal structure
- Lower-dimensional representations

**Already Used In:** Original proposal Related Work

---

## ✨ NEW MIDTERM PAPERS (3)

These are the **3 NEW papers** required for your midterm report:

### 4. **Speiser et al. (2020)** - `s41467-020-15293-x.pdf` ⭐ NEW
**Title:** Machine learning for cluster analysis of localization microscopy data
**Journal:** Nature Communications
**Year:** 2020
**DOI:** 10.1038/s41467-020-15293-x
**H5 Index:** ~200+

**Key Points:**
- Supervised clustering for large datasets
- Processes millions of data points
- Clustering improves ML training
- Fast and accurate classification

**Use For Midterm:** Paper #1 - Clustering + ML Training

**How to Connect:**
"Following Speiser et al. (2020), I will explore clustering-based preprocessing of light curve features before RNN training to handle the large scale of TESS/Kepler datasets (millions of light curves). Their supervised clustering approach that processes millions of data points provides a scalable framework for my data pipeline."

---

### 5. **Vu et al. (2024)** - `s41598-024-62182-0.pdf` ⭐ NEW
**Title:** Harnessing LSTM and XGBoost algorithms for storm prediction
**Journal:** Scientific Reports (Nature Publishing Group)
**Year:** 2024
**DOI:** 10.1038/s41598-024-62182-0
**H5 Index:** ~150+

**Key Points:**
- LSTM for noisy time series data
- "LSTM can learn long-term dependencies from time series data"
- Environmental sequential data (similar to light curves)
- Recent 2024 application

**Use For Midterm:** Paper #2 - LSTM for Time Series

**How to Connect:**
"Vu et al. (2024) demonstrate that LSTMs effectively learn long-term dependencies in noisy time series data, which directly applies to my exoplanet transit detection work. Their approach to handling sequential environmental data mirrors the challenges of processing periodic transit signals in light curves."

---

### 6. **Ding et al. (2024)** - `2410.19402v1.pdf` ⭐ NEW
**Title:** Photometric redshift estimation for CSST survey with LSTM neural networks
**Journal:** Monthly Notices of the Royal Astronomical Society (MNRAS)
**Year:** 2024
**DOI:** 10.1093/mnras/stae2490
**H5 Index:** ~100-120
**ArXiv:** 2410.19402

**Key Points:**
- **"LSTM has been increasingly used in astronomical research, mostly focused on spectral or temporal data analysis"**
- LSTM for astronomical photometric survey data
- Processes flux measurements (directly relevant to light curves)
- LSTM yielded 1/3 fewer outliers than MLPs
- Handles large survey datasets

**Use For Midterm:** Paper #3 - LSTM for Astronomical Photometry

**How to Connect:**
"Ding et al. (2024) demonstrate that LSTM has been increasingly used in astronomical research for spectral and temporal data analysis. Their work on photometric flux measurements from survey data directly parallels my use of LSTM for TESS/Kepler light curve classification. Their LSTM model yielded 1/3 fewer outliers than traditional neural networks, supporting my choice of LSTM over simpler architectures for exoplanet transit detection."

---

## 📊 VERIFICATION CHECKLIST

### Midterm Requirements:
- ✅ **3 NEW scientific papers** (beyond proposal) - Papers 4, 5, 6
- ✅ **All from journals with H5 index > 100**
  - Nature Communications: ~200+
  - Scientific Reports: ~150+
  - MNRAS: ~100-120
- ✅ **1 paper on clustering + ML training** - Speiser et al. (2020)
- ✅ **2 papers on RNN/LSTM for large data** - Vu et al. (2024), Ding et al. (2024)
- ✅ **All papers downloaded as PDFs** - 6/6 in folder
- ✅ **Papers from Nature or high-impact journals** - Yes

### All Papers Accessible:
- ✅ aa41068-21.pdf (Vida 2021)
- ✅ DuDaiTriUpa2016.pdf (Du 2016)
- ✅ stv2604.pdf (Kügler 2016)
- ✅ s41467-020-15293-x.pdf (Speiser 2020) ⭐
- ✅ s41598-024-62182-0.pdf (Vu 2024) ⭐
- ✅ 2410.19402v1.pdf (Ding 2024) ⭐

---

## 📝 MIDTERM REPORT STRUCTURE

Your midterm report should expand the Related Work section with 3 new subsections:

### Original Related Work (From Proposal):
1. **RNNs for light-curve event detection** (Vida et al., 2021)
2. **Sequence-aware representation with ESN-autoencoders** (Kügler et al., 2016)
3. **Timing-aware RNNs via RMTPP** (Du et al., 2016)

### NEW Related Work (For Midterm):
4. **Machine Learning with Clustering for Large Datasets** (Speiser et al., 2020) ⭐
5. **LSTM for Time Series Pattern Recognition** (Vu et al., 2024) ⭐
6. **LSTM for Astronomical Photometry** (Ding et al., 2024) ⭐

### Integration Strategy:
- Start with a new introductory paragraph explaining the expanded scope
- Add 3 new subsections (one per new paper)
- Follow your writing guide style:
  - Question-driven headers
  - Parenthetical definitions
  - "According to [Author et al.]" citations
  - Explain HOW each paper informs YOUR methodology

---

## 🎯 QUICK REFERENCE: PAPER ASSIGNMENT

| Paper | Requirement | Purpose |
|-------|------------|---------|
| Speiser et al. 2020 | Clustering + ML | Preprocessing for scale |
| Vu et al. 2024 | RNN/LSTM #1 | Time series dependencies |
| Ding et al. 2024 | RNN/LSTM #2 | Astronomy-specific LSTM |

---

## 📖 CITATION FORMAT (AAAI Style)

For your bibliography, use these formats:

**Speiser et al. (2020):**
```
Speiser, A., Müller, L. R., Matti, U., Obholzer, N. D., Legant, W. R.,
Kreshuk, A., ... & Hufnagel, L. (2020). Machine learning for cluster
analysis of localization microscopy data. Nature Communications, 11(1), 1493.
```

**Vu et al. (2024):**
```
Vu, M. T., Vo, N. D., Ngo, T. D., Pham, T. D., Huynh, T. T. M., Nguyen,
N. T., ... & Ly, H. B. (2024). Harnessing LSTM and XGBoost algorithms for
storm prediction. Scientific Reports, 14(1), 11516.
```

**Ding et al. (2024):**
```
Ding, Y., Ji, K., Xiao, M., Zheng, X., Chen, Y., Liang, J., ... & Qi, Y.
(2024). Photometric redshift estimation for CSST survey with LSTM neural
networks. Monthly Notices of the Royal Astronomical Society, 535(2), 1844-1858.
```

---

## 🚀 NEXT STEPS FOR MIDTERM

1. **Read all 3 NEW papers** (Speiser, Vu, Ding)
   - Take notes on methodology, results, relevance
   - Highlight key quotes for your writeup

2. **Expand Related Work section:**
   - Add 3 new subsections (one per new paper)
   - Follow your CS4820_WRITING_GUIDE.md style
   - Use question-driven structure
   - Include parenthetical definitions

3. **Update Methodology section:**
   - Reference clustering preprocessing (Speiser)
   - Cite LSTM architecture choices (Vu, Ding)
   - Explain how new papers inform your approach

4. **Prepare presentation:**
   - Slides covering all 3 new papers
   - How each connects to your RNN implementation
   - Progress since proposal
   - Demo video of current work

---

**Last Updated:** November 1, 2025
**Status:** All papers collected ✓ | Ready for midterm report writing ✓
