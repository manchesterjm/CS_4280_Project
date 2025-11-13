# RNN Presentation Speaking Script
## Josh Manchester - 7-Minute Presentation

---

## SLIDE 1: Related Work - Three NEW Papers

**WHAT TO SAY (~45 seconds):**

"For my related work, I selected three new papers from high-impact journals—all with H5 indices well above 100, as required.

First, Speiser 2020 from Nature Communications showed that combining K-means clustering with machine learning improves both accuracy and speed on large datasets.

Second, Vu 2024 from Scientific Reports demonstrated that LSTM networks excel at capturing long-term dependencies in noisy time series data.

Third, Ding 2024 from MNRAS—the top astronomy journal—showed that LSTM reduced outliers by 33% compared to traditional neural networks on real astronomical data.

These three papers directly informed my architecture choices."

---

## SLIDE 2: Why BiLSTM + Clustering?

**WHAT TO SAY (~30 seconds):**

"Based on these papers, I chose a BiLSTM with K-means clustering architecture.

The clustering approach comes from Speiser—it lets my model learn specialized patterns for different types of stars and signals.

The BiLSTM comes from Vu and Ding—it's designed to handle the noisy, irregular time series data we get from TESS light curves.

This combination gave me the best of both worlds: pattern specialization and temporal modeling."

---

## SLIDE 3: Methodology

**WHAT TO SAY (~45 seconds):**

"My dataset has 655 windows—150 from confirmed planets and 505 from non-planets like flares and noise. This 23% positive rate reflects the real imbalance in exoplanet detection.

I use Box Least Squares to extract four features: period, depth, duration, and BLS power. These get clustered into 5 groups by K-means.

The architecture has 4 bidirectional LSTM layers with 256 hidden units. Each window also gets a 32-dimensional cluster embedding that tells the model what type of signal it's looking at.

This diagram shows the preprocessing: raw light curve, cleaning, phase folding, feature extraction, and window extraction."

---

## SLIDE 4: BiLSTM Architecture

**WHAT TO SAY (~30 seconds):**

"Here's the full architecture. The 2048-point window goes through four BiLSTM layers. At the same time, the cluster embedding is computed from the BLS features.

These get concatenated and pass through fully connected layers with dropout for regularization.

The final sigmoid gives us a probability between 0 and 1—planet or not.

The model has about 2.1 million parameters and trains in about 25 seconds per epoch on GPU."

---

## SLIDE 5: Results

**WHAT TO SAY (~45 seconds):**

"My optimized model achieved an AUC of 75.72%.

The bar chart shows all five metrics. The most important is recall at 88.67%—that means I'm finding almost 9 out of 10 real planets. Precision is lower at 38%, which means some false positives, but that's acceptable in astronomy where we verify candidates later.

The confusion matrix shows performance on the test set. Out of 98 windows, I correctly identified 20 true positives and 44 true negatives, with 32 false positives and only 2 false negatives.

I validated this on 100 confirmed exoplanet systems from TESS, and the model correctly ranked known multi-planet systems highest."

---

## SLIDE 6: Learning from Failure

**WHAT TO SAY (~45 seconds):**

"I want to talk about an important failure that taught me about class imbalance.

Initially, I trained on 100 confirmed planet light curves only—no non-planet examples.

The result was catastrophic. The model predicted everything as a planet. 100% false positive rate. Why? Because it learned 'all light curves contain planets'—that was the only pattern it had ever seen.

The solution was to add 300 non-planet examples: stellar flares, noise, eclipsing binaries—all the things that look like transits but aren't.

After retraining on this balanced mix, the model achieved an AUC of 0.69 and actually learned to distinguish real planets from false positives.

This was a crucial lesson in the importance of representative training data."

---

## SLIDE 7: Optuna Optimization

**WHAT TO SAY (~45 seconds):**

"After getting a working baseline, I used Optuna—an automated hyperparameter optimization framework—to improve performance.

It ran 30 trials testing different combinations of layers, batch size, learning rate, and dropout.

The key improvements: 4 layers instead of 3, batch size 128 instead of 64, and learning rate increased from 1e-4 to 2.25e-4.

This progression chart shows the results. The initial approach with 100 planets only failed completely. The balanced baseline got to AUC 0.69. Optuna optimization pushed it to 0.76—a 9% improvement.

On real TESS data, the baseline was too conservative and made zero predictions. The optimized model made 16 predictions out of 300 windows, with much better calibration."

---

## SLIDE 8: Demo

**WHAT TO SAY (~20 seconds):**

"Here's a quick demo showing the model running on real TESS light curves.

You can see it correctly identifies TIC 307210830—which is the L 98-59 system, a confirmed multi-planet system with four known planets.

The model ranks it as the highest confidence detection, showing that it's learned to recognize real planetary transits."

**[Play 20-second video]**

---

## SLIDE 9: What's Next?

**WHAT TO SAY (~30 seconds):**

"For future work, I'm doing cross-mission generalization testing.

The idea is to train on TESS and test on Kepler—two different space telescopes with different cadences and wavelengths, but the same underlying physics.

If it works, that proves my model learned fundamental transit physics and isn't just overfitted to TESS-specific patterns. That would make it ready for future missions like PLATO and ARIEL.

If it fails, we'll learn about domain adaptation challenges in astronomical machine learning—which is also publishable.

Either way, it's a rigorous test of true generalization."

---

## TIMING BREAKDOWN (7 minutes total)

| Slide | Time | Running Total |
|-------|------|---------------|
| 1. Papers | 45s | 0:45 |
| 2. Why BiLSTM | 30s | 1:15 |
| 3. Methodology | 45s | 2:00 |
| 4. Architecture | 30s | 2:30 |
| 5. Results | 45s | 3:15 |
| 6. Failure | 45s | 4:00 |
| 7. Optuna | 45s | 4:45 |
| 8. Demo | 20s | 5:05 |
| 9. What's Next | 30s | 5:35 |
| **Buffer/Questions** | 1:25 | **7:00** |

---

## KEY TALKING POINTS (DON'T FORGET!)

✅ **H5 indices** - Mention on Slide 1 that all are >100
✅ **Class imbalance lesson** - Slide 6 is your story about learning
✅ **Real validation** - Slide 5, tested on 100 confirmed exoplanet systems
✅ **Cross-mission testing** - Slide 9, this is novel and important
✅ **Demo highlight** - L 98-59 is a real multi-planet system

---

## TIPS FOR DELIVERY

1. **Slides 1-2**: Set up why you chose this approach (papers → architecture)
2. **Slides 3-4**: Technical details but keep moving (audience can see diagrams)
3. **Slide 5**: This is your big result—emphasize AUC 75.72% and real validation
4. **Slide 6**: This is your story—failure is interesting and shows learning
5. **Slide 7**: Show the progression visually—people love before/after
6. **Slide 8**: Let the video do the talking, just frame it
7. **Slide 9**: End with forward-looking work—shows you're thinking ahead

---

## IF RUNNING SHORT ON TIME

**Can shorten:**
- Slide 2 (Why BiLSTM) - Just say "Based on these papers, I chose BiLSTM with clustering"
- Slide 4 (Architecture) - Just point to diagram: "Here's the full architecture"
- Slide 7 (Optuna) - Focus on the chart: "Optuna improved AUC from 0.69 to 0.76"

**Never cut:**
- Slide 5 (Results) - This is your main achievement
- Slide 6 (Failure) - This is your best story
- Slide 8 (Demo) - Visual proof it works

---

**TOTAL**: 5:35 of scripted content + 1:25 buffer = 7 minutes

This gives you room for natural pauses, questions, or going slightly over on interesting slides.

Good luck! 🚀
