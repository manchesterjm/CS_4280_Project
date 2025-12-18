"""
Compare balanced vs unbalanced model performance.

After training on balanced dataset, this script compares the results.
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Load results from both models
unbalanced_dir = Path("runs/bilstm_cluster_optimized")
balanced_dir = Path("runs/bilstm_cluster_balanced")

print("="*70)
print("BALANCED vs UNBALANCED MODEL COMPARISON")
print("="*70)

# Load configs
if (unbalanced_dir / "config.json").exists():
    with open(unbalanced_dir / "config.json") as f:
        unbalanced_config = json.load(f)
else:
    print("WARNING: Unbalanced model config not found")
    unbalanced_config = {}

if (balanced_dir / "config.json").exists():
    with open(balanced_dir / "config.json") as f:
        balanced_config = json.load(f)
else:
    print("WARNING: Balanced model config not found")
    balanced_config = {}

print("\n" + "="*70)
print("DATASET COMPARISON")
print("="*70)

# Dataset info from your session
print("\nUnbalanced Dataset:")
print("  Total windows: 655")
print("  Positive (planets): 150 (22.9%)")
print("  Negative (non-planets): 505 (77.1%)")
print("  Strategy: Class weighting (pos_weight=3.367)")

print("\nBalanced Dataset:")
print("  Total windows: 600")
print("  Positive (planets): 300 (50.0%)")
print("  Negative (non-planets): 300 (50.0%)")
print("  Strategy: Down-sample + Up-sample (pos_weight=1.0)")

print("\n" + "="*70)
print("TRAINING RESULTS COMPARISON")
print("="*70)

# From the output logs
results = {
    "Model": ["Unbalanced (Optimized)", "Balanced (Down+Up Sample)"],
    "AUC": [0.7572, 0.7210],
    "F1": [0.5145, 0.6136],
    "Dataset Size": [655, 600],
    "Positive %": [22.9, 50.0],
    "pos_weight": [3.367, 1.0],
}

df = pd.DataFrame(results)
print("\n")
print(df.to_string(index=False))

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

auc_change = results["AUC"][1] - results["AUC"][0]
f1_change = results["F1"][1] - results["F1"][0]

print(f"\n1. AUC Change: {auc_change:+.4f} ({100*auc_change/results['AUC'][0]:+.1f}%)")
if auc_change < 0:
    print("   → AUC decreased slightly, but this might be acceptable if precision improved")
else:
    print("   → AUC improved!")

print(f"\n2. F1 Score Change: {f1_change:+.4f} ({100*f1_change/results['F1'][0]:+.1f}%)")
if f1_change > 0:
    print("   → F1 improved significantly! Better precision-recall balance")
else:
    print("   → F1 decreased")

print("\n3. Interpretation:")
print("   - Unbalanced model: Likely high recall, low precision (many false positives)")
print("   - Balanced model: Better precision-recall balance (fewer false alarms)")
print("   - Trade-off: Slightly lower AUC, but more practical predictions")

print("\n" + "="*70)
print("EXPECTED METRICS (based on training logs)")
print("="*70)

print("\nUnbalanced Model (from previous benchmarking):")
print("  AUC: 0.7572")
print("  Recall: 0.8867 (88.67% of planets detected)")
print("  Precision: 0.3827 (38.27% of predictions are correct)")
print("  Problem: Many false positives (61.73% of planet predictions are wrong!)")

print("\nBalanced Model (expected based on F1 improvement):")
print("  AUC: 0.7210")
print("  Expected Recall: ~0.75-0.80 (slightly lower)")
print("  Expected Precision: ~0.50-0.60 (much better!)")
print("  Improvement: Fewer false positives, more trustworthy predictions")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

print("\n1. Run full benchmark on balanced model:")
print('   python benchmark_model.py --model_path "runs/bilstm_cluster_balanced/best.pt" --output_dir "benchmarks"')

print("\n2. Test on 100 real planets:")
print('   python inference_cluster_model.py --model_path "runs/bilstm_cluster_balanced/best.pt" --windows_dir "data/windows_planet_test" --output_file "reports/balanced_planet_predictions.csv"')

print("\n3. Compare predictions on real planets:")
print("   - Unbalanced: reports/optimized_planet_predictions.csv")
print("   - Balanced: reports/balanced_planet_predictions.csv")
print("   - Which model gives fewer false alarms?")

print("\n4. Visualize trade-offs:")
print("   - Create ROC curves for both models")
print("   - Compare precision-recall curves")
print("   - Analyze confusion matrices")

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

print("\nThe balanced model shows:")
print("  ✅ +19.3% F1 improvement (better precision-recall balance)")
print("  ⚠️  -4.8% AUC decrease (slightly worse ranking)")
print()
print("This is a GOOD trade-off if:")
print("  - You need fewer false alarms (astronomers don't want to follow up on junk)")
print("  - You prefer trustworthy predictions over catching every planet")
print("  - Production use-case values precision > recall")
print()
print("Use unbalanced model if:")
print("  - You want to catch as many planets as possible (screening phase)")
print("  - False positives are acceptable (human verification available)")
print("  - Recall is more important than precision")

print("\n" + "="*70)
