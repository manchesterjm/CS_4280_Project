"""
Compare baseline vs optimized model predictions on real planet data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Read prediction files
baseline_df = pd.read_csv(r'C:\CS_4280_Project\Code\reports\baseline_planet_predictions.csv')
optimized_df = pd.read_csv(r'C:\CS_4280_Project\Code\reports\optimized_planet_predictions.csv')

print("="*80)
print(" BASELINE vs OPTIMIZED MODEL COMPARISON")
print("="*80)

print(f"\nDataset: {len(baseline_df)} windows from {baseline_df['tic_id'].nunique()} confirmed exoplanet systems")

# Baseline metrics
baseline_probs = baseline_df['probability'].values
baseline_preds = (baseline_probs > 0.5).astype(int)

print("\n" + "="*80)
print(" BASELINE MODEL (AUC 0.6947)")
print("="*80)
print(f"Mean probability: {baseline_probs.mean():.4f}")
print(f"Median probability: {np.median(baseline_probs):.4f}")
print(f"Max probability: {baseline_probs.max():.4f}")
print(f"Min probability: {baseline_probs.min():.4f}")
print(f"Std deviation: {baseline_probs.std():.4f}")
print(f"\nPositive predictions (>0.5): {baseline_preds.sum()}/{len(baseline_preds)} ({100*baseline_preds.sum()/len(baseline_preds):.1f}%)")

# Optimized metrics
optimized_probs = optimized_df['probability'].values
optimized_preds = (optimized_probs > 0.5).astype(int)

print("\n" + "="*80)
print(" OPTIMIZED MODEL (AUC 0.7572)")
print("="*80)
print(f"Mean probability: {optimized_probs.mean():.4f}")
print(f"Median probability: {np.median(optimized_probs):.4f}")
print(f"Max probability: {optimized_probs.max():.4f}")
print(f"Min probability: {optimized_probs.min():.4f}")
print(f"Std deviation: {optimized_probs.std():.4f}")
print(f"\nPositive predictions (>0.5): {optimized_preds.sum()}/{len(optimized_preds)} ({100*optimized_preds.sum()/len(optimized_preds):.1f}%)")

# Comparison
print("\n" + "="*80)
print(" IMPROVEMENT METRICS")
print("="*80)
print(f"AUC improvement: 0.6947 → 0.7572 (+{100*(0.7572-0.6947)/0.6947:.1f}%)")
print(f"Mean probability change: {baseline_probs.mean():.4f} → {optimized_probs.mean():.4f} ({100*(optimized_probs.mean()-baseline_probs.mean())/baseline_probs.mean():+.1f}%)")
print(f"Positive predictions change: {baseline_preds.sum()} → {optimized_preds.sum()} ({optimized_preds.sum()-baseline_preds.sum():+d})")

# Per-star analysis
print("\n" + "="*80)
print(" TOP 10 PLANET CANDIDATES (Optimized Model)")
print("="*80)

# Get max probability per TIC
tic_max_probs = optimized_df.groupby('tic_id')['probability'].max().sort_values(ascending=False)
print("\nTIC ID          Max Prob   All 3 Windows")
print("-" * 50)
for tic_id in tic_max_probs.head(10).index:
    max_prob = tic_max_probs[tic_id]
    tic_probs = optimized_df[optimized_df['tic_id'] == tic_id]['probability'].values
    print(f"TIC {tic_id:<12} {max_prob:.4f}     {tic_probs[0]:.4f}, {tic_probs[1]:.4f}, {tic_probs[2]:.4f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Probability distributions
axes[0, 0].hist(baseline_probs, bins=30, alpha=0.6, label='Baseline', color='blue', edgecolor='black')
axes[0, 0].hist(optimized_probs, bins=30, alpha=0.6, label='Optimized', color='green', edgecolor='black')
axes[0, 0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
axes[0, 0].set_xlabel('Probability', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 2. Scatter plot comparison
axes[0, 1].scatter(baseline_probs, optimized_probs, alpha=0.5, s=20)
axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
axes[0, 1].axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='Optimized threshold')
axes[0, 1].axvline(0.5, color='blue', linestyle='--', alpha=0.7, label='Baseline threshold')
axes[0, 1].set_xlabel('Baseline Probability', fontsize=12)
axes[0, 1].set_ylabel('Optimized Probability', fontsize=12)
axes[0, 1].set_title('Baseline vs Optimized Predictions', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 3. Per-TIC max probability comparison
baseline_tic_max = baseline_df.groupby('tic_id')['probability'].max().sort_index()
optimized_tic_max = optimized_df.groupby('tic_id')['probability'].max().sort_index()

x_pos = np.arange(len(baseline_tic_max))
width = 0.35

axes[1, 0].bar(x_pos - width/2, baseline_tic_max.values, width, label='Baseline', alpha=0.8, color='blue')
axes[1, 0].bar(x_pos + width/2, optimized_tic_max.values, width, label='Optimized', alpha=0.8, color='green')
axes[1, 0].axhline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
axes[1, 0].set_xlabel('TIC ID (index)', fontsize=12)
axes[1, 0].set_ylabel('Max Probability', fontsize=12)
axes[1, 0].set_title('Max Probability per Confirmed Exoplanet System', fontsize=14, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].set_ylim([0, 1])

# 4. Metrics comparison bar chart
metrics_names = ['Mean\nProbability', 'Median\nProbability', 'Max\nProbability', 'Positive\nPredictions\n(%)']
baseline_metrics = [
    baseline_probs.mean(),
    np.median(baseline_probs),
    baseline_probs.max(),
    100 * baseline_preds.sum() / len(baseline_preds)
]
optimized_metrics = [
    optimized_probs.mean(),
    np.median(optimized_probs),
    optimized_probs.max(),
    100 * optimized_preds.sum() / len(optimized_preds)
]

x_pos = np.arange(len(metrics_names))
axes[1, 1].bar(x_pos - width/2, baseline_metrics, width, label='Baseline', alpha=0.8, color='blue')
axes[1, 1].bar(x_pos + width/2, optimized_metrics, width, label='Optimized', alpha=0.8, color='green')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(metrics_names, fontsize=10)
axes[1, 1].set_ylabel('Value', fontsize=12)
axes[1, 1].set_title('Metrics Comparison', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(r'C:\CS_4280_Project\Code\reports\model_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved visualization: C:\\CS_4280_Project\\Code\\reports\\model_comparison.png")

# Create improvement summary figure
fig2, ax = plt.subplots(figsize=(10, 6))

improvements = [
    ('AUC', 0.6947, 0.7572),
    ('Mean Prob', baseline_probs.mean(), optimized_probs.mean()),
    ('Median Prob', np.median(baseline_probs), np.median(optimized_probs)),
    ('Max Prob', baseline_probs.max(), optimized_probs.max())
]

labels = [item[0] for item in improvements]
baseline_vals = [item[1] for item in improvements]
optimized_vals = [item[2] for item in improvements]
improvements_pct = [100*(optimized_vals[i]-baseline_vals[i])/baseline_vals[i] for i in range(len(labels))]

x_pos = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x_pos - width/2, baseline_vals, width, label='Baseline (3 layers)', alpha=0.8, color='#3498db')
bars2 = ax.bar(x_pos + width/2, optimized_vals, width, label='Optimized (4 layers)', alpha=0.8, color='#2ecc71')

# Add improvement percentages on top
for i, (bar1, bar2, pct) in enumerate(zip(bars1, bars2, improvements_pct)):
    height = max(bar1.get_height(), bar2.get_height())
    ax.text(i, height + 0.02, f'+{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold', color='green')

ax.set_xlabel('Metric', fontsize=13, fontweight='bold')
ax.set_ylabel('Value', fontsize=13, fontweight='bold')
ax.set_title('BiLSTM+Clustering Model Improvement After Optuna Optimization\n(Tested on 300 windows from 100 confirmed exoplanet systems)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, max(optimized_vals) * 1.15])

plt.tight_layout()
plt.savefig(r'C:\CS_4280_Project\Code\reports\improvement_summary.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization: C:\\CS_4280_Project\\Code\\reports\\improvement_summary.png")

print("\n" + "="*80)
print(" SUMMARY")
print("="*80)
print(f"\nOptuna hyperparameter optimization resulted in:")
print(f"  • AUC improved from 0.6947 to 0.7572 (+9.0%)")
print(f"  • Mean probability increased from {baseline_probs.mean():.4f} to {optimized_probs.mean():.4f}")
print(f"  • Positive predictions increased from {baseline_preds.sum()} to {optimized_preds.sum()}")
print(f"  • Key changes: 4 layers (vs 3), batch size 128 (vs 64), LR 0.000225 (vs 0.0001)")
print(f"\nBoth models tested on real confirmed exoplanet systems from TESS/Kepler missions.")
print("="*80)
