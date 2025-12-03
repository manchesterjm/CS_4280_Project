"""
Generate publication-ready figures for Sector 1 BiLSTM results.

Creates:
1. ROC curve (AUC 0.9174)
2. Confusion matrix (TP=732, FP=1012, TN=4835, FN=0)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})

# Create output directory
output_dir = Path(r'C:\CS_4280_Project\term_project_files\Images\RNN')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Generating Sector 1 BiLSTM Figures")
print("=" * 60)

# Load predictions
print("\n[loading] sector1_test_predictions.csv")
df = pd.read_csv(r'C:\CS_4280_Project\Code\reports\sector1_test_predictions.csv')
print(f"  Loaded {len(df)} predictions")

# Extract y_true and y_scores
y_true = df['label'].values
y_scores = df['probability'].values
y_pred = df['predicted'].values

print(f"  Positives (label=1): {sum(y_true)}")
print(f"  Negatives (label=0): {len(y_true) - sum(y_true)}")

# ============================================================================
# Figure 1: ROC Curve
# ============================================================================

print("\n[1/2] Creating ROC Curve...")

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(5, 4.5))

# Plot ROC curve
ax.plot(fpr, tpr, color='#2E86AB', lw=2, label=f'BiLSTM + Clustering (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier (AUC = 0.50)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - TESS Sector 1 Test Set', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Add AUC annotation
ax.annotate(f'AUC = {roc_auc:.4f}',
            xy=(0.6, 0.3), fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

plt.tight_layout()
roc_path = output_dir / 'bilstm_sector1_roc_curve.png'
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {roc_path}")
plt.close()

# ============================================================================
# Figure 2: Confusion Matrix
# ============================================================================

print("\n[2/2] Creating Confusion Matrix...")

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f"  Confusion matrix shape: {cm.shape}")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

fig, ax = plt.subplots(figsize=(5, 4))

# Create heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Planet', 'Planet'],
            yticklabels=['Non-Planet', 'Planet'],
            annot_kws={'size': 14, 'weight': 'bold'},
            ax=ax)

ax.set_xlabel('Predicted Label', fontweight='bold')
ax.set_ylabel('True Label', fontweight='bold')
ax.set_title('Confusion Matrix - TESS Sector 1 Test Set', fontweight='bold')

# Add metrics annotation
precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (cm[0,0] + cm[1,1]) / cm.sum()

metrics_text = f'Accuracy: {accuracy:.2%}\nPrecision: {precision:.2%}\nRecall: {recall:.2%}\nF1: {f1:.4f}'
ax.text(2.3, 0.5, metrics_text, transform=ax.transData, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

plt.tight_layout()
cm_path = output_dir / 'confusion_matrix.png'
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"  Saved: {cm_path}")
plt.close()

print("\n" + "=" * 60)
print("FIGURE GENERATION COMPLETE")
print("=" * 60)
print(f"\nOutput directory: {output_dir}")
print(f"Files created:")
print(f"  - bilstm_sector1_roc_curve.png")
print(f"  - confusion_matrix.png")
print(f"\nMetrics:")
print(f"  AUC:       {roc_auc:.4f}")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
