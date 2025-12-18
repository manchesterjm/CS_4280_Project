"""
Generate publication-ready figures for final term paper.

Creates:
1. ROC Curve
2. Confusion Matrix
3. Probability Distribution
4. Model Comparison

Updated: December 6, 2025 with final test results (AUC 0.9261)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from pathlib import Path

# Set publication-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
})


def load_data():
    """Load test predictions and ground truth."""
    pred = pd.read_csv('reports/sector1_final_test_predictions.csv')
    y_true = np.load('D:/CS_4280_Project_Backup/Code/data/windows_sector1_full/test/y.npy')
    y_true = y_true[:len(pred)]
    probs = pred['probability'].values
    return y_true, probs


def plot_roc_curve(y_true, probs, output_dir):
    """Generate ROC curve figure."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(fpr, tpr, color='#2563eb', lw=2.5,
            label=f'BiLSTM + Clustering (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--',
            label='Random Classifier (AUC = 0.5)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('ROC Curve - Exoplanet Detection\nTESS Sector 1 Test Set', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.annotate(f'AUC = {roc_auc:.4f}', xy=(0.6, 0.3), fontsize=20,
                fontweight='bold', color='#2563eb')

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[saved] {output_dir / 'roc_curve.png'}")


def plot_confusion_matrix(y_true, probs, output_dir):
    """Generate confusion matrix figure."""
    y_pred = (probs > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                annot_kws={'size': 20, 'fontweight': 'bold'},
                xticklabels=['Non-Planet', 'Planet'],
                yticklabels=['Non-Planet', 'Planet'])

    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title('Confusion Matrix - Exoplanet Detection\nTESS Sector 1 Test Set', fontsize=16)

    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics_text = (f'Accuracy: {accuracy:.2%}\n'
                   f'Precision: {precision:.2%}\n'
                   f'Recall: {recall:.2%}\n'
                   f'Total: {total:,}')
    ax.text(2.3, 0.5, metrics_text, transform=ax.transData, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[saved] {output_dir / 'confusion_matrix.png'}")


def plot_probability_distribution(y_true, probs, output_dir):
    """Generate probability distribution figure."""
    fig, ax = plt.subplots(figsize=(10, 6))

    probs_planets = probs[y_true == 1]
    probs_non_planets = probs[y_true == 0]

    ax.hist(probs_non_planets, bins=50, alpha=0.7,
            label=f'Non-Planets (n={len(probs_non_planets):,})',
            color='#ef4444', density=True)
    ax.hist(probs_planets, bins=50, alpha=0.7,
            label=f'Planets (n={len(probs_planets):,})',
            color='#22c55e', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', lw=2,
               label='Decision Threshold (0.5)')

    ax.set_xlabel('Predicted Probability', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Prediction Probability Distribution\nTESS Sector 1 Test Set', fontsize=16)
    ax.legend(loc='upper center', fontsize=11)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'probability_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[saved] {output_dir / 'probability_distribution.png'}")


def plot_model_comparison(output_dir):
    """Generate model comparison bar chart."""
    models = ['Baseline\n(Oct 2025)', 'Optuna\n(Nov 2025)', 'Sector 1\n(Dec 2025)']
    aucs = [0.6947, 0.7572, 0.9261]
    colors = ['#94a3b8', '#60a5fa', '#22c55e']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, aucs, color=colors, edgecolor='black', linewidth=1.5)

    for bar, auc_val in zip(bars, aucs):
        height = bar.get_height()
        ax.annotate(f'{auc_val:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('AUC Score', fontsize=14)
    ax.set_title('Model Performance Comparison\nExoplanet Detection AUC Improvement', fontsize=16)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    ax.annotate('+9.0%', xy=(1, 0.73), fontsize=12, color='#2563eb', fontweight='bold')
    ax.annotate('+33.3%', xy=(2, 0.88), fontsize=12, color='#16a34a', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[saved] {output_dir / 'model_comparison.png'}")


def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating Publication Figures - December 6, 2025")
    print("=" * 60)

    # Create output directories
    output_dir = Path('D:/CS_4280_Project/term_project_files/Images/RNN')
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_dir = Path('runs/sector1_final_0918/figures')
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n[loading data]")
    y_true, probs = load_data()
    print(f"  Samples: {len(y_true)}")
    print(f"  Positives: {y_true.sum()}")

    # Generate figures to both directories
    print("\n[generating figures]")
    for out_dir in [output_dir, runs_dir]:
        plot_roc_curve(y_true, probs, out_dir)
        plot_confusion_matrix(y_true, probs, out_dir)
        plot_probability_distribution(y_true, probs, out_dir)
        plot_model_comparison(out_dir)

    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Saved to: {output_dir}")
    print(f"Saved to: {runs_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
