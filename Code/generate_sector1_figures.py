"""
Generate publication-ready figures for Sector 1 BiLSTM results.

SOFA Refactored: December 3, 2025
Pylint Score: 10.00/10

Creates:
1. ROC curve with AUC annotation
2. Confusion matrix with metrics
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve


# =============================================================================
# Configuration
# =============================================================================

def setup_plot_style():
    """Configure matplotlib for publication-quality figures."""
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


# =============================================================================
# Data Loading
# =============================================================================

def load_predictions(predictions_path):
    """
    Load prediction results.

    Args:
        predictions_path: path to CSV file

    Returns:
        y_true, y_scores, y_pred arrays
    """
    print(f"[loading] {predictions_path}")
    pred_df = pd.read_csv(predictions_path)
    print(f"  Loaded {len(pred_df)} predictions")

    y_true = pred_df['label'].values
    y_scores = pred_df['probability'].values
    y_pred = pred_df['predicted'].values

    print(f"  Positives (label=1): {sum(y_true)}")
    print(f"  Negatives (label=0): {len(y_true) - sum(y_true)}")

    return y_true, y_scores, y_pred


# =============================================================================
# Figure Generation
# =============================================================================

def create_roc_curve(y_true, y_scores, output_path):
    """
    Create and save ROC curve figure.

    Args:
        y_true: true labels
        y_scores: prediction scores
        output_path: path to save figure
    """
    print("\n[1/2] Creating ROC Curve...")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    _, axes = plt.subplots(figsize=(5, 4.5))

    axes.plot(fpr, tpr, color='#2E86AB', lw=2,
              label=f'BiLSTM + Clustering (AUC = {roc_auc:.4f})')
    axes.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
              label='Random Classifier (AUC = 0.50)')

    axes.set_xlim([0.0, 1.0])
    axes.set_ylim([0.0, 1.05])
    axes.set_xlabel('False Positive Rate')
    axes.set_ylabel('True Positive Rate')
    axes.set_title('ROC Curve - TESS Sector 1 Test Set', fontweight='bold')
    axes.legend(loc='lower right')
    axes.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    axes.annotate(
        f'AUC = {roc_auc:.4f}',
        xy=(0.6, 0.3),
        fontsize=12,
        fontweight='bold',
        bbox={'boxstyle': 'round', 'facecolor': 'white',
              'edgecolor': 'gray', 'alpha': 0.8}
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return roc_auc


def create_confusion_matrix(y_true, y_pred, output_path):
    """
    Create and save confusion matrix figure.

    Args:
        y_true: true labels
        y_pred: predicted labels
        output_path: path to save figure

    Returns:
        dict of metrics
    """
    print("\n[2/2] Creating Confusion Matrix...")

    conf_mat = confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix shape: {conf_mat.shape}")
    print(f"  TN={conf_mat[0,0]}, FP={conf_mat[0,1]}, "
          f"FN={conf_mat[1,0]}, TP={conf_mat[1,1]}")

    _, axes = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        conf_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Non-Planet', 'Planet'],
        yticklabels=['Non-Planet', 'Planet'],
        annot_kws={'size': 14, 'weight': 'bold'},
        ax=axes
    )

    axes.set_xlabel('Predicted Label', fontweight='bold')
    axes.set_ylabel('True Label', fontweight='bold')
    axes.set_title('Confusion Matrix - TESS Sector 1 Test Set', fontweight='bold')

    metrics = compute_metrics(conf_mat)
    metrics_text = (f"Accuracy: {metrics['accuracy']:.2%}\n"
                    f"Precision: {metrics['precision']:.2%}\n"
                    f"Recall: {metrics['recall']:.2%}\n"
                    f"F1: {metrics['f1']:.4f}")

    axes.text(
        2.3, 0.5, metrics_text,
        transform=axes.transData,
        fontsize=9,
        verticalalignment='top',
        bbox={'boxstyle': 'round', 'facecolor': 'white',
              'edgecolor': 'gray', 'alpha': 0.8}
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()

    return metrics


def compute_metrics(conf_mat):
    """
    Compute classification metrics from confusion matrix.

    Args:
        conf_mat: confusion matrix array

    Returns:
        dict with accuracy, precision, recall, f1
    """
    tn, fp, fn, tp = conf_mat[0, 0], conf_mat[0, 1], conf_mat[1, 0], conf_mat[1, 1]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tn + tp) / conf_mat.sum()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_summary(output_dir, roc_auc, metrics):
    """Print generation summary."""
    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("Files created:")
    print("  - bilstm_sector1_roc_curve.png")
    print("  - confusion_matrix.png")
    print("\nMetrics:")
    print(f"  AUC:       {roc_auc:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    setup_plot_style()

    output_dir = Path(r'D:\CS_4280_Project\term_project_files\Images\RNN')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generating Sector 1 BiLSTM Figures")
    print("=" * 60)

    predictions_path = r'D:\CS_4280_Project\Code\reports\sector1_test_predictions.csv'
    y_true, y_scores, y_pred = load_predictions(predictions_path)

    roc_auc = create_roc_curve(
        y_true, y_scores,
        output_dir / 'bilstm_sector1_roc_curve.png'
    )

    metrics = create_confusion_matrix(
        y_true, y_pred,
        output_dir / 'confusion_matrix.png'
    )

    print_summary(output_dir, roc_auc, metrics)


if __name__ == '__main__':
    main()
