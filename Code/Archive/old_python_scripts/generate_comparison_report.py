r"""
Generate Comprehensive Comparison Report

Compares baseline vs optimized model performance and generates visualizations.

Usage:
    python generate_comparison_report.py --baseline_results "C:\CS_4280_Project\Code\benchmarks\baseline_benchmark_*.json" --optimized_results "C:\CS_4280_Project\Code\optuna_results\*_benchmark_*.json" --output_dir "C:\CS_4280_Project\Code\comparison_report"
"""

import argparse
import json
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results(pattern):
    """Load benchmark results from file pattern"""
    files = glob.glob(pattern)
    if not files:
        return None

    # Load most recent file
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    with open(files[0], 'r') as f:
        return json.load(f)


def create_metrics_comparison(baseline, optimized, output_dir):
    """Create side-by-side metrics comparison table"""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    data = {
        'Metric': [m.upper() for m in metrics],
        'Baseline': [baseline['metrics'][m] for m in metrics],
        'Optimized': [optimized['metrics'][m] for m in metrics],
        'Improvement': [optimized['metrics'][m] - baseline['metrics'][m] for m in metrics],
        'Improvement %': [(optimized['metrics'][m] - baseline['metrics'][m]) / baseline['metrics'][m] * 100 for m in metrics]
    }

    df = pd.DataFrame(data)
    df.to_csv(output_dir / 'metrics_comparison.csv', index=False)

    return df


def plot_metrics_comparison(df, output_dir):
    """Plot metrics comparison bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width/2, df['Baseline'], width, label='Baseline', alpha=0.8)
    bars2 = ax.bar(x + width/2, df['Optimized'], width, label='Optimized', alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Baseline vs Optimized Model Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Metric'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(baseline, optimized, output_dir):
    """Plot confusion matrices side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Baseline
    cm_baseline = np.array([[baseline['metrics']['tn'], baseline['metrics']['fp']],
                           [baseline['metrics']['fn'], baseline['metrics']['tp']]])
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=ax1,
               xticklabels=['Predicted Neg', 'Predicted Pos'],
               yticklabels=['Actual Neg', 'Actual Pos'])
    ax1.set_title(f"Baseline (AUC: {baseline['metrics']['auc']:.4f})", fontsize=12, fontweight='bold')

    # Optimized
    cm_optimized = np.array([[optimized['metrics']['tn'], optimized['metrics']['fp']],
                            [optimized['metrics']['fn'], optimized['metrics']['tp']]])
    sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Greens', ax=ax2,
               xticklabels=['Predicted Neg', 'Predicted Pos'],
               yticklabels=['Actual Neg', 'Actual Pos'])
    ax2.set_title(f"Optimized (AUC: {optimized['metrics']['auc']:.4f})", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_comparison(baseline, optimized, output_dir):
    """Plot ROC curves comparison"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Baseline ROC
    fpr_base = baseline['metrics']['roc_curve']['fpr']
    tpr_base = baseline['metrics']['roc_curve']['tpr']
    ax.plot(fpr_base, tpr_base, label=f"Baseline (AUC={baseline['metrics']['auc']:.4f})",
           linewidth=2, alpha=0.8)

    # Optimized ROC
    fpr_opt = optimized['metrics']['roc_curve']['fpr']
    tpr_opt = optimized['metrics']['roc_curve']['tpr']
    ax.plot(fpr_opt, tpr_opt, label=f"Optimized (AUC={optimized['metrics']['auc']:.4f})",
           linewidth=2, alpha=0.8)

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.3)

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_hyperparameter_comparison(baseline, optimized, output_dir):
    """Plot hyperparameter changes"""
    # Compare key hyperparameters
    params = ['hidden_size', 'num_layers', 'dropout', 'lr', 'batch_size', 'n_clusters']

    baseline_config = baseline['config']
    optimized_config = optimized['config']

    # Map config keys
    key_map = {
        'hidden_size': 'hidden',
        'num_layers': 'layers',
        'dropout': 'dropout',
        'lr': 'lr',
        'batch_size': 'batch_size',
        'n_clusters': 'n_clusters'
    }

    data = []
    for param in params:
        base_key = key_map.get(param, param)
        base_val = baseline_config.get(base_key, 'N/A')
        opt_val = optimized_config.get(base_key, 'N/A')

        data.append({
            'Parameter': param,
            'Baseline': base_val,
            'Optimized': opt_val,
            'Changed': '✓' if base_val != opt_val else ''
        })

    df = pd.DataFrame(data)
    df.to_csv(output_dir / 'hyperparameter_comparison.csv', index=False)

    return df


def generate_markdown_report(baseline, optimized, metrics_df, hp_df, output_dir):
    """Generate comprehensive markdown report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Model Optimization Report

**Generated:** {timestamp}

## Executive Summary

This report compares the baseline BiLSTM+Clustering model with the Optuna-optimized version.

### Key Improvements

| Metric | Baseline | Optimized | Improvement | Δ % |
|--------|----------|-----------|-------------|-----|
"""

    for _, row in metrics_df.iterrows():
        report += f"| {row['Metric']} | {row['Baseline']:.4f} | {row['Optimized']:.4f} | {row['Improvement']:+.4f} | {row['Improvement %']:+.2f}% |\n"

    report += f"""

### Highlights

- **AUC Improvement:** {metrics_df[metrics_df['Metric'] == 'AUC']['Improvement %'].values[0]:+.2f}%
- **F1 Improvement:** {metrics_df[metrics_df['Metric'] == 'F1']['Improvement %'].values[0]:+.2f}%
- **Precision Improvement:** {metrics_df[metrics_df['Metric'] == 'PRECISION']['Improvement %'].values[0]:+.2f}%

## Confusion Matrix Comparison

### Baseline
- True Positives: {baseline['metrics']['tp']}
- False Positives: {baseline['metrics']['fp']}
- True Negatives: {baseline['metrics']['tn']}
- False Negatives: {baseline['metrics']['fn']}

### Optimized
- True Positives: {optimized['metrics']['tp']}
- False Positives: {optimized['metrics']['fp']}
- True Negatives: {optimized['metrics']['tn']}
- False Negatives: {optimized['metrics']['fn']}

## Hyperparameter Changes

| Parameter | Baseline | Optimized | Changed |
|-----------|----------|-----------|---------|
"""

    for _, row in hp_df.iterrows():
        report += f"| {row['Parameter']} | {row['Baseline']} | {row['Optimized']} | {row['Changed']} |\n"

    report += f"""

## Model Architecture

### Baseline
- Parameters: {baseline['model_parameters']['total']:,}
- Hidden Size: {baseline['config']['hidden']}
- Layers: {baseline['config']['layers']}
- Dropout: {baseline['config']['dropout']}

### Optimized
- Parameters: {optimized['model_parameters']['total']:,}
- Hidden Size: {optimized['config']['hidden']}
- Layers: {optimized['config']['layers']}
- Dropout: {optimized['config']['dropout']}

## Visualizations

![Metrics Comparison](metrics_comparison.png)

![Confusion Matrices](confusion_matrices.png)

![ROC Curves](roc_comparison.png)

## Recommendations

"""

    # Add recommendations based on results
    auc_improvement = metrics_df[metrics_df['Metric'] == 'AUC']['Improvement %'].values[0]

    if auc_improvement > 5:
        report += "✅ **Significant improvement achieved!** The optimized model shows substantial gains over the baseline.\n\n"
    elif auc_improvement > 2:
        report += "✓ **Moderate improvement.** The optimization has yielded measurable benefits.\n\n"
    else:
        report += "⚠️ **Limited improvement.** Consider exploring additional architectural changes or data augmentation.\n\n"

    report += f"""
### Next Steps
1. **Test on real planet data** (Planet_LightCurve_Data) to validate generalization
2. **Compare with validation set** performance from training
3. **Consider ensemble methods** combining multiple optimized models
4. **Investigate attention mechanisms** for improved interpretability

## Dataset Information

- **Total Windows:** {baseline['dataset_size']}
- **Positive Samples:** {baseline['positive_samples']} ({100*baseline['positive_samples']/baseline['dataset_size']:.1f}%)
- **Training Strategy:** Stratified split with class weighting

## Reproducibility

### Baseline Configuration
```json
{json.dumps(baseline['config'], indent=2)}
```

### Optimized Configuration
```json
{json.dumps(optimized['config'], indent=2)}
```

---

*Report generated by Claude Code for CS4280 Exoplanet Detection Project*
"""

    with open(output_dir / 'OPTIMIZATION_REPORT.md', 'w') as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_results', type=str,
                       default=r'C:\CS_4280_Project\Code\benchmarks\baseline_benchmark_*.json',
                       help='Baseline results file pattern')
    parser.add_argument('--optimized_results', type=str,
                       default=r'C:\CS_4280_Project\Code\benchmarks\optimized_benchmark_*.json',
                       help='Optimized results file pattern')
    parser.add_argument('--output_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\comparison_report',
                       help='Output directory for comparison report')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(" GENERATING COMPARISON REPORT")
    print("="*70)

    # Load results
    print("\nLoading results...")
    baseline = load_results(args.baseline_results)
    optimized = load_results(args.optimized_results)

    if baseline is None:
        print("❌ No baseline results found!")
        return

    if optimized is None:
        print("❌ No optimized results found!")
        return

    print("✓ Baseline results loaded")
    print("✓ Optimized results loaded")

    # Generate comparisons
    print("\nGenerating comparisons...")
    metrics_df = create_metrics_comparison(baseline, optimized, output_dir)
    print("✓ Metrics comparison created")

    hp_df = plot_hyperparameter_comparison(baseline, optimized, output_dir)
    print("✓ Hyperparameter comparison created")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_metrics_comparison(metrics_df, output_dir)
    print("✓ Metrics comparison plot saved")

    plot_confusion_matrices(baseline, optimized, output_dir)
    print("✓ Confusion matrices saved")

    plot_roc_comparison(baseline, optimized, output_dir)
    print("✓ ROC comparison saved")

    # Generate report
    print("\nGenerating markdown report...")
    generate_markdown_report(baseline, optimized, metrics_df, hp_df, output_dir)
    print("✓ Report saved")

    print(f"\n{'='*70}")
    print(f"REPORT COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - OPTIMIZATION_REPORT.md")
    print(f"  - metrics_comparison.csv")
    print(f"  - metrics_comparison.png")
    print(f"  - confusion_matrices.png")
    print(f"  - roc_comparison.png")
    print(f"  - hyperparameter_comparison.csv")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
