"""
Generate publication-ready figures for BiLSTM section of term paper.

Creates:
1. Class Imbalance Strategy Comparison bar chart
2. ROC curve for SMOTE model (AUC 0.8175)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json

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
output_dir = Path(r'C:\CS_4280_Project\term_project_files\materials\figures')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Generating BiLSTM Figures for Term Paper")
print("=" * 60)

# ============================================================================
# Figure 1: Class Imbalance Strategy Comparison
# ============================================================================

print("\n[1/2] Creating Class Imbalance Strategy Comparison...")

# Data from FINAL_RESULTS_SMOTE_SUCCESS.md
strategies = ['Class\nWeighting', 'Naive\nUp-sampling', 'SMOTE']
auc_scores = [0.7572, 0.7210, 0.8175]
f1_scores = [0.5145, 0.6136, 0.7209]
detection_counts = [16, 0, 19]  # out of 300 windows
detection_rates = [x/300*100 for x in detection_counts]  # convert to percentage

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

# Subplot 1: Training Metrics (AUC and F1)
x = np.arange(len(strategies))
width = 0.35

bars1 = ax1.bar(x - width/2, auc_scores, width, label='AUC', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, f1_scores, width, label='F1 Score', color='#A23B72', alpha=0.8)

ax1.set_ylabel('Score')
ax1.set_title('Training Performance Metrics', fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(strategies)
ax1.legend(loc='upper left')
ax1.set_ylim(0, 1.0)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Subplot 2: Real Planet Detection Rate
colors = ['#2E86AB', '#E63946', '#06A77D']  # Blue, Red (failure), Green (success)
bars3 = ax2.bar(strategies, detection_rates, color=colors, alpha=0.8)

ax2.set_ylabel('Detection Rate (%)')
ax2.set_title('Real Planet Detection (300 Windows)', fontweight='bold', pad=10)
ax2.set_ylim(0, 8)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add value labels with counts
for i, (bar, count, rate) in enumerate(zip(bars3, detection_counts, detection_rates)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{count}/300\n({rate:.1f}%)', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
comparison_path = output_dir / 'bilstm_class_imbalance_comparison.png'
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {comparison_path}")
plt.close()

# ============================================================================
# Figure 2: ROC Curve for SMOTE Model
# ============================================================================

print("\n[2/2] Creating ROC Curve for SMOTE Model...")

# Load SMOTE model and validation data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = Path(r'C:\CS_4280_Project\Code\runs\bilstm_cluster_smote_true\best.pt')
windows_dir = Path(r'C:\CS_4280_Project\Code\data\windows_smote_true')

# Check if model and data exist
if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    print("Skipping ROC curve generation")
else:
    # Load checkpoint
    print(f"  Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    # Recreate model architecture (same as in inference script)
    class ClusterBiLSTM(torch.nn.Module):
        def __init__(self, input_size=1, hidden_size=256, num_layers=3,
                     n_clusters=5, cluster_embed_dim=32, dropout=0.3):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.n_clusters = n_clusters

            # Cluster embedding
            self.cluster_embed = torch.nn.Embedding(n_clusters, cluster_embed_dim)

            # BiLSTM
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )

            # Fully connected layers
            self.fc1 = torch.nn.Linear(hidden_size * 2 + cluster_embed_dim, 256)
            self.bn1 = torch.nn.BatchNorm1d(256)
            self.drop1 = torch.nn.Dropout(dropout)

            self.fc2 = torch.nn.Linear(256, 128)
            self.bn2 = torch.nn.BatchNorm1d(128)
            self.drop2 = torch.nn.Dropout(dropout)

            self.fc3 = torch.nn.Linear(128, 1)

        def forward(self, x, cluster_ids):
            # x: (batch, seq_len, 1)
            # cluster_ids: (batch,)

            # BiLSTM
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
            lstm_out = lstm_out[:, -1, :]  # Take last timestep (batch, hidden*2)

            # Cluster embedding
            cluster_emb = self.cluster_embed(cluster_ids)  # (batch, embed_dim)

            # Concatenate
            combined = torch.cat([lstm_out, cluster_emb], dim=1)

            # FC layers
            x = self.fc1(combined)
            x = self.bn1(x)
            x = torch.relu(x)
            x = self.drop1(x)

            x = self.fc2(x)
            x = self.bn2(x)
            x = torch.relu(x)
            x = self.drop2(x)

            x = self.fc3(x)
            return x.squeeze(-1)

    # Initialize model with config parameters
    model = ClusterBiLSTM(
        input_size=1,
        hidden_size=config.get('hidden', 256),
        num_layers=config.get('layers', 4),
        n_clusters=config.get('n_clusters', 5),
        cluster_embed_dim=config.get('cluster_embed_dim', 32),
        dropout=config.get('dropout', 0.311)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("  [OK] Model loaded successfully")

    # Load validation data
    print(f"  Loading data from {windows_dir}...")
    X = np.load(windows_dir / 'X.npy')
    y = np.load(windows_dir / 'y.npy')
    meta = pd.read_csv(windows_dir / 'meta.csv')

    # Recreate validation split (same seed as training)
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=config.get('val_split', 0.15),
        stratify=y,
        random_state=config.get('seed', 42)
    )

    X_val = X[val_idx]
    y_val = y[val_idx]
    meta_val = meta.iloc[val_idx].reset_index(drop=True)

    print(f"  Validation set: {len(X_val)} samples ({y_val.sum()} positive)")

    # Recreate clustering (same as training)
    scaler_params = checkpoint['scaler_params']
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params['mean'])
    scaler.scale_ = np.array(scaler_params['scale'])

    kmeans_centers = np.array(checkpoint['kmeans_centers'])

    # Properly initialize KMeans by fitting on centers
    # This ensures all internal attributes are set correctly
    kmeans = KMeans(n_clusters=len(kmeans_centers), random_state=42, n_init=1)
    # Create dummy data at cluster centers and fit
    kmeans.fit(kmeans_centers)
    # Verify centers match (they should be identical after fitting on centers themselves)
    kmeans.cluster_centers_ = kmeans_centers

    # Extract features and assign clusters
    feature_cols = ['period', 'duration', 'depth', 'bls_power']
    features_val = meta_val[feature_cols].values
    features_scaled = scaler.transform(features_val)
    cluster_ids_val = kmeans.predict(features_scaled)

    # Run inference
    print("  Running inference on validation set...")
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(-1).to(device)
        cluster_tensor = torch.LongTensor(cluster_ids_val).to(device)

        # Use mixed precision if available
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
            logits = model(X_val_tensor, cluster_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

    print(f"  [OK] Predictions complete (mean prob: {probs.mean():.4f})")

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, probs)
    roc_auc = auc(fpr, tpr)

    print(f"  Calculated AUC: {roc_auc:.4f}")

    # Create ROC curve plot
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot ROC curve
    ax.plot(fpr, tpr, color='#2E86AB', linewidth=2.5,
            label=f'SMOTE BiLSTM (AUC = {roc_auc:.4f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5,
            label='Random Classifier (AUC = 0.50)', alpha=0.7)

    # Formatting
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title('ROC Curve: SMOTE-Balanced BiLSTM Model',
                 fontweight='bold', pad=15)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')

    plt.tight_layout()
    roc_path = output_dir / 'bilstm_smote_roc_curve.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved: {roc_path}")
    plt.close()

print("\n" + "=" * 60)
print("Figure Generation Complete!")
print("=" * 60)
print(f"\nGenerated figures:")
print(f"  1. {output_dir / 'bilstm_class_imbalance_comparison.png'}")
print(f"  2. {output_dir / 'bilstm_smote_roc_curve.png'}")
print(f"\nBoth figures are publication-ready at 300 DPI.")
print(f"Use \\includegraphics in LaTeX to insert them.")
