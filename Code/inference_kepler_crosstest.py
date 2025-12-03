"""
Cross-dataset inference: Run TESS-trained model on Kepler data.

This script tests generalization by running a TESS Sector 1 trained model
on Kepler confirmed exoplanet data.

Purpose: Generate numbers for the paper showing TESS -> Kepler domain shift.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn


class ClusterBiLSTM(nn.Module):
    """BiLSTM with cluster-aware processing (must match training)."""

    def __init__(self, config):
        super().__init__()
        self.n_clusters = config['n_clusters']
        self.cluster_embed = nn.Embedding(
            config['n_clusters'],
            config['cluster_embed_dim']
        )
        self.lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'] if config['num_layers'] > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        # Original architecture used separate layers
        combined_size = config['hidden_size'] * 2 + config['cluster_embed_dim']
        self.fc1 = nn.Linear(combined_size, config['hidden_size'])
        self.bn1 = nn.BatchNorm1d(config['hidden_size'])
        self.fc2 = nn.Linear(config['hidden_size'], config['hidden_size'] // 2)
        self.bn2 = nn.BatchNorm1d(config['hidden_size'] // 2)
        self.fc3 = nn.Linear(config['hidden_size'] // 2, 1)
        self.dropout = nn.Dropout(config['dropout'])
        self.relu = nn.ReLU()

    def forward(self, flux, cluster_ids):
        cluster_emb = self.cluster_embed(cluster_ids)
        _, (hidden, _) = self.lstm(flux)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        combined = torch.cat([hidden_cat, cluster_emb], dim=1)

        x = self.dropout(combined)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits.squeeze(-1)


def main():
    """Run TESS model on Kepler data and report metrics."""
    print("=" * 70)
    print("TESS -> KEPLER CROSS-DATASET GENERALIZATION TEST")
    print("=" * 70)

    # Paths
    model_path = r"D:\CS_4280_Project_Backup\Code\runs\bilstm_cluster_sector1\best.pt"
    kepler_dir = r"D:\CS_4280_Project_Backup\Code\data\windows_kepler"

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load Kepler data
    print("\n[1] Loading Kepler data...")
    flux_data = np.load(f"{kepler_dir}/X.npy")
    labels = np.load(f"{kepler_dir}/y.npy")
    meta = pd.read_csv(f"{kepler_dir}/meta.csv")

    print(f"    Windows: {len(flux_data)}")
    print(f"    Unique stars: {meta['filename'].nunique()}")
    print(f"    All positive (confirmed planets): {labels.sum()}/{len(labels)}")

    # Load TESS model
    print("\n[2] Loading TESS-trained model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Get actual n_clusters from saved weights (may differ from config)
    actual_n_clusters = checkpoint['model_state_dict']['cluster_embed.weight'].shape[0]

    model_config = {
        'input_size': 1,
        'hidden_size': config['hidden'],
        'num_layers': config['layers'],
        'dropout': config['dropout'],
        'n_clusters': actual_n_clusters,  # Use actual from checkpoint
        'cluster_embed_dim': config.get('cluster_embed_dim', 32)
    }
    print(f"    n_clusters from checkpoint: {actual_n_clusters}")

    model = ClusterBiLSTM(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"    Loaded: {model_path}")
    print(f"    Config: hidden={config['hidden']}, layers={config['layers']}")

    # Run inference (using cluster 0 for all - no BLS features in Kepler data)
    print("\n[3] Running inference...")
    flux_tensor = torch.from_numpy(flux_data).float().unsqueeze(-1)
    cluster_ids = torch.zeros(len(flux_data), dtype=torch.long)  # All cluster 0

    all_probs = []
    batch_size = 64

    with torch.no_grad():
        for i in range(0, len(flux_tensor), batch_size):
            batch_flux = flux_tensor[i:i+batch_size].to(device)
            batch_clusters = cluster_ids[i:i+batch_size].to(device)
            logits = model(batch_flux, batch_clusters)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs)
    predictions = (probs > 0.5).astype(int)

    # Calculate metrics
    print("\n" + "=" * 70)
    print("RESULTS: TESS MODEL ON KEPLER DATA")
    print("=" * 70)

    # Since all Kepler labels are 1 (confirmed planets), AUC isn't meaningful
    # But we can report detection rate
    detection_rate = predictions.sum() / len(predictions)
    mean_prob = probs.mean()

    print(f"\nKepler Dataset (all confirmed exoplanets):")
    print(f"  Total windows: {len(labels)}")
    print(f"  Unique stars: {meta['filename'].nunique()}")
    print(f"  Ground truth: 100% positive (all confirmed planets)")

    print(f"\nModel Predictions:")
    print(f"  Predicted positive: {predictions.sum()}/{len(predictions)} ({detection_rate:.1%})")
    print(f"  Predicted negative: {len(predictions) - predictions.sum()}/{len(predictions)} ({1-detection_rate:.1%})")
    print(f"  Mean probability: {mean_prob:.4f}")
    print(f"  Min probability: {probs.min():.4f}")
    print(f"  Max probability: {probs.max():.4f}")

    # For comparison, calculate what recall would be (since all are true positives)
    # If model predicts positive -> True Positive, if negative -> False Negative
    recall = predictions.sum() / len(predictions)  # Same as detection rate

    print(f"\nKey Metric (Recall on Kepler planets):")
    print(f"  Recall: {recall:.4f} ({recall:.1%})")
    print(f"  (This means the TESS model only detected {recall:.1%} of confirmed Kepler planets)")

    # Per-star analysis
    print("\n[4] Per-star analysis...")
    meta['probability'] = probs
    meta['prediction'] = predictions

    star_stats = meta.groupby('filename').agg({
        'probability': ['mean', 'max'],
        'prediction': 'sum',
        'label': 'count'
    }).reset_index()
    star_stats.columns = ['filename', 'mean_prob', 'max_prob', 'positive_windows', 'total_windows']

    detected_stars = (star_stats['positive_windows'] > 0).sum()
    print(f"  Stars with at least 1 positive prediction: {detected_stars}/{len(star_stats)} ({detected_stars/len(star_stats):.1%})")

    # Save results
    output_path = r"D:\CS_4280_Project\Code\reports\kepler_crosstest_predictions.csv"
    meta.to_csv(output_path, index=False)
    print(f"\n[5] Results saved to: {output_path}")

    # Summary for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    print(f"""
Dataset: 93 confirmed Kepler exoplanet host stars (279 windows)
Model: BiLSTM + Clustering trained on TESS Sector 1

Results:
- Recall: {recall:.1%} (only {predictions.sum()} of {len(predictions)} windows detected)
- Mean probability: {mean_prob:.4f}
- Stars detected: {detected_stars}/{len(star_stats)} ({detected_stars/len(star_stats):.1%})

Conclusion: Model trained on TESS data {'SUCCEEDED' if recall > 0.5 else 'FAILED'} to generalize to Kepler data.
The {1-recall:.1%} miss rate demonstrates {'minimal' if recall > 0.8 else 'significant'} domain shift between missions.
""")
    print("=" * 70)


if __name__ == '__main__':
    main()
