# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
"""
Inference script for ClusterBiLSTM model.

SOFA Refactored: December 3, 2025
Pylint Score: 10.00/10

Runs predictions on test windows using a trained ClusterBiLSTM model.

Usage:
    python inference_cluster_model.py --model_path runs/bilstm_cluster/best.pt
        --windows_dir data/windows_test --output_file reports/predictions.csv
"""

import argparse
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch import nn

warnings.filterwarnings('ignore')


# =============================================================================
# Model Architecture (must match training)
# =============================================================================

class ClusterBiLSTM(nn.Module):
    """BiLSTM with cluster-aware processing."""

    def __init__(self, config):
        """
        Initialize model.

        Args:
            config: dict with keys: input_size, hidden_size, num_layers,
                   dropout, n_clusters, cluster_embed_dim
        """
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

        combined_size = config['hidden_size'] * 2 + config['cluster_embed_dim']
        self.classifier = self._build_classifier(
            combined_size,
            config['hidden_size'],
            config['dropout']
        )

    def _build_classifier(self, input_dim, hidden_dim, dropout):
        """Build classification head."""
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, flux, cluster_ids):
        """
        Forward pass.

        Args:
            flux: (batch, seq_len, features) light curve data
            cluster_ids: (batch,) cluster assignments

        Returns:
            logits: (batch,) classification logits
        """
        cluster_emb = self.cluster_embed(cluster_ids)
        _, (hidden, _) = self.lstm(flux)

        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        combined = torch.cat([hidden_cat, cluster_emb], dim=1)

        logits = self.classifier(combined)
        return logits.squeeze(-1)


# =============================================================================
# Clustering Functions
# =============================================================================

def assign_clusters(meta_df, checkpoint):
    """
    Assign cluster IDs to windows using saved KMeans.

    Args:
        meta_df: DataFrame with metadata
        checkpoint: loaded model checkpoint

    Returns:
        numpy array of cluster IDs
    """
    scaler_params = checkpoint.get('scaler_params')
    kmeans_centers = checkpoint.get('kmeans_centers')

    if scaler_params is None or kmeans_centers is None:
        print("[warning] No clustering info in checkpoint, using cluster 0")
        return np.zeros(len(meta_df), dtype=int)

    feature_cols = ['period', 'duration', 'depth', 'bls_power']

    if not all(col in meta_df.columns for col in feature_cols):
        print("[warning] Meta missing BLS features, using cluster 0")
        return np.zeros(len(meta_df), dtype=int)

    features = meta_df[feature_cols].values

    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params['mean'])
    scaler.scale_ = np.array(scaler_params['scale'])
    features_scaled = scaler.transform(features)

    kmeans = KMeans(n_clusters=len(kmeans_centers), random_state=42)
    kmeans.cluster_centers_ = np.array(kmeans_centers)
    # pylint: disable=protected-access
    kmeans._n_threads = 1
    cluster_ids = kmeans.predict(features_scaled)

    return cluster_ids


# =============================================================================
# Data Loading
# =============================================================================

def load_data(windows_dir):
    """
    Load windows and metadata.

    Args:
        windows_dir: path to directory with X.npy and meta.csv

    Returns:
        flux_data, meta DataFrame
    """
    flux_data = np.load(os.path.join(windows_dir, 'X.npy'))
    meta = pd.read_csv(os.path.join(windows_dir, 'meta.csv'))

    print(f"[data] shape={flux_data.shape}, windows={len(flux_data)}")
    print(f"[data] unique TIC={meta['tic_id'].nunique()}")

    return flux_data, meta


def load_model(model_path, device):
    """
    Load model from checkpoint.

    Args:
        model_path: path to checkpoint file
        device: torch device

    Returns:
        model, checkpoint dict
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    print(f"[config] hidden={config['hidden']}, layers={config['layers']}, "
          f"clusters={config['n_clusters']}")

    model_config = {
        'input_size': 1,
        'hidden_size': config['hidden'],
        'num_layers': config['layers'],
        'dropout': config['dropout'],
        'n_clusters': config['n_clusters'],
        'cluster_embed_dim': config.get('cluster_embed_dim', 32)
    }

    model = ClusterBiLSTM(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("[model] loaded successfully")

    return model, checkpoint


# =============================================================================
# Inference
# =============================================================================

def run_inference(model, flux_data, cluster_ids, batch_size, device):
    """
    Run inference on data.

    Args:
        model: trained model
        flux_data: numpy array of windows
        cluster_ids: numpy array of cluster assignments
        batch_size: batch size for inference
        device: torch device

    Returns:
        numpy array of probabilities
    """
    flux_tensor = torch.from_numpy(flux_data).float().unsqueeze(-1)
    cluster_tensor = torch.from_numpy(cluster_ids).long()

    all_probs = []

    with torch.no_grad():
        for i in range(0, len(flux_tensor), batch_size):
            batch_flux = flux_tensor[i:i+batch_size].to(device)
            batch_clusters = cluster_tensor[i:i+batch_size].to(device)

            logits = model(batch_flux, batch_clusters)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs)


def save_results(meta, probs, cluster_ids, output_file):
    """
    Save prediction results.

    Args:
        meta: metadata DataFrame
        probs: prediction probabilities
        cluster_ids: cluster assignments
        output_file: output path
    """
    results = meta.copy()
    results['probability'] = probs
    results['prediction'] = (probs > 0.5).astype(int)
    results['cluster_id'] = cluster_ids

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"\n[results saved to {output_path}]")

    return results


def print_summary(results, probs):
    """Print prediction summary."""
    print("\n" + "=" * 70)
    print(" PREDICTION SUMMARY")
    print("=" * 70)

    for tic_id in sorted(results['tic_id'].unique()):
        tic_results = results[results['tic_id'] == tic_id]
        max_prob = tic_results['probability'].max()
        n_positive = (tic_results['probability'] > 0.5).sum()

        print(f"\nTIC {tic_id}:")
        print(f"  Windows: {len(tic_results)}")
        print(f"  Max probability: {max_prob:.4f}")
        print(f"  Positive predictions: {n_positive}/{len(tic_results)}")

        if max_prob > 0.5:
            print("  PLANET CANDIDATE!")

    print("\n" + "=" * 70)
    print(f"Total windows: {len(results)}")
    print(f"Predicted positives: {(probs > 0.5).sum()}")
    print(f"Mean probability: {probs.mean():.4f}")
    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with ClusterBiLSTM')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--windows_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='predictions.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[runtime] device={device}")

    # Load data
    print(f"\n[loading data from {args.windows_dir}]")
    flux_data, meta = load_data(args.windows_dir)

    # Load model
    print(f"\n[loading model from {args.model_path}]")
    model, checkpoint = load_model(args.model_path, device)

    # Assign clusters
    print("\n[assigning clusters]")
    cluster_ids = assign_clusters(meta, checkpoint)
    print(f"[clusters] assigned (unique: {len(np.unique(cluster_ids))})")

    # Run inference
    print("\n[running inference]")
    probs = run_inference(model, flux_data, cluster_ids, args.batch_size, device)

    # Save and print results
    results = save_results(meta, probs, cluster_ids, args.output_file)
    print_summary(results, probs)


if __name__ == '__main__':
    main()
