"""
Fine-tune TESS-trained model for Kepler domain adaptation.

This script implements the professor's recommendation:
- Start with a model trained on 100% TESS data
- Fine-tune on a mix of ~95% TESS + ~5% Kepler
- Use a lower learning rate for stable fine-tuning

This approach helps the model generalize across different missions
(TESS vs Kepler) which have different:
- Cadence (2 min vs 30 min)
- Noise characteristics
- Transit depths

Usage:
    python finetune_for_kepler.py
        --tess_model "runs/sector1_final_0918/best.pt"
        --tess_dir "data/windows_sector1_full/train"
        --kepler_dir "data/windows_kepler"
        --kepler_fraction 0.05
        --epochs 10
        --lr 0.00001
        --save_dir "runs/finetuned_kepler"
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# =============================================================================
# Model Architecture (must match training script exactly)
# =============================================================================

class ClusterBiLSTM(nn.Module):
    """BiLSTM with cluster embeddings for exoplanet detection."""

    def __init__(self, config):
        """
        Initialize model.

        Args:
            config: dict with keys: input_size, hidden_size, num_layers,
                   dropout, n_clusters, cluster_embed_dim
        """
        super().__init__()

        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']

        # Cluster embedding
        self.cluster_embed = nn.Embedding(
            config['n_clusters'],
            config['cluster_embed_dim']
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'] if config['num_layers'] > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # Classification head
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
# Data Classes
# =============================================================================

@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning."""
    tess_model_path: str
    tess_dir: str
    kepler_dir: str
    kepler_fraction: float
    epochs: int
    lr: float
    batch_size: int
    save_dir: str
    seed: int


# =============================================================================
# Data Loading
# =============================================================================

def load_windows(windows_dir):
    """Load windows, labels, and metadata from directory."""
    windows_dir = Path(windows_dir)

    flux_data = np.load(windows_dir / 'X.npy')
    labels = np.load(windows_dir / 'y.npy')
    meta = pd.read_csv(windows_dir / 'meta.csv')

    print(f"Loaded {len(flux_data)} windows from {windows_dir}")
    print(f"  Positive: {np.sum(labels == 1)}, Negative: {np.sum(labels == 0)}")

    return flux_data, labels, meta


def create_mixed_dataset(tess_data, kepler_data, kepler_fraction, seed=42):
    """
    Create a mixed dataset with specified Kepler fraction.

    Args:
        tess_data: tuple of (flux, labels, meta) from TESS
        kepler_data: tuple of (flux, labels, meta) from Kepler
        kepler_fraction: fraction of Kepler data (e.g., 0.05 for 5%)
        seed: random seed

    Returns:
        mixed_flux, mixed_labels, mixed_meta, source_flags
    """
    np.random.seed(seed)

    tess_flux, tess_labels, tess_meta = tess_data
    kepler_flux, kepler_labels, kepler_meta = kepler_data

    # Calculate how many of each to include
    # Target: kepler_fraction of total = kepler_size / (tess_size + kepler_size)
    # Solving: kepler_size = tess_size * kepler_fraction / (1 - kepler_fraction)
    n_tess = len(tess_flux)
    n_kepler_target = int(n_tess * kepler_fraction / (1 - kepler_fraction))
    n_kepler = min(n_kepler_target, len(kepler_flux))

    print(f"\nCreating mixed dataset:")
    print(f"  TESS windows: {n_tess}")
    print(f"  Kepler windows: {n_kepler} (target: {n_kepler_target})")
    print(f"  Kepler fraction: {n_kepler / (n_tess + n_kepler) * 100:.1f}%")

    # Sample Kepler data if needed
    if n_kepler < len(kepler_flux):
        kepler_idx = np.random.choice(len(kepler_flux), n_kepler, replace=False)
        kepler_flux = kepler_flux[kepler_idx]
        kepler_labels = kepler_labels[kepler_idx]
        kepler_meta = kepler_meta.iloc[kepler_idx].reset_index(drop=True)

    # Combine datasets
    mixed_flux = np.concatenate([tess_flux, kepler_flux], axis=0)
    mixed_labels = np.concatenate([tess_labels, kepler_labels], axis=0)

    # Track source
    source_flags = np.array(['tess'] * n_tess + ['kepler'] * n_kepler)

    # Shuffle
    shuffle_idx = np.random.permutation(len(mixed_flux))
    mixed_flux = mixed_flux[shuffle_idx]
    mixed_labels = mixed_labels[shuffle_idx]
    source_flags = source_flags[shuffle_idx]

    return mixed_flux, mixed_labels, source_flags


def get_stat_features(meta_df):
    """Extract statistical features from metadata for clustering."""
    feature_cols = ['mean', 'std', 'var', 'skew', 'range', 'median', 'mad', 'peak_to_peak']
    available_cols = [c for c in feature_cols if c in meta_df.columns]

    if not available_cols:
        print("Warning: No statistical features in metadata, computing from scratch")
        return None

    return meta_df[available_cols].values


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, train_loader, optimizer, criterion, device, scaler):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for flux_batch, label_batch, cluster_batch in train_loader:
        flux_batch = flux_batch.to(device)
        label_batch = label_batch.to(device)
        cluster_batch = cluster_batch.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(flux_batch, cluster_batch)
            loss = criterion(logits, label_batch)

        if torch.isnan(loss):
            print("NaN loss detected, skipping batch")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, data_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for flux_batch, label_batch, cluster_batch in data_loader:
            flux_batch = flux_batch.to(device)
            cluster_batch = cluster_batch.to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(flux_batch, cluster_batch)
                probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    if np.any(np.isnan(all_probs)):
        return {'auc': 0.0, 'f1': 0.0, 'accuracy': 0.0}

    preds = (all_probs > 0.5).astype(int)

    return {
        'auc': roc_auc_score(all_labels, all_probs),
        'f1': f1_score(all_labels, preds, zero_division=0),
        'accuracy': accuracy_score(all_labels, preds)
    }


# =============================================================================
# Fine-Tuning Pipeline
# =============================================================================

def finetune_model(config: FineTuneConfig):
    """
    Fine-tune a pre-trained TESS model on mixed TESS+Kepler data.

    Args:
        config: FineTuneConfig with all parameters
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-trained model
    print(f"\nLoading pre-trained model from {config.tess_model_path}")
    checkpoint = torch.load(config.tess_model_path, map_location=device)
    model_config = checkpoint['config']

    # Build config dict for model (map checkpoint keys to model keys)
    model_init_config = {
        'input_size': 1,
        'hidden_size': model_config['hidden'],
        'num_layers': model_config['layers'],
        'n_clusters': model_config['n_clusters'],
        'cluster_embed_dim': model_config.get('cluster_embed_dim', 64),
        'dropout': model_config['dropout']
    }

    # Recreate model with same architecture
    model = ClusterBiLSTM(model_init_config).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Original validation AUC: {checkpoint.get('val_metrics', {}).get('auc', 'unknown')}")

    # Get clustering configuration from checkpoint
    scaler_params = checkpoint['scaler_params']
    kmeans_centers = checkpoint['kmeans_centers']
    n_clusters = len(kmeans_centers)

    # Load TESS and Kepler data
    print("\nLoading TESS data...")
    tess_flux, tess_labels, tess_meta = load_windows(config.tess_dir)

    print("\nLoading Kepler data...")
    kepler_flux, kepler_labels, kepler_meta = load_windows(config.kepler_dir)

    # Create mixed dataset
    mixed_flux, mixed_labels, source_flags = create_mixed_dataset(
        (tess_flux, tess_labels, tess_meta),
        (kepler_flux, kepler_labels, kepler_meta),
        config.kepler_fraction,
        config.seed
    )

    # Compute statistical features for clustering (compute from flux data)
    # We refit K-means on the mixed data since Kepler lacks some TESS features
    print(f"\nComputing statistical features for clustering ({n_clusters} clusters)...")
    features = np.column_stack([
        np.mean(mixed_flux, axis=1),
        np.std(mixed_flux, axis=1),
        np.var(mixed_flux, axis=1),
        np.percentile(mixed_flux, 75, axis=1) - np.percentile(mixed_flux, 25, axis=1),
    ])

    # Clip outliers and scale
    features = np.clip(features, np.percentile(features, 1, axis=0),
                       np.percentile(features, 99, axis=0))
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Fit new K-means on mixed data (same number of clusters as original)
    kmeans = KMeans(n_clusters=n_clusters, random_state=config.seed, n_init=10)
    cluster_ids = kmeans.fit_predict(features_scaled)
    print(f"Cluster distribution: {np.bincount(cluster_ids)}")

    # Split into train/val (90/10)
    n_total = len(mixed_flux)
    n_train = int(0.9 * n_total)
    indices = np.random.permutation(n_total)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    # Create tensors
    train_flux = torch.tensor(mixed_flux[train_idx, :, np.newaxis], dtype=torch.float32)
    train_labels = torch.tensor(mixed_labels[train_idx], dtype=torch.float32)
    train_clusters = torch.tensor(cluster_ids[train_idx], dtype=torch.long)

    val_flux = torch.tensor(mixed_flux[val_idx, :, np.newaxis], dtype=torch.float32)
    val_labels = torch.tensor(mixed_labels[val_idx], dtype=torch.float32)
    val_clusters = torch.tensor(cluster_ids[val_idx], dtype=torch.long)

    # Create dataloaders
    train_dataset = TensorDataset(train_flux, train_labels, train_clusters)
    val_dataset = TensorDataset(val_flux, val_labels, val_clusters)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0)

    # Setup optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)

    # Calculate pos_weight for mixed dataset
    n_pos = np.sum(mixed_labels == 1)
    n_neg = np.sum(mixed_labels == 0)
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"\nClass balance: {n_pos} positive, {n_neg} negative")
    print(f"pos_weight: {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    scaler_amp = torch.amp.GradScaler('cuda')

    # Training loop
    print(f"\n{'='*70}")
    print("FINE-TUNING")
    print(f"{'='*70}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"Batch size: {config.batch_size}")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")

    best_auc = 0.0
    best_epoch = 0

    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 device, scaler_amp)
        val_metrics = evaluate(model, val_loader, device)

        print(f"[epoch {epoch+1:2d}/{config.epochs}] "
              f"loss={train_loss:.4f} "
              f"val_auc={val_metrics['auc']:.4f} "
              f"val_f1={val_metrics['f1']:.4f}", flush=True)

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_epoch = epoch + 1

            # Save best model (with new scaler/kmeans fitted on mixed data)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'config': model_config,
                'scaler_params': {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                },
                'kmeans_centers': kmeans.cluster_centers_.tolist(),
                'val_metrics': val_metrics,
                'finetune_config': {
                    'kepler_fraction': config.kepler_fraction,
                    'lr': config.lr,
                    'epochs': config.epochs
                }
            }, save_dir / 'best.pt')

            print(f"  [best] saved (AUC={best_auc:.4f})")

    # Save final model
    torch.save({
        'epoch': config.epochs,
        'model_state_dict': model.state_dict(),
        'config': model_config,
        'scaler_params': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist()
        },
        'kmeans_centers': kmeans.cluster_centers_.tolist(),
        'val_metrics': val_metrics
    }, save_dir / 'last.pt')

    # Save fine-tuning config
    with open(save_dir / 'finetune_config.json', 'w') as f:
        json.dump({
            'tess_model_path': config.tess_model_path,
            'kepler_fraction': config.kepler_fraction,
            'lr': config.lr,
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'best_epoch': best_epoch,
            'best_auc': best_auc,
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'n_tess': int(np.sum(source_flags == 'tess')),
            'n_kepler': int(np.sum(source_flags == 'kepler'))
        }, f, indent=2)

    print(f"\n{'='*70}")
    print("FINE-TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"Best validation AUC: {best_auc:.4f} (epoch {best_epoch})")
    print(f"Model saved to: {save_dir}")
    print(f"{'='*70}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune TESS model for Kepler domain adaptation'
    )
    parser.add_argument('--tess_model', type=str,
                        default=r'D:\CS_4280_Project\Code\runs\sector1_final_0918\best.pt',
                        help='Path to pre-trained TESS model')
    parser.add_argument('--tess_dir', type=str,
                        default=r'D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train',
                        help='Path to TESS training windows')
    parser.add_argument('--kepler_dir', type=str,
                        default=r'D:\CS_4280_Project\Code\data\windows_kepler',
                        help='Path to Kepler windows')
    parser.add_argument('--kepler_fraction', type=float, default=0.05,
                        help='Fraction of Kepler data in mixed dataset (default: 0.05)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate for fine-tuning (lower than original)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--save_dir', type=str,
                        default=r'D:\CS_4280_Project\Code\runs\finetuned_kepler',
                        help='Directory to save fine-tuned model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    config = FineTuneConfig(
        tess_model_path=args.tess_model,
        tess_dir=args.tess_dir,
        kepler_dir=args.kepler_dir,
        kepler_fraction=args.kepler_fraction,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        seed=args.seed
    )

    finetune_model(config)


if __name__ == '__main__':
    main()
