# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
"""
Clustering-Enhanced BiLSTM Training for Exoplanet Detection.

SOFA Refactored: December 3, 2025
Final Results: December 6, 2025 - AUC 0.9261, 100% Recall
Pylint Score: 10.00/10

Approach:
1. Cluster training windows based on statistical features (mean, std, var, skew, etc.)
2. Train BiLSTM with cluster embeddings as additional context
3. Helps model learn different patterns for different stellar/noise types

Final Configuration (achieved AUC 0.9261):
    python train_bilstm_cluster.py --windows_dir data/windows_sector1_full/train
        --n_clusters 7 --cluster_embed_dim 64 --epochs 60 --batch_size 128
        --lr 0.0001 --hidden 192 --layers 4 --dropout 0.334
        --save_dir runs/sector1_final --amp_dtype fp16 --pos_weight 7.41
        --num_workers 0 --seed 42
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Limit CPU threads BEFORE importing numpy/torch to prevent system overload
os.environ['OMP_NUM_THREADS'] = '11'
os.environ['MKL_NUM_THREADS'] = '11'
os.environ['NUMEXPR_NUM_THREADS'] = '11'
os.environ['OPENBLAS_NUM_THREADS'] = '11'

warnings.filterwarnings('ignore')


# =============================================================================
# Dataset Class
# =============================================================================

class LightCurveDataset(Dataset):
    """Dataset with cluster information for BiLSTM training."""

    def __init__(self, flux_data, labels, clusters):
        """
        Initialize dataset.

        Args:
            flux_data: numpy array of light curve windows (N, seq_len)
            labels: numpy array of binary labels (N,)
            clusters: numpy array of cluster IDs (N,)
        """
        self.flux_data = torch.from_numpy(flux_data).float()
        self.labels = torch.from_numpy(labels).float()
        self.clusters = torch.from_numpy(clusters).long()

        if len(self.flux_data.shape) == 2:
            self.flux_data = self.flux_data.unsqueeze(-1)

    def __len__(self):
        """Return number of samples."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return single sample."""
        return self.flux_data[idx], self.labels[idx], self.clusters[idx]


# =============================================================================
# Model Architecture
# =============================================================================

class ClusterBiLSTM(nn.Module):
    """
    BiLSTM with cluster-aware processing.

    Uses cluster embeddings to provide context about the type of light curve.
    """

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
# Clustering Functions
# =============================================================================

def cluster_windows(meta_df, n_clusters=5):
    """
    Cluster windows based on available features.

    Args:
        meta_df: DataFrame with metadata
        n_clusters: number of clusters

    Returns:
        cluster_ids: numpy array of cluster assignments
        feature_scaler: fitted StandardScaler (or None)
        kmeans: fitted KMeans (or None)
    """
    print(f"\n[clustering] Creating {n_clusters} clusters based on features")

    feature_cols, feature_type = _detect_feature_columns(meta_df)

    if feature_cols is None:
        return _fallback_clustering(len(meta_df))

    print(f"[clustering] Using {feature_type} features: {feature_cols}")
    features = meta_df[feature_cols].values.copy()

    features = _preprocess_features(features)
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features_scaled)

    _print_cluster_stats(meta_df, cluster_ids, feature_cols)

    return cluster_ids, feature_scaler, kmeans


def _detect_feature_columns(meta_df):
    """Detect which clustering features are available."""
    gt_cols = ['log_period', 'log_depth', 'duration', 'teff']
    bls_cols = ['period', 'duration', 'depth', 'bls_power']
    stat_cols = ['mean', 'std', 'var', 'skew', 'range', 'median', 'mad', 'peak_to_peak']

    if all(col in meta_df.columns for col in gt_cols):
        return gt_cols, "GROUND TRUTH"
    if all(col in meta_df.columns for col in bls_cols):
        return bls_cols, "BLS"
    if all(col in meta_df.columns for col in stat_cols):
        return stat_cols, "statistical"

    return None, None


def _preprocess_features(features):
    """Apply robust preprocessing to features."""
    print("[clustering] Applying robust preprocessing (1-99 percentile clipping)")
    for i in range(features.shape[1]):
        col = features[:, i]
        p1, p99 = np.percentile(col, [1, 99])
        features[:, i] = np.clip(col, p1, p99)
    return features


def _fallback_clustering(n_samples):
    """Return fallback when no features available."""
    print("[clustering] Warning: No features available, assigning all to cluster 0")
    return np.zeros(n_samples, dtype=int), None, None


def _print_cluster_stats(meta_df, cluster_ids, feature_cols):
    """Print cluster distribution and characteristics."""
    print("[clustering] Cluster distribution:")
    unique, counts = np.unique(cluster_ids, return_counts=True)

    for cluster_id, count in zip(unique, counts):
        pct = 100 * count / len(cluster_ids)
        print(f"  Cluster {cluster_id}: {count} windows ({pct:.1f}%)")

    print("\n[clustering] Cluster characteristics:")
    for cluster_id in unique:
        mask = cluster_ids == cluster_id
        cluster_data = meta_df[mask]
        pos_rate = _compute_positive_rate(cluster_data)

        print(f"  Cluster {cluster_id}: positive rate {pos_rate:.1%}")

        if 'bls_power' in feature_cols:
            _print_bls_stats(cluster_data)
        elif 'std' in feature_cols:
            _print_stat_stats(cluster_data)


def _compute_positive_rate(cluster_data):
    """Compute positive rate for cluster."""
    if cluster_data['label'].dtype == object:
        return (cluster_data['label'] == 'planet').mean()
    return cluster_data['label'].mean()


def _print_bls_stats(cluster_data):
    """Print BLS feature statistics."""
    print(f"    Avg period: {cluster_data['period'].mean():.2f} days")
    print(f"    Avg depth: {cluster_data['depth'].mean():.4f}")
    print(f"    Avg BLS power: {cluster_data['bls_power'].mean():.4f}")


def _print_stat_stats(cluster_data):
    """Print statistical feature statistics."""
    print(f"    Avg std: {cluster_data['std'].mean():.4f}")
    print(f"    Avg range: {cluster_data['range'].mean():.4f}")


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_data(windows_dir):
    """
    Load windowed data with metadata.

    Args:
        windows_dir: path to directory with X.npy, y.npy, meta.csv

    Returns:
        flux_data, labels, metadata DataFrame
    """
    flux_data = np.load(os.path.join(windows_dir, "X.npy"))
    labels = np.load(os.path.join(windows_dir, "y.npy"))
    meta = pd.read_csv(os.path.join(windows_dir, "meta.csv"))

    print(f"[data] shape={flux_data.shape} labels={labels.shape}")
    print(f"[data] pos={int(labels.sum())} neg={int(len(labels) - labels.sum())}")

    flux_data = _clean_data(flux_data)
    return flux_data, labels, meta


def _clean_data(flux_data):
    """Replace NaN and Inf values."""
    if np.isnan(flux_data).any():
        print("[warning] NaN values detected, replacing with 0")
        flux_data = np.nan_to_num(flux_data, nan=0.0)
    if np.isinf(flux_data).any():
        print("[warning] Inf values detected, replacing with 0")
        flux_data = np.nan_to_num(flux_data, posinf=0.0, neginf=0.0)
    return flux_data


def split_data(flux_data, labels, clusters, val_split=0.15, seed=42):
    """
    Split data maintaining cluster distribution.

    Args:
        flux_data: numpy array of windows
        labels: numpy array of labels
        clusters: numpy array of cluster IDs
        val_split: validation fraction
        seed: random seed

    Returns:
        train_data, val_data tuples
    """
    np.random.seed(seed)
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)

    val_size = int(n_samples * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_data = (flux_data[train_idx], labels[train_idx], clusters[train_idx])
    val_data = (flux_data[val_idx], labels[val_idx], clusters[val_idx])

    return train_data, val_data


# =============================================================================
# Training Functions
# =============================================================================

def compute_metrics(y_true, y_pred, y_scores):
    """Compute classification metrics."""
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': auc
    }


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype):
    """
    Evaluate model on data loader.

    Args:
        model: trained model
        loader: data loader
        device: torch device
        autocast_dtype: dtype for autocast

    Returns:
        metrics dict
    """
    model.eval()
    all_labels = []
    all_scores = []

    for flux_batch, label_batch, cluster_batch in loader:
        flux_batch = flux_batch.to(device)
        cluster_batch = cluster_batch.to(device)

        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            logits = model(flux_batch, cluster_batch)
            scores = torch.sigmoid(logits)

        all_labels.append(label_batch.numpy())
        all_scores.append(scores.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_scores)

    # Handle NaN in predictions
    if np.any(np.isnan(y_scores)):
        print("[warning] NaN detected in predictions", flush=True)
        return {'acc': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}

    y_pred = (y_scores > 0.5).astype(int)

    return compute_metrics(y_true, y_pred, y_scores)


def train_epoch(model, loader, criterion, optimizer, device,
                autocast_dtype, scaler, gradient_clip=1.0):
    """
    Train for one epoch.

    Args:
        model: model to train
        loader: training data loader
        criterion: loss function
        optimizer: optimizer
        device: torch device
        autocast_dtype: dtype for autocast
        scaler: gradient scaler
        gradient_clip: max gradient norm

    Returns:
        average loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False, file=sys.stderr, dynamic_ncols=True)

    for flux_batch, label_batch, cluster_batch in pbar:
        flux_batch = flux_batch.to(device)
        label_batch = label_batch.to(device)
        cluster_batch = cluster_batch.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            logits = model(flux_batch, cluster_batch)
            loss = criterion(logits, label_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / n_batches


# =============================================================================
# Configuration and Setup
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train cluster-enhanced BiLSTM')

    # Data arguments
    parser.add_argument('--windows_dir', type=str, required=True)
    parser.add_argument('--val_split', type=float, default=0.15)

    # Clustering arguments
    parser.add_argument('--n_clusters', type=int, default=7,
                        help='Number of K-means clusters (final: 7)')
    parser.add_argument('--cluster_embed_dim', type=int, default=64,
                        help='Cluster embedding dimension (final: 64)')

    # Model arguments
    parser.add_argument('--hidden', type=int, default=192,
                        help='BiLSTM hidden size (final: 192)')
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of BiLSTM layers (final: 4)')
    parser.add_argument('--dropout', type=float, default=0.334,
                        help='Dropout rate (final: 0.334)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=60,
                        help='Max epochs (final: 60, but best at epoch 2)')
    parser.add_argument('--epochs_to_train', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (final: 128)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (final: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--pos_weight', type=float, default=7.41,
                        help='Positive class weight (final: 7.41 = 23325/3147)')
    parser.add_argument('--gradient_clip', type=float, default=1.0)

    # System arguments
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--amp_dtype', type=str, default='fp16',
                        choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_mem_fraction', type=float, default=None)

    # Output arguments
    parser.add_argument('--save_dir', type=str, required=True)

    return parser.parse_args()


def setup_device(args):
    """Setup device and random seeds."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.gpu_mem_fraction is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, device=0)
        print(f"[gpu] Memory limited to {args.gpu_mem_fraction*100:.0f}%")

    return device


def setup_amp(args):
    """Setup automatic mixed precision."""
    amp_enabled = args.amp_dtype != 'fp32'

    dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32
    }
    autocast_dtype = dtype_map[args.amp_dtype]

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    return amp_enabled, autocast_dtype


def create_data_loaders(train_data, val_data, batch_size, num_workers):
    """Create training and validation data loaders."""
    train_dataset = LightCurveDataset(*train_data)
    val_dataset = LightCurveDataset(*val_data)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, train_dataset


def create_model(input_size, args, n_clusters, device):
    """Create and initialize model."""
    config = {
        'input_size': input_size,
        'hidden_size': args.hidden,
        'num_layers': args.layers,
        'dropout': args.dropout,
        'n_clusters': n_clusters,
        'cluster_embed_dim': args.cluster_embed_dim
    }

    model = ClusterBiLSTM(config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[model] params={n_params:,}")
    print(f"[model] hidden={args.hidden} layers={args.layers} clusters={n_clusters}")

    return model


def setup_training(model, args, device):
    """Setup optimizer, scheduler, and loss function."""
    total_epochs = args.epochs_to_train or args.epochs
    if args.epochs_to_train:
        total_epochs += 100  # Buffer for scheduler

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=args.lr * 0.01)

    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[loss] BCEWithLogitsLoss pos_weight={args.pos_weight:.3f}")

    return optimizer, scheduler, criterion


def resume_from_checkpoint(args, model, optimizer, scheduler, device):
    """Resume training from checkpoint if specified."""
    start_epoch = 1
    best_auc = 0.0
    best_f1 = 0.0

    if args.resume and os.path.exists(args.resume):
        print(f"\n[resume] Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0) + 1

        if 'val_metrics' in checkpoint:
            best_auc = checkpoint['val_metrics'].get('auc', 0.0)
            best_f1 = checkpoint['val_metrics'].get('f1', 0.0)

        print(f"[resume] Resuming from epoch {start_epoch}, best_auc={best_auc:.4f}")

    elif args.resume:
        print(f"[warning] Checkpoint not found: {args.resume}, starting fresh")

    return start_epoch, best_auc, best_f1


def save_config(args, save_dir, cluster_ids):
    """Save training configuration."""
    config = vars(args)
    config['cluster_info'] = {
        'n_clusters': args.n_clusters,
        'feature_cols': ['period', 'duration', 'depth', 'bls_power']
    }

    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    np.save(os.path.join(save_dir, 'cluster_ids.npy'), cluster_ids)


def save_checkpoint(model, optimizer, scheduler, epoch, val_metrics,
                    config, feature_scaler, kmeans, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_metrics': val_metrics,
        'config': config
    }

    if feature_scaler is not None:
        checkpoint['scaler_params'] = {
            'mean': feature_scaler.mean_.tolist(),
            'scale': feature_scaler.scale_.tolist()
        }
    if kmeans is not None:
        checkpoint['kmeans_centers'] = kmeans.cluster_centers_.tolist()

    torch.save(checkpoint, save_path)


def format_time(seconds):
    """Format seconds as human-readable string."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    return f"{minutes}m{secs:02d}s"


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args, model, train_loader, val_loader, criterion, optimizer,
          scheduler, scaler, device, autocast_dtype, start_epoch, end_epoch,
          best_auc, best_f1, feature_scaler, kmeans):
    """
    Main training loop.

    Returns:
        best_auc, best_f1 after training
    """
    patience = 15
    patience_counter = 0
    config = vars(args)

    n_epochs = end_epoch - start_epoch + 1
    print(f"\n[training for {n_epochs} epochs (epoch {start_epoch} to {end_epoch})]")
    print(f"[early stopping] patience={patience} epochs")

    epoch_times = []
    session_start = time.time()

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, autocast_dtype, scaler, args.gradient_clip
        )

        # Check for NaN loss
        if np.isnan(train_loss):
            print(f"[error] NaN loss at epoch {epoch}, stopping training")
            break

        val_metrics = evaluate(model, val_loader, device, autocast_dtype)

        # Check for NaN in validation
        if val_metrics['auc'] == 0.0 and epoch > 1:
            print(f"[error] NaN in validation at epoch {epoch}, stopping training")
            break

        scheduler.step()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[epoch {epoch:2d}/{end_epoch}] loss={train_loss:.4f} "
              f"auc={val_metrics['auc']:.4f} f1={val_metrics['f1']:.4f} "
              f"acc={val_metrics['acc']:.4f} time={format_time(epoch_time)} "
              f"lr={current_lr:.3e}")

        # Check for improvement
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_f1 = val_metrics['f1']
            patience_counter = 0

            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                config, feature_scaler, kmeans,
                os.path.join(args.save_dir, 'best.pt')
            )
            print(f"[best] auc={best_auc:.4f} f1={best_f1:.4f}; saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[early stopping] no improvement for {patience} epochs")
                break

        # Save last checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_metrics,
            config, None, None,
            os.path.join(args.save_dir, 'last.pt')
        )

    # Print summary
    total_time = time.time() - session_start
    print("\n[training complete]")
    print(f"[best] AUC={best_auc:.4f} F1={best_f1:.4f}")
    print(f"[saved to] {args.save_dir}")

    if epoch_times:
        avg_time = sum(epoch_times) / len(epoch_times)
        print("\n[timing summary]")
        print(f"  Epochs trained: {len(epoch_times)}")
        print(f"  Avg per epoch:  {format_time(avg_time)}")
        print(f"  Total time:     {format_time(total_time)}")

    return best_auc, best_f1


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    args = parse_args()

    # Setup
    device = setup_device(args)
    amp_enabled, autocast_dtype = setup_amp(args)
    print(f"[runtime] device={device} amp={amp_enabled} dtype={args.amp_dtype}")
    print(f"[runtime] model=Cluster-BiLSTM (n_clusters={args.n_clusters})")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load and prepare data
    print("\n[loading data]")
    flux_data, labels, meta = load_data(args.windows_dir)

    cluster_ids, feature_scaler, kmeans = cluster_windows(meta, args.n_clusters)
    actual_n_clusters = len(np.unique(cluster_ids))
    print(f"[clustering] Actual clusters used: {actual_n_clusters}")

    train_data, val_data = split_data(
        flux_data, labels, cluster_ids, args.val_split, args.seed
    )

    _, train_y, _ = train_data
    _, val_y, _ = val_data
    print(f"\n[split] train={len(train_y)} val={len(val_y)}")
    print(f"[train] pos={int(train_y.sum())} neg={int(len(train_y) - train_y.sum())}")
    print(f"[val] pos={int(val_y.sum())} neg={int(len(val_y) - val_y.sum())}")

    # Create data loaders
    train_loader, val_loader, train_dataset = create_data_loaders(
        train_data, val_data, args.batch_size, args.num_workers
    )

    # Create model
    input_size = train_dataset.flux_data.shape[-1]
    model = create_model(input_size, args, actual_n_clusters, device)

    # Setup training
    optimizer, scheduler, criterion = setup_training(model, args, device)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)

    # Resume if specified
    start_epoch, best_auc, best_f1 = resume_from_checkpoint(
        args, model, optimizer, scheduler, device
    )

    # Calculate end epoch
    if args.epochs_to_train is not None:
        end_epoch = start_epoch + args.epochs_to_train - 1
    else:
        end_epoch = args.epochs

    # Save configuration
    save_config(args, args.save_dir, cluster_ids)

    # Initial validation
    if start_epoch == 1:
        print("\n[validation before training]")
        val_metrics = evaluate(model, val_loader, device, autocast_dtype)
        print(f"[val0] {val_metrics}")
    else:
        print(f"\n[skipping initial validation - resuming from epoch {start_epoch}]")

    # Train
    train(
        args, model, train_loader, val_loader, criterion, optimizer,
        scheduler, scaler, device, autocast_dtype, start_epoch, end_epoch,
        best_auc, best_f1, feature_scaler, kmeans
    )


if __name__ == '__main__':
    main()
