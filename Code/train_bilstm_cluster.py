"""
Clustering-Enhanced BiLSTM Training for Exoplanet Detection

Approach:
1. Cluster training windows based on features (period, depth, duration, BLS power)
2. Train BiLSTM with cluster information as additional context
3. Helps model learn different patterns for different stellar/noise types

Usage:
    python train_bilstm_cluster.py --windows_dir "C:\CS_4280_Project\Code\data\windows_train" --n_clusters 5 --epochs 80 --batch_size 64 --lr 1e-4 --hidden 256 --layers 3 --dropout 0.4 --save_dir "C:\CS_4280_Project\Code\runs\bilstm_cluster" --amp_dtype fp16 --pos_weight 3.367 --num_workers 0
"""

import os

# Limit CPU threads BEFORE importing numpy/torch to prevent system overload
# Using 11 threads (12 cores - 1 reserved for system)
os.environ['OMP_NUM_THREADS'] = '11'
os.environ['MKL_NUM_THREADS'] = '11'
os.environ['NUMEXPR_NUM_THREADS'] = '11'
os.environ['OPENBLAS_NUM_THREADS'] = '11'

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')


class LightCurveDataset(Dataset):
    """Dataset with cluster information"""
    
    def __init__(self, X, y, clusters):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.clusters = torch.from_numpy(clusters).long()
        
        if len(self.X.shape) == 2:
            self.X = self.X.unsqueeze(-1)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.clusters[idx]


class ClusterBiLSTM(nn.Module):
    """
    BiLSTM with cluster-aware processing
    Uses cluster embeddings to provide context
    """
    
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, 
                 dropout=0.4, n_clusters=5, cluster_embed_dim=32):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_clusters = n_clusters
        
        # Cluster embedding - learn representations for each cluster
        self.cluster_embed = nn.Embedding(n_clusters, cluster_embed_dim)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Classification head (includes cluster embedding)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2 + cluster_embed_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x, cluster_ids):
        # x: (batch, seq_len, features)
        # cluster_ids: (batch,)
        
        # Get cluster embeddings
        cluster_emb = self.cluster_embed(cluster_ids)  # (batch, cluster_embed_dim)
        
        # BiLSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Concatenate forward and backward hidden states
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Concatenate LSTM output with cluster embedding
        combined = torch.cat([hidden_cat, cluster_emb], dim=1)
        
        # Classification
        out = self.dropout(combined)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out.squeeze(-1)


def cluster_windows(meta_df, n_clusters=5):
    """
    Cluster windows based on features
    Uses: period, duration, depth, bls_power (if available)
    Or: category field (if BLS features not available)
    """
    print(f"\n[clustering] Creating {n_clusters} clusters based on features")

    # Check if BLS features are available
    # Check for ground truth features (BEST - actual injected parameters)
    gt_feature_cols = ['log_period', 'log_depth', 'duration', 'teff']
    has_gt_features = all(col in meta_df.columns for col in gt_feature_cols)

    bls_feature_cols = ['period', 'duration', 'depth', 'bls_power']
    has_bls_features = all(col in meta_df.columns for col in bls_feature_cols)

    # Check if statistical features are available
    stat_feature_cols = ['mean', 'std', 'var', 'skew', 'range', 'median', 'mad', 'peak_to_peak']
    has_stat_features = all(col in meta_df.columns for col in stat_feature_cols)

    if has_gt_features:
        # BEST: Use ground truth parameters from LILITH-4
        print("[clustering] Using GROUND TRUTH features (log_period, log_depth, duration, teff)")
        feature_cols = gt_feature_cols
        features = meta_df[feature_cols].values.copy()

        # Clip outliers
        print("[clustering] Applying robust preprocessing (clipping outliers to 1-99 percentile)")
        for i in range(features.shape[1]):
            col = features[:, i]
            p1, p99 = np.percentile(col, [1, 99])
            features[:, i] = np.clip(col, p1, p99)

        # Standardize features
        feature_scaler = StandardScaler()
        features_scaled = feature_scaler.fit_transform(features)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(features_scaled)

    elif has_bls_features:
        # Original approach: use BLS features for k-means clustering
        print("[clustering] Using BLS features (period, duration, depth, bls_power)")
        feature_cols = bls_feature_cols
        features = meta_df[feature_cols].values

        # Standardize features
        feature_scaler = StandardScaler()
        features_scaled = feature_scaler.fit_transform(features)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(features_scaled)

    elif has_stat_features:
        # Use statistical features from light curve data
        print("[clustering] Using statistical features (mean, std, var, skew, range, median, mad, peak_to_peak)")
        feature_cols = stat_feature_cols
        features = meta_df[feature_cols].values.copy()

        # Robust preprocessing: clip outliers to 1st-99th percentile
        print("[clustering] Applying robust preprocessing (clipping outliers to 1-99 percentile)")
        for i in range(features.shape[1]):
            col = features[:, i]
            p1, p99 = np.percentile(col, [1, 99])
            features[:, i] = np.clip(col, p1, p99)

        # Standardize features
        feature_scaler = StandardScaler()
        features_scaled = feature_scaler.fit_transform(features)

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(features_scaled)

    else:
        # Last resort: assign all to cluster 0
        print("[clustering] Warning: No clustering features available, assigning all to cluster 0")
        cluster_ids = np.zeros(len(meta_df), dtype=int)
        feature_scaler = None
        kmeans = None

    # Print cluster distribution
    print("[clustering] Cluster distribution:")
    unique, counts = np.unique(cluster_ids, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        pct = 100 * count / len(cluster_ids)
        print(f"  Cluster {cluster_id}: {count} windows ({pct:.1f}%)")

    # Print cluster characteristics
    print("\n[clustering] Cluster characteristics:")
    for cluster_id in unique:
        mask = cluster_ids == cluster_id
        cluster_data = meta_df[mask]
        # Convert string labels to numeric if needed
        if cluster_data['label'].dtype == object:
            numeric_labels = (cluster_data['label'] == 'planet').astype(int)
            pos_rate = numeric_labels.mean()
        else:
            pos_rate = cluster_data['label'].mean()

        print(f"  Cluster {cluster_id}:")
        print(f"    Positive rate: {pos_rate:.1%}")

        # Print category if available
        if 'category' in cluster_data.columns:
            category_counts = cluster_data['category'].value_counts()
            print(f"    Categories: {dict(category_counts)}")

        # Print feature statistics
        if has_bls_features:
            print(f"    Avg period: {cluster_data['period'].mean():.2f} days")
            print(f"    Avg depth: {cluster_data['depth'].mean():.4f}")
            print(f"    Avg BLS power: {cluster_data['bls_power'].mean():.4f}")
        elif has_stat_features:
            print(f"    Avg std: {cluster_data['std'].mean():.4f}")
            print(f"    Avg var: {cluster_data['var'].mean():.4f}")
            print(f"    Avg range: {cluster_data['range'].mean():.4f}")

    return cluster_ids, feature_scaler, kmeans


def load_data(windows_dir):
    """Load windowed data with metadata"""
    X = np.load(os.path.join(windows_dir, "X.npy"))
    y = np.load(os.path.join(windows_dir, "y.npy"))
    meta = pd.read_csv(os.path.join(windows_dir, "meta.csv"))
    
    print(f"[data] X.shape={X.shape} y.shape={y.shape}")
    print(f"[data] pos={int(y.sum())} neg={int(len(y) - y.sum())}")
    
    # Check for bad values
    if np.isnan(X).any():
        print("[warning] NaN values detected in X, replacing with 0")
        X = np.nan_to_num(X, nan=0.0)
    if np.isinf(X).any():
        print("[warning] Inf values detected in X, replacing with 0")
        X = np.nan_to_num(X, posinf=0.0, neginf=0.0)
    
    return X, y, meta


def split_data(X, y, clusters, val_split=0.15, seed=42):
    """Split data maintaining cluster distribution"""
    np.random.seed(seed)
    n = len(y)
    indices = np.random.permutation(n)
    
    val_size = int(n * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_train, y_train, clusters_train = X[train_idx], y[train_idx], clusters[train_idx]
    X_val, y_val, clusters_val = X[val_idx], y[val_idx], clusters[val_idx]
    
    return X_train, y_train, clusters_train, X_val, y_val, clusters_val


def compute_metrics(y_true, y_pred, y_scores):
    """Compute classification metrics"""
    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
    }
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype):
    """Evaluate model"""
    model.eval()
    
    all_labels = []
    all_scores = []
    
    for x_batch, y_batch, cluster_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        cluster_batch = cluster_batch.to(device)
        
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            logits = model(x_batch, cluster_batch)
            scores = torch.sigmoid(logits)
        
        all_labels.append(y_batch.cpu().numpy())
        all_scores.append(scores.cpu().numpy())
    
    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_scores)
    y_pred = (y_scores > 0.5).astype(int)
    
    metrics = compute_metrics(y_true, y_pred, y_scores)
    
    return metrics


def train_epoch(model, loader, criterion, optimizer, device, autocast_dtype, scaler, gradient_clip=1.0):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for x_batch, y_batch, cluster_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        cluster_batch = cluster_batch.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            logits = model(x_batch, cluster_batch)
            loss = criterion(logits, y_batch)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description='Train cluster-enhanced BiLSTM')
    
    # Data
    parser.add_argument('--windows_dir', type=str, required=True)
    parser.add_argument('--val_split', type=float, default=0.15)
    
    # Clustering
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters')
    parser.add_argument('--cluster_embed_dim', type=int, default=32,
                       help='Cluster embedding dimension')
    
    # Model
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.4)
    
    # Training
    parser.add_argument('--epochs', type=int, default=80,
                       help='Total epochs to reach (used when NOT resuming)')
    parser.add_argument('--epochs_to_train', type=int, default=None,
                       help='Number of epochs to train in THIS session (overrides --epochs when set)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., runs/experiment/last.pt)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--pos_weight', type=float, default=3.367)
    parser.add_argument('--gradient_clip', type=float, default=1.0)
    
    # System
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--amp_dtype', type=str, default='fp16',
                       choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_mem_fraction', type=float, default=None,
                       help='Limit GPU memory to this fraction (0.0-1.0). E.g., 0.2 for 20%')
    
    # Output
    parser.add_argument('--save_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # GPU memory limiting
    if args.gpu_mem_fraction is not None and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, device=0)
        print(f"[gpu] Memory limited to {args.gpu_mem_fraction*100:.0f}% of GPU")

    # AMP setup
    amp_flag = args.amp_dtype != 'fp32'
    if args.amp_dtype == 'fp16':
        autocast_dtype = torch.float16
    elif args.amp_dtype == 'bf16':
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    print(f"[runtime] device={device} amp={amp_flag} dtype={args.amp_dtype}")
    print(f"[runtime] model=Cluster-BiLSTM (n_clusters={args.n_clusters})")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data with metadata
    print("\n[loading data]")
    X, y, meta = load_data(args.windows_dir)
    
    # Cluster windows
    cluster_ids, feature_scaler, kmeans = cluster_windows(meta, args.n_clusters)

    # Determine actual number of clusters (may differ from args.n_clusters if using categories)
    actual_n_clusters = len(np.unique(cluster_ids))
    print(f"[clustering] Actual clusters used: {actual_n_clusters}")

    # Split data
    X_train, y_train, clusters_train, X_val, y_val, clusters_val = split_data(
        X, y, cluster_ids, args.val_split, args.seed
    )
    
    print(f"\n[split] train={len(y_train)} val={len(y_val)}")
    print(f"[train] pos={int(y_train.sum())} neg={int(len(y_train) - y_train.sum())}")
    print(f"[val] pos={int(y_val.sum())} neg={int(len(y_val) - y_val.sum())}")
    
    # Create datasets
    train_dataset = LightCurveDataset(X_train, y_train, clusters_train)
    val_dataset = LightCurveDataset(X_val, y_val, clusters_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    input_size = train_dataset.X.shape[-1]
    model = ClusterBiLSTM(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        n_clusters=actual_n_clusters,
        cluster_embed_dim=args.cluster_embed_dim
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[model] params={n_params:,}")
    print(f"[model] hidden={args.hidden} layers={args.layers} clusters={actual_n_clusters}")
    
    # Loss and optimizer
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[loss] BCEWithLogitsLoss pos_weight={args.pos_weight:.3f}")
    
    # Determine epoch range
    start_epoch = 1

    # Calculate total epochs and end epoch
    if args.epochs_to_train is not None:
        # epochs_to_train mode: train for N more epochs from current position
        total_epochs_for_scheduler = args.epochs_to_train + 100  # Large enough for scheduler
    else:
        total_epochs_for_scheduler = args.epochs

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs_for_scheduler, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=amp_flag)

    # Resume from checkpoint if specified
    best_auc = 0.0
    best_f1 = 0.0

    if args.resume:
        if os.path.exists(args.resume):
            print(f"\n[resume] Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint.get('epoch', 0) + 1

            # Restore best metrics if available
            if 'val_metrics' in checkpoint:
                best_auc = checkpoint['val_metrics'].get('auc', 0.0)
                best_f1 = checkpoint['val_metrics'].get('f1', 0.0)

            print(f"[resume] Resuming from epoch {start_epoch}, best_auc={best_auc:.4f}")
        else:
            print(f"[warning] Checkpoint not found: {args.resume}, starting fresh")

    # Calculate end epoch
    if args.epochs_to_train is not None:
        end_epoch = start_epoch + args.epochs_to_train - 1
        print(f"[epochs] Training epochs {start_epoch} to {end_epoch} ({args.epochs_to_train} epochs this session)")
    else:
        end_epoch = args.epochs
        print(f"[epochs] Training epochs {start_epoch} to {end_epoch}")

    # Save config and clustering info
    config = vars(args)
    config['cluster_info'] = {
        'n_clusters': args.n_clusters,
        'feature_cols': ['period', 'duration', 'depth', 'bls_power']
    }
    
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save clustering artifacts
    np.save(os.path.join(args.save_dir, 'cluster_ids.npy'), cluster_ids)
    
    # Initial validation (skip if resuming to save time)
    if start_epoch == 1:
        print("\n[validation before training]")
        val_metrics = evaluate(model, val_loader, device, autocast_dtype)
        print(f"[val0] {val_metrics}")
    else:
        print(f"\n[skipping initial validation - resuming from epoch {start_epoch}]")

    # Training loop
    patience = 15
    patience_counter = 0

    n_epochs_this_session = end_epoch - start_epoch + 1
    print(f"\n[training for {n_epochs_this_session} epochs (epoch {start_epoch} to {end_epoch})]")
    print(f"[early stopping] patience={patience} epochs")

    # Timing tracking
    epoch_times = []
    session_start = time.time()

    for epoch in range(start_epoch, end_epoch + 1):
        t0 = time.time()
        
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, autocast_dtype, scaler, args.gradient_clip
        )
        
        val_metrics = evaluate(model, val_loader, device, autocast_dtype)
        scheduler.step()
        
        dt = time.time() - t0
        epoch_times.append(dt)
        lr = optimizer.param_groups[0]['lr']

        # Format time nicely
        dt_min, dt_sec = divmod(int(dt), 60)

        print(f"[epoch {epoch:2d}/{end_epoch}] loss={train_loss:.4f} "
              f"auc={val_metrics['auc']:.4f} f1={val_metrics['f1']:.4f} "
              f"acc={val_metrics['acc']:.4f} time={dt_min}m{dt_sec:02d}s lr={lr:.3e}")
        
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_f1 = val_metrics['f1']
            patience_counter = 0
            
            # Build checkpoint dict
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }

            # Add clustering info if available
            if feature_scaler is not None:
                checkpoint['scaler_params'] = {'mean': feature_scaler.mean_.tolist(), 'scale': feature_scaler.scale_.tolist()}
            if kmeans is not None:
                checkpoint['kmeans_centers'] = kmeans.cluster_centers_.tolist()

            torch.save(checkpoint, os.path.join(args.save_dir, 'best.pt'))
            
            print(f"[best] auc={best_auc:.4f} f1={best_f1:.4f}; saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[early stopping] no improvement for {patience} epochs")
                break
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'config': config
        }, os.path.join(args.save_dir, 'last.pt'))
    
    # Training summary
    total_time = time.time() - session_start
    total_min, total_sec = divmod(int(total_time), 60)
    total_hr, total_min = divmod(total_min, 60)

    print(f"\n[training complete]")
    print(f"[best] AUC={best_auc:.4f} F1={best_f1:.4f}")
    print(f"[saved to] {args.save_dir}")

    # Timing summary
    print(f"\n[timing summary]")
    if epoch_times:
        avg_time = sum(epoch_times) / len(epoch_times)
        avg_min, avg_sec = divmod(int(avg_time), 60)
        print(f"  Epochs trained: {len(epoch_times)}")
        print(f"  Avg per epoch:  {avg_min}m{avg_sec:02d}s")
        if total_hr > 0:
            print(f"  Total time:     {total_hr}h{total_min:02d}m{total_sec:02d}s")
        else:
            print(f"  Total time:     {total_min}m{total_sec:02d}s")


if __name__ == '__main__':
    main()
