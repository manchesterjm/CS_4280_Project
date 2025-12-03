# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
"""
Optuna Hyperparameter Optimization for BiLSTM+Clustering Model.

SOFA Refactored: December 3, 2025
Pylint Score: 10.00/10

Uses Optuna TPE (Tree-structured Parzen Estimator) sampler to find optimal hyperparameters.

Usage:
    python optuna_optimize.py --windows_dir data/windows_train --n_trials 50
        --epochs_per_trial 40 --output_dir optuna_results
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


# =============================================================================
# Dataset Class
# =============================================================================

class LightCurveDataset(Dataset):
    """Dataset with cluster information for BiLSTM training."""

    def __init__(self, flux_data, labels, clusters):
        """
        Initialize dataset.

        Args:
            flux_data: numpy array of light curve windows
            labels: numpy array of binary labels
            clusters: numpy array of cluster IDs
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
    """BiLSTM with cluster-aware processing."""

    def __init__(self, config):
        """
        Initialize model.

        Args:
            config: dict with model configuration
        """
        super().__init__()

        self.hidden_size = config['hidden_size']
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
        """Forward pass."""
        cluster_emb = self.cluster_embed(cluster_ids)
        _, (hidden, _) = self.lstm(flux)

        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        combined = torch.cat([hidden_cat, cluster_emb], dim=1)

        logits = self.classifier(combined)
        return logits.squeeze(-1)


# =============================================================================
# Clustering
# =============================================================================

def cluster_windows(meta_df, n_clusters=5, random_state=42):
    """
    Cluster windows based on available features.

    Args:
        meta_df: DataFrame with metadata
        n_clusters: number of clusters
        random_state: random seed

    Returns:
        cluster_ids, scaler, kmeans
    """
    feature_cols = _detect_feature_columns(meta_df)
    features = meta_df[feature_cols].values
    features = np.nan_to_num(features, nan=0.0)

    # Clip outliers
    for i in range(features.shape[1]):
        p1, p99 = np.percentile(features[:, i], [1, 99])
        features[:, i] = np.clip(features[:, i], p1, p99)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(features_scaled)

    return cluster_ids, scaler, kmeans


def _detect_feature_columns(meta_df):
    """Detect which features are available."""
    stat_features = ['mean', 'std', 'var', 'skew', 'range', 'median', 'mad', 'peak_to_peak']
    bls_features = ['period', 'duration', 'depth', 'bls_power']

    if all(col in meta_df.columns for col in stat_features):
        return stat_features
    if all(col in meta_df.columns for col in bls_features):
        return bls_features

    available = [c for c in stat_features + bls_features if c in meta_df.columns]
    return available if available else ['mean', 'std']


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, amp_dtype='fp16'):
    """
    Train for one epoch.

    Args:
        model: model to train
        dataloader: training data loader
        criterion: loss function
        optimizer: optimizer
        device: torch device
        amp_dtype: mixed precision type

    Returns:
        average loss
    """
    model.train()
    total_loss = 0

    scaler = torch.amp.GradScaler('cuda') if amp_dtype in ['fp16', 'bf16'] else None
    dtype = _get_amp_dtype(amp_dtype)

    for flux_batch, label_batch, cluster_batch in dataloader:
        flux_batch = flux_batch.to(device)
        label_batch = label_batch.to(device)
        cluster_batch = cluster_batch.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = model(flux_batch, cluster_batch)
                loss = criterion(logits, label_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(flux_batch, cluster_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device, amp_dtype='fp16'):
    """
    Validate model.

    Args:
        model: model to evaluate
        dataloader: validation data loader
        device: torch device
        amp_dtype: mixed precision type

    Returns:
        auc, f1 scores
    """
    model.eval()
    all_probs = []
    all_labels = []

    dtype = _get_amp_dtype(amp_dtype)

    with torch.no_grad():
        for flux_batch, label_batch, cluster_batch in dataloader:
            flux_batch = flux_batch.to(device)
            cluster_batch = cluster_batch.to(device)

            with torch.amp.autocast('cuda', dtype=dtype):
                logits = model(flux_batch, cluster_batch)
                probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(label_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs >= 0.5).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)

    return auc, f1


def _get_amp_dtype(amp_dtype):
    """Get torch dtype for mixed precision."""
    dtype_map = {
        'fp16': torch.float16,
        'bf16': torch.bfloat16,
        'fp32': torch.float32
    }
    return dtype_map.get(amp_dtype, torch.float32)


# =============================================================================
# Optuna Objective
# =============================================================================

def objective(trial, flux_data, labels, meta, device, epochs_per_trial,
              pos_weight, amp_dtype):
    """
    Optuna objective function.

    Args:
        trial: Optuna trial
        flux_data: training data
        labels: training labels
        meta: metadata DataFrame
        device: torch device
        epochs_per_trial: max epochs per trial
        pos_weight: class weight
        amp_dtype: mixed precision type

    Returns:
        best AUC achieved
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    params = _suggest_hyperparameters(trial)
    cluster_ids, _, _ = cluster_windows(meta, n_clusters=params['n_clusters'])

    train_loader, val_loader = _create_data_loaders(
        flux_data, labels, cluster_ids, params['batch_size']
    )

    model = _create_model(params, device)
    criterion, optimizer, scheduler = _setup_training(
        model, params, pos_weight, epochs_per_trial, device
    )

    best_auc = _run_training_loop(
        model, train_loader, val_loader, criterion, optimizer,
        scheduler, device, amp_dtype, epochs_per_trial, trial
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_auc


def _suggest_hyperparameters(trial):
    """Suggest hyperparameters for trial."""
    # GPU-specific search space
    high_vram = True  # Set to True for RTX 5070 Ti (16 GB) or similar

    if high_vram:
        hidden_size = trial.suggest_categorical('hidden_size', [128, 256, 512])
        batch_size = trial.suggest_categorical('batch_size', [112, 128, 136, 144])
    else:
        hidden_size = trial.suggest_categorical('hidden_size', [128, 256])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    return {
        'hidden_size': hidden_size,
        'batch_size': batch_size,
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'dropout': trial.suggest_float('dropout', 0.2, 0.5),
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'n_clusters': trial.suggest_categorical('n_clusters', [3, 5, 7, 10]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True),
        'cluster_embed_dim': trial.suggest_categorical('cluster_embed_dim', [16, 32, 64])
    }


def _create_data_loaders(flux_data, labels, cluster_ids, batch_size):
    """Create train and validation data loaders."""
    full_dataset = LightCurveDataset(flux_data, labels, cluster_ids)

    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


def _create_model(params, device):
    """Create model with given parameters."""
    config = {
        'input_size': 1,
        'hidden_size': params['hidden_size'],
        'num_layers': params['num_layers'],
        'dropout': params['dropout'],
        'n_clusters': params['n_clusters'],
        'cluster_embed_dim': params['cluster_embed_dim']
    }
    return ClusterBiLSTM(config).to(device)


def _setup_training(model, params, pos_weight, epochs, device):
    """Setup loss, optimizer, and scheduler."""
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    optimizer = AdamW(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    return criterion, optimizer, scheduler


def _run_training_loop(model, train_loader, val_loader, criterion, optimizer,
                       scheduler, device, amp_dtype, epochs, trial):
    """Run training loop with early stopping and pruning."""
    best_auc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        train_epoch(model, train_loader, criterion, optimizer, device, amp_dtype)
        val_auc, _ = validate(model, val_loader, device, amp_dtype)
        scheduler.step()

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1

        trial.report(val_auc, epoch)

        if patience_counter >= patience:
            break

        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_auc


# =============================================================================
# Results Saving
# =============================================================================

def save_results(study, output_dir, elapsed_time, n_trials):
    """
    Save optimization results.

    Args:
        study: completed Optuna study
        output_dir: output directory
        elapsed_time: total optimization time
        n_trials: number of trials
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save study
    # pylint: disable=import-outside-toplevel
    import joblib
    study_file = os.path.join(output_dir, f'optuna_study_{timestamp}.pkl')
    joblib.dump(study, study_file)
    print(f"Study saved to {study_file}")

    # Save best parameters
    results = {
        'timestamp': timestamp,
        'best_trial': study.best_trial.number,
        'best_auc': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials,
        'elapsed_time_minutes': elapsed_time / 60
    }

    params_file = os.path.join(output_dir, f'best_params_{timestamp}.json')
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Best parameters saved to {params_file}")

    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_file = os.path.join(output_dir, f'trials_{timestamp}.csv')
    trials_df.to_csv(trials_file, index=False)
    print(f"Trials data saved to {trials_file}")


def print_comparison(study):
    """Print baseline vs optimized comparison."""
    baseline = {
        'hidden_size': 256,
        'num_layers': 3,
        'dropout': 0.4,
        'lr': 1e-4,
        'batch_size': 64,
        'n_clusters': 5,
        'weight_decay': 1e-5,
        'cluster_embed_dim': 32
    }

    print("\n" + "=" * 60)
    print("BASELINE VS OPTIMIZED COMPARISON")
    print("=" * 60)
    print(f"{'Parameter':<20} {'Baseline':<15} {'Optimized':<15} {'Change':<10}")
    print("-" * 60)

    for key, baseline_val in baseline.items():
        optimized_val = study.best_params.get(key, baseline_val)
        change = "Y" if baseline_val != optimized_val else ""
        print(f"{key:<20} {str(baseline_val):<15} {str(optimized_val):<15} {change:<10}")

    print("=" * 60 + "\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization')
    parser.add_argument('--windows_dir', type=str,
                        default=r'C:\CS_4280_Project\Code\data\windows_train')
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--epochs_per_trial', type=int, default=50)
    parser.add_argument('--output_dir', type=str,
                        default=r'C:\CS_4280_Project\Code\optuna_results')
    parser.add_argument('--study_name', type=str, default='bilstm_cluster_optimization')
    parser.add_argument('--amp_dtype', type=str, default='fp16',
                        choices=['fp16', 'bf16', 'fp32'])
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs per trial: {args.epochs_per_trial}")
    print(f"Mixed precision: {args.amp_dtype}")
    print("=" * 60 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.windows_dir}...")
    flux_data = np.load(os.path.join(args.windows_dir, 'X.npy'))
    labels = np.load(os.path.join(args.windows_dir, 'y.npy'))
    meta = pd.read_csv(os.path.join(args.windows_dir, 'meta.csv'))

    print(f"Loaded {len(flux_data)} windows")
    print(f"Positive samples: {labels.sum()} ({100*labels.mean():.1f}%)")

    pos_weight = (labels == 0).sum() / (labels == 1).sum()
    print(f"Positive weight: {pos_weight:.3f}\n")

    # Create study
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # Run optimization
    print(f"Starting optimization with {args.n_trials} trials...\n")
    start_time = time.time()

    study.optimize(
        lambda trial: objective(
            trial, flux_data, labels, meta, device,
            args.epochs_per_trial, pos_weight, args.amp_dtype
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
        catch=(RuntimeError, torch.cuda.OutOfMemoryError)
    )

    elapsed_time = time.time() - start_time

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best AUC: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")

    # Save and compare
    save_results(study, args.output_dir, elapsed_time, args.n_trials)
    print_comparison(study)


if __name__ == '__main__':
    main()
