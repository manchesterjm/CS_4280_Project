r"""
Optuna Hyperparameter Optimization for BiLSTM+Clustering Model

Uses Optuna TPE (Tree-structured Parzen Estimator) sampler to find optimal hyperparameters.

Usage:
    python optuna_optimize.py --windows_dir "C:\CS_4280_Project\Code\data\windows_train" --n_trials 50 --epochs_per_trial 40 --output_dir "C:\CS_4280_Project\Code\optuna_results"
"""

import argparse
import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


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
    """BiLSTM with cluster-aware processing"""

    def __init__(self, input_size=1, hidden_size=256, num_layers=3,
                 dropout=0.4, n_clusters=5, cluster_embed_dim=32):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_clusters = n_clusters

        # Cluster embedding
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

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2 + cluster_embed_dim, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x, cluster_ids):
        cluster_emb = self.cluster_embed(cluster_ids)
        lstm_out, (hidden, cell) = self.lstm(x)

        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)

        combined = torch.cat([hidden_cat, cluster_emb], dim=1)

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


def cluster_windows(meta_df, n_clusters=5, random_state=42):
    """Cluster windows based on features"""
    feature_cols = ['period', 'duration', 'depth', 'bls_power']
    features = meta_df[feature_cols].values

    # Fill NaN values with 0 (for non-planet windows)
    features = np.nan_to_num(features, nan=0.0)

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(features_scaled)

    return cluster_ids, scaler, kmeans


def train_epoch(model, dataloader, criterion, optimizer, device, amp_dtype='fp16'):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    scaler = torch.amp.GradScaler('cuda') if amp_dtype in ['fp16', 'bf16'] else None
    dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16 if amp_dtype == 'bf16' else torch.float32

    for x_batch, y_batch, cluster_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        cluster_batch = cluster_batch.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler:
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = model(x_batch, cluster_batch)
                loss = criterion(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x_batch, cluster_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, device, amp_dtype='fp16'):
    """Validate model"""
    model.eval()
    all_probs = []
    all_labels = []

    dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16 if amp_dtype == 'bf16' else torch.float32

    with torch.no_grad():
        for x_batch, y_batch, cluster_batch in dataloader:
            x_batch = x_batch.to(device)
            cluster_batch = cluster_batch.to(device)

            with torch.amp.autocast('cuda', dtype=dtype):
                logits = model(x_batch, cluster_batch)
                probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    auc = roc_auc_score(all_labels, all_probs)
    preds = (all_probs >= 0.5).astype(int)
    f1 = f1_score(all_labels, preds, zero_division=0)

    return auc, f1


def objective(trial, X, y, meta, device, epochs_per_trial, pos_weight, amp_dtype):
    """Optuna objective function"""

    # Clear GPU cache at start of each trial
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Suggest hyperparameters (removed 512 from hidden_size to avoid GPU crash)
    hidden_size = trial.suggest_categorical('hidden_size', [128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    n_clusters = trial.suggest_categorical('n_clusters', [3, 5, 7, 10])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    cluster_embed_dim = trial.suggest_categorical('cluster_embed_dim', [16, 32, 64])

    # Cluster data
    cluster_ids, _, _ = cluster_windows(meta, n_clusters=n_clusters, random_state=42)

    # Create dataset
    full_dataset = LightCurveDataset(X, y, cluster_ids)

    # Split into train and validation
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    model = ClusterBiLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        n_clusters=n_clusters,
        cluster_embed_dim=cluster_embed_dim
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_per_trial)

    # Training loop
    best_auc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(epochs_per_trial):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, amp_dtype)
        val_auc, val_f1 = validate(model, val_loader, device, amp_dtype)
        scheduler.step()

        # Update best AUC
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1

        # Report intermediate value for pruning
        trial.report(val_auc, epoch)

        # Early stopping
        if patience_counter >= patience:
            break

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Clear GPU cache before returning
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_auc


def main():
    parser = argparse.ArgumentParser(description='Optuna Hyperparameter Optimization')
    parser.add_argument('--windows_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\data\windows_train',
                       help='Path to windows directory')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of Optuna trials')
    parser.add_argument('--epochs_per_trial', type=int, default=50,
                       help='Max epochs per trial (with early stopping)')
    parser.add_argument('--output_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\optuna_results',
                       help='Output directory for results')
    parser.add_argument('--study_name', type=str, default='bilstm_cluster_optimization',
                       help='Name of the Optuna study')
    parser.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'],
                       help='Mixed precision dtype')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Trials: {args.n_trials}")
    print(f"Epochs per trial: {args.epochs_per_trial}")
    print(f"Mixed precision: {args.amp_dtype}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.windows_dir}...")
    X = np.load(os.path.join(args.windows_dir, 'X.npy'))
    y = np.load(os.path.join(args.windows_dir, 'y.npy'))
    meta = pd.read_csv(os.path.join(args.windows_dir, 'meta.csv'))

    print(f"Loaded {len(X)} windows")
    print(f"Positive samples: {y.sum()} ({100*y.mean():.1f}%)")

    # Calculate pos_weight for class imbalance
    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"Positive weight: {pos_weight:.3f}\n")

    # Create Optuna study
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',  # Maximize AUC
        sampler=sampler,
        pruner=pruner
    )

    # Run optimization
    print(f"Starting optimization with {args.n_trials} trials...\n")
    start_time = time.time()

    study.optimize(
        lambda trial: objective(
            trial, X, y, meta, device, args.epochs_per_trial, pos_weight, args.amp_dtype
        ),
        n_trials=args.n_trials,
        show_progress_bar=True,
        catch=(RuntimeError, torch.cuda.OutOfMemoryError)  # Continue even if CUDA errors occur
    )

    elapsed_time = time.time() - start_time

    # Results
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best AUC: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save study
    study_file = os.path.join(args.output_dir, f'optuna_study_{timestamp}.pkl')
    import joblib
    joblib.dump(study, study_file)
    print(f"Study saved to {study_file}")

    # Save best parameters
    best_params_file = os.path.join(args.output_dir, f'best_params_{timestamp}.json')
    results = {
        'timestamp': timestamp,
        'best_trial': study.best_trial.number,
        'best_auc': study.best_value,
        'best_params': study.best_params,
        'n_trials': args.n_trials,
        'elapsed_time_minutes': elapsed_time / 60
    }

    with open(best_params_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Best parameters saved to {best_params_file}")

    # Save trials dataframe
    trials_df = study.trials_dataframe()
    trials_file = os.path.join(args.output_dir, f'trials_{timestamp}.csv')
    trials_df.to_csv(trials_file, index=False)
    print(f"Trials data saved to {trials_file}")

    # Create comparison with baseline
    baseline_config = {
        'hidden_size': 256,
        'num_layers': 3,
        'dropout': 0.4,
        'lr': 1e-4,
        'batch_size': 64,
        'n_clusters': 5,
        'weight_decay': 1e-5,
        'cluster_embed_dim': 32
    }

    print(f"\n{'='*60}")
    print(f"BASELINE VS OPTIMIZED COMPARISON")
    print(f"{'='*60}")
    print(f"{'Parameter':<20} {'Baseline':<15} {'Optimized':<15} {'Change':<10}")
    print(f"{'-'*60}")
    for key in baseline_config.keys():
        baseline_val = baseline_config[key]
        optimized_val = study.best_params.get(key, baseline_val)
        change = "✓" if baseline_val != optimized_val else ""
        print(f"{key:<20} {str(baseline_val):<15} {str(optimized_val):<15} {change:<10}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
