r"""
Benchmark Current BiLSTM+Clustering Model

Evaluates the current best model and saves detailed metrics for comparison.

Usage:
    python benchmark_model.py --model_path "C:\CS_4280_Project\Code\runs\bilstm_cluster\best.pt" --output_dir "C:\CS_4280_Project\Code\benchmarks"
"""

import argparse
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    accuracy_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tqdm import tqdm


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


def load_checkpoint(model_path, device):
    """Load model checkpoint"""
    print(f"\n[loading] Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint['config']
    print(f"[loading] Model configuration:")
    for key, value in config.items():
        if key != 'cluster_info':
            print(f"  {key}: {value}")

    return checkpoint


def recreate_clusters(checkpoint, meta_df):
    """Recreate cluster assignments from checkpoint"""
    print("\n[clustering] Recreating cluster assignments")

    # Extract saved clustering parameters
    scaler_params = checkpoint['scaler_params']
    kmeans_centers = np.array(checkpoint['kmeans_centers'])

    # Recreate scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params['mean'])
    scaler.scale_ = np.array(scaler_params['scale'])
    if 'var' in scaler_params:
        scaler.var_ = np.array(scaler_params['var'])
    scaler.n_features_in_ = len(scaler.mean_)

    # Get features and scale
    feature_cols = ['period', 'duration', 'depth', 'bls_power']
    features = meta_df[feature_cols].values
    features_scaled = scaler.transform(features)

    # Compute distances to cluster centers and assign to nearest
    distances = np.linalg.norm(features_scaled[:, np.newaxis] - kmeans_centers, axis=2)
    cluster_ids = np.argmin(distances, axis=1)

    n_clusters = len(kmeans_centers)
    print(f"[clustering] Assigned {len(cluster_ids)} windows to {n_clusters} clusters")

    return cluster_ids


def evaluate_model(model, dataloader, device, amp_dtype='fp16'):
    """Evaluate model on dataset"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x_batch, y_batch, cluster_batch in tqdm(dataloader, desc="Evaluating"):
            x_batch = x_batch.to(device)
            cluster_batch = cluster_batch.to(device)

            # Mixed precision inference
            dtype = torch.float16 if amp_dtype == 'fp16' else torch.bfloat16 if amp_dtype == 'bf16' else torch.float32
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = model(x_batch, cluster_batch)
                probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= 0.5).astype(int)

    return all_labels, all_preds, all_probs


def calculate_metrics(labels, preds, probs):
    """Calculate comprehensive metrics"""
    metrics = {}

    # Classification metrics
    metrics['accuracy'] = accuracy_score(labels, preds)
    metrics['precision'] = precision_score(labels, preds, zero_division=0)
    metrics['recall'] = recall_score(labels, preds, zero_division=0)
    metrics['f1'] = f1_score(labels, preds, zero_division=0)
    metrics['auc'] = roc_auc_score(labels, probs)

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['tn'] = int(cm[0, 0])
    metrics['fp'] = int(cm[0, 1])
    metrics['fn'] = int(cm[1, 0])
    metrics['tp'] = int(cm[1, 1])

    # ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    metrics['roc_curve'] = {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Benchmark BiLSTM+Clustering Model')
    parser.add_argument('--model_path', type=str,
                       default=r'C:\CS_4280_Project\Code\runs\bilstm_cluster\best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--windows_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\data\windows_train',
                       help='Path to windows directory')
    parser.add_argument('--output_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\benchmarks',
                       help='Output directory for benchmark results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[setup] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print(f"\n[data] Loading data from {args.windows_dir}")
    X = np.load(os.path.join(args.windows_dir, 'X.npy'))
    y = np.load(os.path.join(args.windows_dir, 'y.npy'))
    meta = pd.read_csv(os.path.join(args.windows_dir, 'meta.csv'))

    print(f"[data] Loaded {len(X)} windows")
    print(f"[data] Positive samples: {y.sum()} ({100*y.mean():.1f}%)")

    # Load checkpoint
    checkpoint = load_checkpoint(args.model_path, device)
    config = checkpoint['config']

    # Recreate clusters
    cluster_ids = recreate_clusters(checkpoint, meta)

    # Create dataset (use all data for benchmarking)
    dataset = LightCurveDataset(X, y, cluster_ids)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load model
    print("\n[model] Loading model")
    model = ClusterBiLSTM(
        input_size=1,
        hidden_size=config['hidden'],
        num_layers=config['layers'],
        dropout=config['dropout'],
        n_clusters=config['n_clusters'],
        cluster_embed_dim=config.get('cluster_embed_dim', 32)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total parameters: {total_params:,}")
    print(f"[model] Trainable parameters: {trainable_params:,}")

    # Evaluate
    print("\n[evaluation] Running evaluation...")
    labels, preds, probs = evaluate_model(model, dataloader, device, config.get('amp_dtype', 'fp16'))

    # Calculate metrics
    print("\n[metrics] Calculating metrics...")
    metrics = calculate_metrics(labels, preds, probs)

    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']:4d}  |  FP: {metrics['fp']:4d}")
    print(f"  FN: {metrics['fn']:4d}  |  TP: {metrics['tp']:4d}")
    print("="*60)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f'baseline_benchmark_{timestamp}.json')

    results = {
        'timestamp': timestamp,
        'model_path': args.model_path,
        'dataset_size': int(len(X)),
        'positive_samples': int(y.sum()),
        'config': config,
        'metrics': metrics,
        'model_parameters': {
            'total': int(total_params),
            'trainable': int(trainable_params)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[output] Results saved to {output_file}")

    # Also save a simple CSV for easy comparison
    csv_file = os.path.join(args.output_dir, f'baseline_metrics_{timestamp}.csv')
    metrics_df = pd.DataFrame({
        'Model': ['Baseline'],
        'Timestamp': [timestamp],
        'AUC': [metrics['auc']],
        'F1': [metrics['f1']],
        'Accuracy': [metrics['accuracy']],
        'Precision': [metrics['precision']],
        'Recall': [metrics['recall']],
        'TP': [metrics['tp']],
        'FP': [metrics['fp']],
        'TN': [metrics['tn']],
        'FN': [metrics['fn']]
    })
    metrics_df.to_csv(csv_file, index=False)
    print(f"[output] CSV metrics saved to {csv_file}")


if __name__ == '__main__':
    main()
