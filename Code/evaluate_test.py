"""Evaluate trained model on held-out test set."""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

class ClusterBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=3,
                 n_clusters=5, cluster_embed_dim=32, dropout=0.4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_clusters = n_clusters

        # Cluster embedding
        self.cluster_embed = nn.Embedding(n_clusters, cluster_embed_dim)

        # BiLSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)

        # Classification head (matching training script exactly)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    # Load checkpoint
    print(f"[loading model] {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']

    # Load test data
    print(f"[loading test data] {args.test_dir}")
    X = np.load(f"{args.test_dir}/X.npy")
    y = np.load(f"{args.test_dir}/y.npy")
    meta = pd.read_csv(f"{args.test_dir}/meta.csv")

    print(f"[test data] X.shape={X.shape} y.shape={y.shape}")
    print(f"[test data] pos={y.sum()} neg={len(y) - y.sum()}")

    # Reconstruct clustering
    scaler_params = checkpoint['scaler_params']
    kmeans_centers = checkpoint['kmeans_centers']

    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params['mean'])
    scaler.scale_ = np.array(scaler_params['scale'])

    kmeans = KMeans(n_clusters=len(kmeans_centers), random_state=42)
    kmeans.cluster_centers_ = np.array(kmeans_centers)
    kmeans._n_threads = 1

    # Get cluster features - use ground truth features that match training
    # The model was trained with: log_period, log_depth, duration, teff
    feature_cols = ['log_period', 'log_depth', 'duration', 'teff']

    # Check which features are available
    available_cols = [c for c in feature_cols if c in meta.columns]

    if len(available_cols) != len(feature_cols):
        print(f"[warning] Expected {feature_cols}, found {available_cols}")
        # Fall back to what's available
        feature_cols = available_cols

    print(f"[clustering] Using features: {feature_cols}")

    if len(feature_cols) == 0:
        print("[error] No clustering features found in test metadata!")
        return

    # Ensure metadata matches X length
    if len(meta) != len(X):
        print(f"[warning] meta.csv has {len(meta)} rows but X.npy has {len(X)} samples")
        print(f"[warning] Truncating metadata to match X")
        meta = meta.iloc[:len(X)]

    features = meta[feature_cols].values

    # Handle missing values
    features = np.nan_to_num(features, nan=0.0)

    # Clip outliers (same as training)
    for i in range(features.shape[1]):
        p1, p99 = np.percentile(features[:, i], [1, 99])
        features[:, i] = np.clip(features[:, i], p1, p99)

    features_scaled = scaler.transform(features)
    cluster_ids = kmeans.predict(features_scaled)

    assert len(cluster_ids) == len(X), f"Cluster IDs ({len(cluster_ids)}) must match X ({len(X)})"

    print(f"[clustering] Cluster distribution: {np.bincount(cluster_ids, minlength=len(kmeans_centers))}")

    # Build model
    model = ClusterBiLSTM(
        input_size=1,
        hidden_size=config['hidden'],
        num_layers=config['layers'],
        n_clusters=config['n_clusters'],
        cluster_embed_dim=config.get('cluster_embed_dim', 32),
        dropout=config['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[model] loaded, params={sum(p.numel() for p in model.parameters()):,}")

    # Run inference
    print("[inference] Running...")
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    cluster_tensor = torch.tensor(cluster_ids, dtype=torch.long)

    batch_size = 128
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = X_tensor[i:i+batch_size].to(device)
            c_batch = cluster_tensor[i:i+batch_size].to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x_batch, c_batch)
                probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())

    probs = np.array(all_probs)
    preds = (probs >= 0.5).astype(int)

    # Compute metrics
    auc = roc_auc_score(y, probs)
    f1 = f1_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    accuracy = accuracy_score(y, preds)

    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"  AUC:       {auc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp:4d}  FP={fp:4d}")
    print(f"    FN={fn:4d}  TN={tn:4d}")
    print("="*50)

    # Save predictions if requested
    if args.output_file:
        results = meta.copy()
        results['probability'] = probs
        results['predicted'] = preds
        results['actual'] = y
        results.to_csv(args.output_file, index=False)
        print(f"\n[saved] {args.output_file}")


if __name__ == '__main__':
    main()
