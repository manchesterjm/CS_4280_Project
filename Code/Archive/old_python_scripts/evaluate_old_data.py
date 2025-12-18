"""Evaluate trained model on older datasets that lack ground truth features."""
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

        self.cluster_embed = nn.Embedding(n_clusters, cluster_embed_dim)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)

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
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    # Load checkpoint
    print(f"[loading model] {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Load data
    print(f"[loading data] {args.data_dir}")
    X = np.load(f"{args.data_dir}/X.npy")
    meta = pd.read_csv(f"{args.data_dir}/meta.csv")

    # Check for y.npy (labels)
    y_path = f"{args.data_dir}/y.npy"
    has_labels = True
    try:
        y = np.load(y_path)
        print(f"[data] X.shape={X.shape} y.shape={y.shape}")
        print(f"[data] pos={y.sum()} neg={len(y) - y.sum()}")
    except FileNotFoundError:
        has_labels = False
        y = None
        print(f"[data] X.shape={X.shape} (no labels)")

    # Ensure metadata matches X length
    if len(meta) != len(X):
        print(f"[warning] meta.csv has {len(meta)} rows but X.npy has {len(X)} samples")
        meta = meta.iloc[:len(X)]

    # Create features from old data format
    # Model was trained with: log_period, log_depth, duration, teff
    # Old data has: period, duration, depth, bls_power
    print("[clustering] Converting old data format to ground truth features")

    # Compute log_period from period
    if 'period' in meta.columns:
        meta['log_period'] = np.log10(meta['period'].clip(lower=0.1))
    else:
        meta['log_period'] = 1.0  # default

    # Compute log_depth from depth
    # Training data uses log10(depth_ppm) where depth_ppm is ~100-10000
    # Old data stores depth as fractional (0.001 = 1000 ppm) or already in ppm
    if 'depth' in meta.columns:
        depth_vals = meta['depth'].values.copy()
        if depth_vals.max() < 1:  # fractional depth, convert to ppm
            depth_ppm = depth_vals * 1e6
        elif depth_vals.max() > 1e6:  # already very large, might be in weird units
            depth_ppm = depth_vals
        else:  # assume already in ppm
            depth_ppm = depth_vals
        # Clip to reasonable range and take log
        depth_ppm = np.clip(depth_ppm, 10, 100000)  # 10 ppm to 10%
        meta['log_depth'] = np.log10(depth_ppm)
    else:
        meta['log_depth'] = 3.0  # default (~1000 ppm)

    # Duration - training data is in hours, old data might be in days
    if 'duration' in meta.columns:
        dur_vals = meta['duration'].values
        if dur_vals.max() < 1:  # likely in days, convert to hours
            meta['duration'] = dur_vals * 24
        # else assume already in hours
    else:
        meta['duration'] = 4.0  # default 4 hours

    # teff not available in old data - use median from training
    meta['teff'] = 5800  # Solar-like default

    feature_cols = ['log_period', 'log_depth', 'duration', 'teff']
    features = meta[feature_cols].values

    print(f"[clustering] Feature ranges:")
    for i, col in enumerate(feature_cols):
        print(f"  {col}: [{features[:, i].min():.3f}, {features[:, i].max():.3f}]")

    # Handle missing values
    features = np.nan_to_num(features, nan=0.0)

    # Clip outliers
    for i in range(features.shape[1]):
        p1, p99 = np.percentile(features[:, i], [1, 99])
        features[:, i] = np.clip(features[:, i], p1, p99)

    # Use scaler from checkpoint
    scaler_params = checkpoint['scaler_params']
    kmeans_centers = checkpoint['kmeans_centers']

    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params['mean'])
    scaler.scale_ = np.array(scaler_params['scale'])

    features_scaled = scaler.transform(features)

    kmeans = KMeans(n_clusters=len(kmeans_centers), random_state=42)
    kmeans.cluster_centers_ = np.array(kmeans_centers)
    kmeans._n_threads = 1

    cluster_ids = kmeans.predict(features_scaled)
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

    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)

    if has_labels:
        auc = roc_auc_score(y, probs)
        f1 = f1_score(y, preds)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        accuracy = accuracy_score(y, preds)

        print(f"  AUC:       {auc:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")

        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        print(f"\n  Confusion Matrix:")
        print(f"    TP={tp:4d}  FP={fp:4d}")
        print(f"    FN={fn:4d}  TN={tn:4d}")
    else:
        # No labels - show prediction summary
        print(f"  Total samples: {len(probs)}")
        print(f"  Predicted planets: {preds.sum()} ({100*preds.sum()/len(preds):.1f}%)")
        print(f"  Predicted non-planets: {len(preds) - preds.sum()}")
        print(f"\n  Probability distribution:")
        print(f"    Mean: {probs.mean():.4f}")
        print(f"    Median: {np.median(probs):.4f}")
        print(f"    Min: {probs.min():.4f}")
        print(f"    Max: {probs.max():.4f}")

    print("="*50)

    # Show top candidates
    top_k = min(20, len(probs))
    top_idx = np.argsort(probs)[::-1][:top_k]

    print(f"\n[top {top_k} candidates]")
    print(f"{'Rank':>4} {'TIC ID':>12} {'Prob':>8} {'Period':>10} {'Depth':>10}")
    print("-" * 50)

    for rank, idx in enumerate(top_idx, 1):
        tic = meta.iloc[idx].get('tic_id', 'N/A')
        period = meta.iloc[idx].get('period', 0)
        depth = meta.iloc[idx].get('depth', 0)
        print(f"{rank:4d} {tic:>12} {probs[idx]:8.4f} {period:10.4f} {depth:10.4f}")

    # Save predictions
    if args.output_file:
        results = meta.copy()
        results['probability'] = probs
        results['predicted'] = preds
        if has_labels:
            results['actual'] = y
        results.to_csv(args.output_file, index=False)
        print(f"\n[saved] {args.output_file}")


if __name__ == '__main__':
    main()
