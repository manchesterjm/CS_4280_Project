"""
Inference script for ClusterBiLSTM model
Runs predictions on test windows

Usage:
    python inference_cluster_model.py --model_path "C:\CS_4280_Project\Code\runs\bilstm_cluster\best.pt" --windows_dir "C:\CS_4280_Project\Code\data\windows_test" --output_file "C:\CS_4280_Project\Code\reports\test_predictions.csv"
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ClusterBiLSTM(nn.Module):
    """Same architecture as training"""
    
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, 
                 dropout=0.4, n_clusters=5, cluster_embed_dim=32):
        super().__init__()
        
        self.n_clusters = n_clusters
        self.cluster_embed = nn.Embedding(n_clusters, cluster_embed_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
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


def assign_clusters(meta_df, checkpoint):
    """Assign cluster IDs to windows using saved KMeans"""
    
    # Get saved cluster info
    scaler_params = checkpoint.get('scaler_params', None)
    kmeans_centers = checkpoint.get('kmeans_centers', None)
    
    if scaler_params is None or kmeans_centers is None:
        print("[warning] No clustering info in checkpoint, assigning cluster 0 to all")
        return np.zeros(len(meta_df), dtype=int)
    
    # Extract features
    feature_cols = ['period', 'duration', 'depth', 'bls_power']
    
    # Check if meta has these columns (from BLS)
    if not all(col in meta_df.columns for col in feature_cols):
        print(f"[warning] Meta missing BLS features, assigning cluster 0 to all")
        return np.zeros(len(meta_df), dtype=int)
    
    features = meta_df[feature_cols].values
    
    # Recreate scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_params['mean'])
    scaler.scale_ = np.array(scaler_params['scale'])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Recreate KMeans and assign clusters
    kmeans = KMeans(n_clusters=len(kmeans_centers), random_state=42)
    kmeans.cluster_centers_ = np.array(kmeans_centers)
    cluster_ids = kmeans.predict(features_scaled)
    
    return cluster_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--windows_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='predictions.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[runtime] device={device}")
    
    # Load data
    print(f"\n[loading data from {args.windows_dir}]")
    X = np.load(os.path.join(args.windows_dir, 'X.npy'))
    meta = pd.read_csv(os.path.join(args.windows_dir, 'meta.csv'))
    
    print(f"[data] X.shape={X.shape}, windows={len(X)}")
    print(f"[data] unique TIC={meta['tic_id'].nunique()}")
    
    # Load checkpoint
    print(f"\n[loading model from {args.model_path}]")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    config = checkpoint['config']
    print(f"[config] hidden={config['hidden']}, layers={config['layers']}, clusters={config['n_clusters']}")
    
    # Assign clusters
    print(f"\n[assigning clusters]")
    cluster_ids = assign_clusters(meta, checkpoint)
    print(f"[clusters] assigned (unique: {len(np.unique(cluster_ids))})")
    
    # Create model
    model = ClusterBiLSTM(
        input_size=1,
        hidden_size=config['hidden'],
        num_layers=config['layers'],
        dropout=config['dropout'],
        n_clusters=config['n_clusters'],
        cluster_embed_dim=config.get('cluster_embed_dim', 32)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"[model] loaded successfully")
    
    # Prepare data
    X_tensor = torch.from_numpy(X).float().unsqueeze(-1)  # (N, 2048, 1)
    cluster_tensor = torch.from_numpy(cluster_ids).long()
    
    # Run inference
    print(f"\n[running inference]")
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(X_tensor), args.batch_size):
            batch_X = X_tensor[i:i+args.batch_size].to(device)
            batch_clusters = cluster_tensor[i:i+args.batch_size].to(device)
            
            logits = model(batch_X, batch_clusters)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
    
    probs = np.concatenate(all_probs)
    
    # Create results DataFrame
    results = meta.copy()
    results['probability'] = probs
    results['prediction'] = (probs > 0.5).astype(int)
    results['cluster_id'] = cluster_ids
    
    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    
    print(f"\n[results saved to {output_path}]")
    
    # Summary
    print(f"\n" + "="*70)
    print(" PREDICTION SUMMARY")
    print("="*70)
    
    for tic_id in sorted(results['tic_id'].unique()):
        tic_results = results[results['tic_id'] == tic_id]
        max_prob = tic_results['probability'].max()
        n_positive = (tic_results['probability'] > 0.5).sum()
        
        print(f"\nTIC {tic_id}:")
        print(f"  Windows: {len(tic_results)}")
        print(f"  Max probability: {max_prob:.4f}")
        print(f"  Positive predictions: {n_positive}/{len(tic_results)}")
        
        if max_prob > 0.5:
            print(f"  ⭐ PLANET CANDIDATE!")
    
    print("\n" + "="*70)
    print(f"Total windows: {len(results)}")
    print(f"Predicted positives: {(probs > 0.5).sum()}")
    print(f"Mean probability: {probs.mean():.4f}")
    print("="*70)


if __name__ == '__main__':
    main()
