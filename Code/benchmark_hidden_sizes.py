"""
Benchmark hidden_size training times for BiLSTM model.

Tests hidden_size = [128, 256, 512] to determine optimal search space.
Runs 3 epochs each to estimate time per epoch.

Usage:
    python benchmark_hidden_sizes.py
"""

import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class LightCurveDataset(Dataset):
    """Dataset for benchmark."""

    def __init__(self, flux_data, labels, clusters):
        self.flux_data = torch.from_numpy(flux_data).float().unsqueeze(-1)
        self.labels = torch.from_numpy(labels).float()
        self.clusters = torch.from_numpy(clusters).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.flux_data[idx], self.labels[idx], self.clusters[idx]


class ClusterBiLSTM(nn.Module):
    """BiLSTM model for benchmark."""

    def __init__(self, hidden_size, num_layers=4, n_clusters=5, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.cluster_embed = nn.Embedding(n_clusters, 64)

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        combined_size = hidden_size * 2 + 64
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, flux, cluster_ids):
        cluster_emb = self.cluster_embed(cluster_ids)
        _, (hidden, _) = self.lstm(flux)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        combined = torch.cat([hidden_cat, cluster_emb], dim=1)
        return self.classifier(combined).squeeze(-1)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_hidden_size(hidden_size, dataloader, device, n_epochs=3):
    """Benchmark a single hidden_size configuration."""
    print(f"\n{'='*60}", flush=True)
    print(f"Benchmarking hidden_size={hidden_size}", flush=True)
    print(f"{'='*60}", flush=True)

    model = ClusterBiLSTM(hidden_size=hidden_size).to(device)
    params = count_parameters(model)
    print(f"Model parameters: {params:,}", flush=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=0.0001)
    scaler = torch.amp.GradScaler('cuda')

    epoch_times = []

    for epoch in range(n_epochs):
        model.train()
        epoch_start = time.time()

        for flux, labels, clusters in dataloader:
            flux = flux.to(device)
            labels = labels.to(device)
            clusters = clusters.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(flux, clusters)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print(f"  Epoch {epoch+1}: {epoch_time:.1f} sec", flush=True)

    avg_time = np.mean(epoch_times)
    print(f"Average epoch time: {avg_time:.1f} sec ({avg_time/60:.2f} min)", flush=True)

    # Clear GPU memory
    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return {
        'hidden_size': hidden_size,
        'parameters': params,
        'avg_epoch_time_sec': avg_time,
        'avg_epoch_time_min': avg_time / 60,
        'est_20_epochs_min': (avg_time * 20) / 60
    }


def main():
    """Run benchmark."""
    print("\n" + "="*60, flush=True)
    print("HIDDEN SIZE BENCHMARK", flush=True)
    print("="*60, flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Load data
    data_dir = r"D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train"
    print(f"Loading data from {data_dir}...", flush=True)

    flux_data = np.load(f"{data_dir}/X.npy")
    labels = np.load(f"{data_dir}/y.npy")
    meta = pd.read_csv(f"{data_dir}/meta.csv")

    print(f"Loaded {len(flux_data)} windows", flush=True)

    # Cluster
    stat_features = ['mean', 'std', 'var', 'skew', 'range', 'median', 'mad', 'peak_to_peak']
    features = meta[stat_features].values
    features = np.nan_to_num(features, nan=0.0)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features_scaled)

    # Create dataloader with batch_size=112 (same as best trial)
    dataset = LightCurveDataset(flux_data, labels, cluster_ids)
    dataloader = DataLoader(dataset, batch_size=112, shuffle=True, num_workers=0)

    print(f"Batches per epoch: {len(dataloader)}", flush=True)

    # Benchmark each hidden_size
    hidden_sizes = [128, 192, 256]
    results = []

    for hidden_size in hidden_sizes:
        result = benchmark_hidden_size(hidden_size, dataloader, device, n_epochs=3)
        results.append(result)

    # Print summary
    print("\n" + "="*60, flush=True)
    print("BENCHMARK RESULTS SUMMARY", flush=True)
    print("="*60, flush=True)
    print(f"{'Hidden Size':<12} {'Parameters':<15} {'Epoch Time':<12} {'20 Epochs':<12}", flush=True)
    print("-"*60, flush=True)

    for r in results:
        print(f"{r['hidden_size']:<12} {r['parameters']:>12,}   {r['avg_epoch_time_min']:.2f} min      {r['est_20_epochs_min']:.1f} min", flush=True)

    print("="*60, flush=True)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(r"D:\CS_4280_Project\Code\hidden_size_benchmark.csv", index=False)
    print("\nResults saved to hidden_size_benchmark.csv", flush=True)


if __name__ == '__main__':
    main()
