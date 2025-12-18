"""
Batch Size Benchmark Script
Tests batch sizes 8, 16, 32, 64, 128 for 30 seconds each
Estimates time to complete one epoch

November 29, 2025
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# CPU thread limits
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# Model definition (same as train_bilstm_cluster.py)
class ClusterBiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=4, dropout=0.311,
                 n_clusters=5, cluster_embed_dim=32):
        super().__init__()
        self.cluster_embedding = nn.Embedding(n_clusters, cluster_embed_dim)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.bn1 = nn.BatchNorm1d(hidden_size * 2 + cluster_embed_dim)
        self.fc1 = nn.Linear(hidden_size * 2 + cluster_embed_dim, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, cluster_ids):
        cluster_emb = self.cluster_embedding(cluster_ids)
        lstm_out, (h_n, _) = self.lstm(x)
        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h_cat = torch.cat([h_fwd, h_bwd, cluster_emb], dim=1)
        out = self.bn1(h_cat)
        out = self.dropout(self.relu(self.bn2(self.fc1(out))))
        out = self.dropout(self.relu(self.bn3(self.fc2(out))))
        out = self.fc3(out)
        return out.squeeze(-1)


def benchmark_batch_size(batch_size, X, y, cluster_ids, model, criterion, optimizer,
                         device, duration_sec=30):
    """Run training for duration_sec and estimate epoch time"""

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(cluster_ids, dtype=torch.long)
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    total_batches = len(loader)

    model.train()
    scaler = torch.amp.GradScaler('cuda')

    batches_completed = 0
    start_time = time.time()

    print(f"\n  Batch size {batch_size}: {total_batches} batches/epoch")
    print(f"  Running for {duration_sec} seconds...", end=" ", flush=True)

    while True:
        for x_batch, y_batch, c_batch in loader:
            x_batch = x_batch.unsqueeze(-1).to(device)
            y_batch = y_batch.to(device)
            c_batch = c_batch.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x_batch, c_batch)
                loss = criterion(logits, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batches_completed += 1
            elapsed = time.time() - start_time

            if elapsed >= duration_sec:
                break

        if time.time() - start_time >= duration_sec:
            break

    elapsed = time.time() - start_time
    time_per_batch = elapsed / batches_completed
    estimated_epoch_time = time_per_batch * total_batches

    print(f"Done!")
    print(f"  Batches completed: {batches_completed}")
    print(f"  Time per batch: {time_per_batch:.3f} sec")
    print(f"  Estimated epoch time: {estimated_epoch_time:.1f} sec ({estimated_epoch_time/60:.2f} min)")

    return {
        'batch_size': batch_size,
        'total_batches': total_batches,
        'batches_completed': batches_completed,
        'time_per_batch': time_per_batch,
        'estimated_epoch_sec': estimated_epoch_time,
        'estimated_epoch_min': estimated_epoch_time / 60
    }


def main():
    print("=" * 60)
    print("BATCH SIZE BENCHMARK")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    data_dir = r"D:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train"

    # Check if data exists, if not try backup location
    if not os.path.exists(data_dir):
        data_dir = r"E:\CS_4280_Project_Backup\Code\data\windows_sector1_full\train"

    if not os.path.exists(data_dir):
        print(f"ERROR: Data not found at {data_dir}")
        sys.exit(1)

    print(f"Loading data from: {data_dir}")

    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))

    print(f"Dataset: {len(X)} samples")

    # Create dummy cluster IDs (0-4)
    cluster_ids = np.random.randint(0, 5, size=len(X))

    # Calculate pos_weight
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    pos_weight = n_neg / n_pos
    print(f"Class balance: {n_pos} pos / {n_neg} neg (pos_weight={pos_weight:.2f})")

    # Batch sizes to test (112 to 144 in increments of 8)
    batch_sizes = [112, 120, 128, 136, 144]
    results = []

    print("\n" + "=" * 60)
    print("BENCHMARKING (20 sec per batch size)")
    print("=" * 60)

    for bs in batch_sizes:
        # Create fresh model for each test
        model = ClusterBiLSTM(
            input_size=1,
            hidden_size=256,
            num_layers=4,
            dropout=0.311,
            n_clusters=5,
            cluster_embed_dim=32
        ).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

        result = benchmark_batch_size(
            batch_size=bs,
            X=X, y=y, cluster_ids=cluster_ids,
            model=model, criterion=criterion, optimizer=optimizer,
            device=device, duration_sec=20
        )
        results.append(result)

        # Clear GPU memory
        del model, criterion, optimizer
        torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Batch Size':<12} {'Batches/Epoch':<15} {'Time/Batch':<12} {'Est. Epoch Time':<20}")
    print("-" * 60)

    for r in results:
        print(f"{r['batch_size']:<12} {r['total_batches']:<15} {r['time_per_batch']:.3f} sec      {r['estimated_epoch_min']:.2f} min")

    print("\n" + "=" * 60)

    # Find optimal
    fastest = min(results, key=lambda x: x['estimated_epoch_sec'])
    print(f"FASTEST: Batch size {fastest['batch_size']} ({fastest['estimated_epoch_min']:.2f} min/epoch)")
    print("=" * 60)


if __name__ == '__main__':
    main()
