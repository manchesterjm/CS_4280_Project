"""
Simple BiLSTM training without clustering (for Sector 1 dataset)
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import time


class SimpleBiLSTM(nn.Module):
    """Simple BiLSTM without clustering"""
    def __init__(self, input_size=1, hidden_size=256, num_layers=4, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # Fully connected layers
        lstm_out_size = hidden_size * 2  # bidirectional
        self.fc1 = nn.Linear(lstm_out_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use final hidden states from both directions
        h_fwd = h_n[-2, :, :]  # forward
        h_bwd = h_n[-1, :, :]  # backward
        h = torch.cat([h_fwd, h_bwd], dim=1)

        # Fully connected layers
        out = torch.relu(self.bn1(self.fc1(h)))
        out = self.dropout1(out)

        out = torch.relu(self.bn2(self.fc2(out)))
        out = self.dropout2(out)

        out = self.fc3(out)
        return out.squeeze(-1)


class LightCurveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(-1)  # (N, seq_len, 1)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_metrics(y_true, y_pred, y_scores):
    """Compute classification metrics"""
    return {
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
    }


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype):
    """Evaluate model"""
    model.eval()
    y_true = []
    y_scores = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)

        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            logits = model(x_batch)
            probs = torch.sigmoid(logits)

        y_true.extend(y_batch.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= 0.5).astype(int)

    # Check for NaN
    if np.isnan(y_scores).any():
        print("[WARNING] NaN values in predictions!")
        y_scores = np.nan_to_num(y_scores, nan=0.5)

    metrics = compute_metrics(y_true, y_pred, y_scores)
    return metrics


def train_epoch(model, loader, criterion, optimizer, scaler, device, autocast_dtype, gradient_clip):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for x_batch, y_batch in tqdm(loader, desc="Training", leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

        # Check for NaN loss
        if torch.isnan(loss):
            print("[WARNING] NaN loss detected, skipping batch")
            continue

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--windows_dir', type=str, required=True)
    parser.add_argument('--val_split', type=float, default=0.15)

    # Model
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.311)

    # Training
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.000225)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--pos_weight', type=float, default=3.367)
    parser.add_argument('--gradient_clip', type=float, default=1.0)

    # System
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--amp_dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'])
    parser.add_argument('--seed', type=int, default=42)

    # Output
    parser.add_argument('--save_dir', type=str, required=True)

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # AMP setup
    amp_flag = args.amp_dtype != 'fp32'
    if args.amp_dtype == 'fp16':
        autocast_dtype = torch.float16
    elif args.amp_dtype == 'bf16':
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32

    print(f"[runtime] device={device} amp={amp_flag} dtype={args.amp_dtype}")
    print(f"[runtime] model=Simple-BiLSTM")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    print("\n[loading data]")
    X = np.load(os.path.join(args.windows_dir, 'X.npy'))
    y = np.load(os.path.join(args.windows_dir, 'y.npy'))

    print(f"[data] X.shape={X.shape} y.shape={y.shape}")
    print(f"[data] pos={int(y.sum())} neg={int(len(y) - y.sum())}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_split, random_state=args.seed, stratify=y
    )

    print(f"\n[split] train={len(y_train)} val={len(y_val)}")
    print(f"[train] pos={int(y_train.sum())} neg={int(len(y_train) - y_train.sum())}")
    print(f"[val] pos={int(y_val.sum())} neg={int(len(y_val) - y_val.sum())}")

    # Create datasets
    train_dataset = LightCurveDataset(X_train, y_train)
    val_dataset = LightCurveDataset(X_val, y_val)

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
    model = SimpleBiLSTM(
        input_size=1,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[model] params={n_params:,}")
    print(f"[model] hidden={args.hidden} layers={args.layers}")

    # Loss and optimizer
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )

    scaler = torch.amp.GradScaler('cuda')

    print(f"[loss] BCEWithLogitsLoss pos_weight={args.pos_weight}")

    # Training config
    config = vars(args)

    # Validation before training
    print("\n[validation before training]")
    val_metrics = evaluate(model, val_loader, device, autocast_dtype)
    print(f"[val0] {val_metrics}")

    # Training loop
    print(f"\n[training for {args.epochs} epochs]")
    patience = 15
    patience_counter = 0
    best_auc = 0
    best_f1 = 0

    print(f"[early stopping] patience={patience} epochs")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                               device, autocast_dtype, args.gradient_clip)

        # Validate
        val_metrics = evaluate(model, val_loader, device, autocast_dtype)

        # Step scheduler
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        dt = time.time() - t0

        print(f"[epoch {epoch:2d}/{args.epochs}] loss={avg_loss:.4f} "
              f"auc={val_metrics['auc']:.4f} f1={val_metrics['f1']:.4f} "
              f"acc={val_metrics['acc']:.4f} dt={dt:.1f}s lr={lr:.3e}")

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_f1 = val_metrics['f1']
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(args.save_dir, 'best.pt'))

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

    print(f"\n[training complete]")
    print(f"[best] auc={best_auc:.4f} f1={best_f1:.4f}")


if __name__ == '__main__':
    main()
