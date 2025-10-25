"""
Bidirectional LSTM for Exoplanet Detection
RNN-only architecture (no CNN or Transformers)

BiLSTM processes sequences in both directions to capture better context
Should perform better than simple LSTM

Usage:
    python train_bilstm.py --windows_dir "C:\CS_4280_Project\Code\data\windows_train" --epochs 60 --batch_size 128 --lr 5e-4 --hidden 256 --layers 3 --dropout 0.4 --save_dir "C:\CS_4280_Project\Code\runs\bilstm" --amp_dtype fp16 --pos_weight 3.367 --num_workers 0
"""

import argparse
import os
import json
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

warnings.filterwarnings('ignore')


class LightCurveDataset(Dataset):
    """Dataset for light curve windows"""
    
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        
        if len(self.X.shape) == 2:
            self.X = self.X.unsqueeze(-1)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM for exoplanet detection
    Processes sequence forward and backward for better context
    """
    
    def __init__(self, input_size=1, hidden_size=256, num_layers=3, dropout=0.4):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Key difference from simple LSTM
        )
        
        # Multi-layer classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        
        # BiLSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # hidden: (num_layers * 2, batch, hidden_size)
        # Concatenate forward and backward from last layer
        hidden_fwd = hidden[-2]  # Last layer, forward direction
        hidden_bwd = hidden[-1]  # Last layer, backward direction
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Multi-layer classification with batch norm
        out = self.dropout(hidden_cat)
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


def load_data(windows_dir):
    """Load windowed data"""
    X = np.load(os.path.join(windows_dir, "X.npy"))
    y = np.load(os.path.join(windows_dir, "y.npy"))
    
    print(f"[data] X.shape={X.shape} y.shape={y.shape}")
    print(f"[data] pos={int(y.sum())} neg={int(len(y) - y.sum())}")
    
    # Check for bad values
    if np.isnan(X).any():
        print("[warning] NaN values detected in X, replacing with 0")
        X = np.nan_to_num(X, nan=0.0)
    if np.isinf(X).any():
        print("[warning] Inf values detected in X, replacing with 0")
        X = np.nan_to_num(X, posinf=0.0, neginf=0.0)
    
    return X, y


def split_data(X, y, val_split=0.15, seed=42):
    """Split data into train and validation sets"""
    np.random.seed(seed)
    n = len(y)
    indices = np.random.permutation(n)
    
    val_size = int(n * val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    return X_train, y_train, X_val, y_val


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
    """Evaluate model on validation set"""
    model.eval()
    
    all_labels = []
    all_scores = []
    
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            logits = model(x_batch)
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
    
    for x_batch, y_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', dtype=autocast_dtype):
            logits = model(x_batch)
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
    parser = argparse.ArgumentParser(description='Train BiLSTM for exoplanet detection')
    
    # Data
    parser.add_argument('--windows_dir', type=str, required=True,
                       help='Directory containing X.npy, y.npy, meta.csv')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split fraction')
    
    # Model
    parser.add_argument('--hidden', type=int, default=256,
                       help='Hidden size (larger than simple LSTM)')
    parser.add_argument('--layers', type=int, default=3,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.4,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=60,
                       help='Number of epochs (more than before)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate (higher than before)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--pos_weight', type=float, default=3.367,
                       help='Positive class weight for imbalance')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                       help='Gradient clipping max norm')
    
    # System
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader workers (use 0 for Windows)')
    parser.add_argument('--amp_dtype', type=str, default='fp16',
                       choices=['fp16', 'bf16', 'fp32'],
                       help='Mixed precision dtype')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save model and logs')
    
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
    
    # Enable optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    print(f"[runtime] device={device} amp={amp_flag} dtype={args.amp_dtype}")
    print(f"[runtime] model=BiLSTM (Bidirectional LSTM)")
    print(f"[runtime] num_workers={args.num_workers}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    print("\n[loading data]")
    X, y = load_data(args.windows_dir)
    X_train, y_train, X_val, y_val = split_data(X, y, args.val_split, args.seed)
    
    print(f"[split] train={len(y_train)} val={len(y_val)}")
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
    
    # Get input size
    input_size = train_dataset.X.shape[-1]
    print(f"[model] input_size={input_size}")
    
    # Create model
    model = BiLSTM(
        input_size=input_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,}")
    print(f"[model] hidden={args.hidden} layers={args.layers} bidirectional=True")
    
    # Loss function with class weight
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"[loss] BCEWithLogitsLoss pos_weight={args.pos_weight:.3f}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    # AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=amp_flag)
    
    # Initial validation
    print("\n[validation before training]")
    val_metrics = evaluate(model, val_loader, device, autocast_dtype)
    print(f"[val0] {val_metrics}")
    
    # Training loop
    best_auc = 0.0
    best_f1 = 0.0
    patience = 15
    patience_counter = 0
    
    print(f"\n[training for {args.epochs} epochs]")
    print(f"[early stopping] patience={patience} epochs")
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, autocast_dtype, scaler, args.gradient_clip
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, autocast_dtype)
        
        # Step scheduler
        scheduler.step()
        
        dt = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        
        print(f"[epoch {epoch:2d}/{args.epochs}] loss={train_loss:.4f} "
              f"auc={val_metrics['auc']:.4f} f1={val_metrics['f1']:.4f} "
              f"acc={val_metrics['acc']:.4f} dt={dt:.1f}s lr={lr:.3e}")
        
        # Save best model (based on AUC)
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
            print(f"[best] auc={best_auc:.4f} f1={best_f1:.4f}; saved -> {args.save_dir}/best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[early stopping] no improvement for {patience} epochs")
                break
        
        # Save last model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_metrics': val_metrics,
            'config': config
        }, os.path.join(args.save_dir, 'last.pt'))
    
    print(f"\n[training complete]")
    print(f"[best] AUC={best_auc:.4f} F1={best_f1:.4f}")
    print(f"[saved to] {args.save_dir}")


if __name__ == '__main__':
    main()
