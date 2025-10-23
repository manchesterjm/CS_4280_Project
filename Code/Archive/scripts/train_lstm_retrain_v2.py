#!/usr/bin/env python
"""
train_lstm_retrain_v2.py

Purpose
-------
Train (or continue training) a simple LSTM classifier on pre-windowed light curves
stored as X.npy (shape [N, seq_len]) and y.npy (shape [N]) in a --windows_dir.

Key fixes vs prior version
--------------------------
1) Robust AMP handling: evaluation casts tensors to float32 on CPU before .numpy(),
   avoiding bfloat16 -> NumPy errors.
2) Read-only NumPy arrays: ensures X/y are writable before torch.from_numpy().
3) Windows path escape warnings removed; no stray shell commands in code.

Usage (PowerShell)
------------------
python .\train_lstm_retrain_v2.py `
  --windows_dir "C:\\CS_4280_Project\\Code\\data\\windows_train" `
  --epochs 40 --batch_size 256 --lr 3e-4 --hidden 128 --layers 2 --dropout 0.2 `
  --save_dir "C:\\CS_4280_Project\\Code\\runs\\lstm_v2" --amp_dtype fp16

You can resume training with --resume <ckpt_path>.
"""
from __future__ import annotations

import argparse
import os
import time
import math
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # minimal fallback

# Optional metric: ROC-AUC if scikit-learn is available
try:
    from sklearn.metrics import roc_auc_score
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# ---------------------------
# Repro / Speed Preferences
# ---------------------------
torch.backends.cudnn.benchmark = True  # faster on fixed input sizes
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


@dataclass
class TrainConfig:
    windows_dir: str
    epochs: int = 40
    batch_size: int = 256
    lr: float = 3e-4
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.2
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 42
    save_dir: str = "runs/lstm_v2"
    amp_dtype: str = "fp16"  # choices: "none", "fp16", "bf16"
    resume: Optional[str] = None
    patience: int = 10


class WindowsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2, f"X must be 2D [N, seq_len], got {X.shape}"
        assert y.ndim == 1 and len(y) == len(X), "y must be 1D, same length as X"
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)

        # ensure writable for torch.from_numpy
        if not self.X.flags.writeable:
            self.X = self.X.copy()
        if not self.y.flags.writeable:
            self.y = self.y.copy()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        # Convert to (seq_len, features=1)
        x = torch.from_numpy(x.copy()).float().unsqueeze(-1)  # [T] -> [T,1]
        y = torch.tensor(y, dtype=torch.float32)
        return x, y


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden,
                            num_layers=layers,
                            dropout=(dropout if layers > 1 else 0.0),
                            bidirectional=False,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: [B, T, 1]
        out, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state from top layer: h_n shape [num_layers, B, H]
        last_h = h_n[-1]  # [B, H]
        last_h = self.dropout(last_h)
        logits = self.fc(last_h).squeeze(-1)  # [B]
        return logits


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_windows(windows_dir: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    x_path = os.path.join(windows_dir, "X.npy")
    y_path = os.path.join(windows_dir, "y.npy")
    meta_path = os.path.join(windows_dir, "meta.csv")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(f"Expected X.npy and y.npy under {windows_dir}")

    X = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)
    meta = None
    if os.path.exists(meta_path):
        try:
            meta = np.genfromtxt(meta_path, delimiter=",", names=True, dtype=None, encoding=None)
        except Exception:
            meta = None
    return X, y, meta


def split_dataset(X: np.ndarray, y: np.ndarray, val_frac: float, test_frac: float, seed: int):
    N = len(X)
    n_test = int(round(N * test_frac))
    n_val = int(round(N * val_frac))
    n_train = N - n_val - n_test
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Bad splits (N={N}): train {n_train}, val {n_val}, test {n_test}")

    ds = WindowsDataset(X, y)
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)
    return train_ds, val_ds, test_ds


def make_loaders(train_ds, val_ds, test_ds, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    common = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **common)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader


def compute_class_pos_weight(y: np.ndarray) -> float:
    # pos_weight = (#neg / #pos) for BCEWithLogits
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0:
        return 1.0
    return max(1.0, neg / pos)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp_dtype: str) -> Dict[str, float]:
    model.eval()
    probs_list = []
    targets_list = []

    use_amp = (amp_dtype != "none")
    autocast_dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                logits = model(xb)
                p = torch.sigmoid(logits)
        else:
            logits = model(xb)
            p = torch.sigmoid(logits)

        # Always detach->float32->cpu before numpy to avoid bf16 issues
        probs_list.append(p.detach().to(torch.float32).cpu())
        targets_list.append(yb.detach().to(torch.float32).cpu())

    probs = torch.cat(probs_list).numpy()
    targets = torch.cat(targets_list).numpy()

    # Threshold metrics
    thr = 0.5
    pred = (probs >= thr).astype(np.float32)
    tp = float(((pred == 1) & (targets == 1)).sum())
    tn = float(((pred == 0) & (targets == 0)).sum())
    fp = float(((pred == 1) & (targets == 0)).sum())
    fn = float(((pred == 0) & (targets == 1)).sum())
    acc = (tp + tn) / max(1.0, tp + tn + fp + fn)
    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = (2 * prec * rec / max(1e-12, (prec + rec))) if (prec + rec) > 0 else 0.0

    metrics = {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    if HAVE_SKLEARN:
        try:
            auc = float(roc_auc_score(targets, probs))
            metrics["auc"] = auc
        except Exception:
            pass

    return metrics


def save_checkpoint(path: str, state: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, scaler: Optional[torch.cuda.amp.GradScaler]):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ("scaler" in ckpt):
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt.get("epoch", 0)
    best_val = ckpt.get("best_val", -math.inf)
    return epoch, best_val


# ---------------------------
# Train Loop
# ---------------------------

def train(cfg: TrainConfig):
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_flag = cfg.amp_dtype != "none"

    print(f"[runtime] device={device}  amp={amp_flag}  dtype={cfg.amp_dtype}  tf32=True  cudnn.benchmark=True")

    X, y, meta = load_windows(cfg.windows_dir)
    print(f"[windows] total={len(X)} seq_len={X.shape[1]}  pos={(y==1).sum()} neg={(y==0).sum()}")

    train_ds, val_ds, test_ds = split_dataset(X, y, cfg.val_frac, cfg.test_frac, cfg.seed)
    tr_loader, va_loader, te_loader = make_loaders(train_ds, val_ds, test_ds, cfg.batch_size)

    input_size = 1  # each timestep has 1 feature (flux)
    model = LSTMClassifier(input_size=input_size, hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout)
    model.to(device)

    pos_weight_val = compute_class_pos_weight(y)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=amp_flag)
    autocast_dtype = torch.float16 if cfg.amp_dtype == "fp16" else torch.bfloat16

    start_epoch = 0
    best_val = -math.inf

    os.makedirs(cfg.save_dir, exist_ok=True)
    last_path = os.path.join(cfg.save_dir, "last.pt")
    best_path = os.path.join(cfg.save_dir, "best.pt")
    config_path = os.path.join(cfg.save_dir, "config.json")

    # Save config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    if cfg.resume is not None and os.path.exists(cfg.resume):
        print(f"[resume] loading checkpoint: {cfg.resume}")
        start_epoch, best_val = load_checkpoint(cfg.resume, model, optimizer, scaler)
        print(f"[resume] start_epoch={start_epoch} best_val={best_val:.4f}")

    # Initial evaluation
    va_metrics = evaluate(model, va_loader, device, cfg.amp_dtype)
    key_metric = va_metrics.get("auc", va_metrics.get("f1", va_metrics.get("acc", 0.0)))
    best_val = max(best_val, key_metric)
    print(f"[val0] {va_metrics}")

    epochs_no_improve = 0
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        t0 = time.time()

        running_loss = 0.0
        n_batches = 0

        for xb, yb in tqdm(tr_loader, desc=f"epoch {epoch+1}/{cfg.epochs}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if amp_flag:
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            running_loss += float(loss.detach().item())
            n_batches += 1

        # Validation
        va_metrics = evaluate(model, va_loader, device, cfg.amp_dtype)
        key_metric = va_metrics.get("auc", va_metrics.get("f1", va_metrics.get("acc", 0.0)))
        scheduler.step(key_metric)

        dt = time.time() - t0
        avg_loss = running_loss / max(1, n_batches)
        print(f"[epoch {epoch+1}] loss={avg_loss:.4f}  val={va_metrics}  dt={dt:.1f}s  lr={optimizer.param_groups[0]['lr']:.3e}")

        # Save "last"
        save_checkpoint(last_path, {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if amp_flag else None,
            "best_val": best_val,
        })

        # Early stopping / Best
        if key_metric > best_val + 1e-6:
            best_val = key_metric
            save_checkpoint(best_path, {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if amp_flag else None,
                "best_val": best_val,
            })
            epochs_no_improve = 0
            print(f"[best] improved to {best_val:.4f}; saved -> {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"[early-stop] no improvement for {cfg.patience} epochs; stopping.")
                break

    # Load best and evaluate on test
    if os.path.exists(best_path):
        print(f"[final] loading best checkpoint: {best_path}")
        _ = load_checkpoint(best_path, model, optimizer, scaler)

    te_metrics = evaluate(model, te_loader, device, cfg.amp_dtype)
    print(f"[test] {te_metrics}")


# ---------------------------
# Argparse
# ---------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train LSTM on windowed light curves (v2)")
    p.add_argument('--windows_dir', type=str, required=True, help='Folder with X.npy and y.npy')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_dir', type=str, default='runs/lstm_v2')
    p.add_argument('--amp_dtype', type=str, default='fp16', choices=['none', 'fp16', 'bf16'])
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--patience', type=int, default=10, help='epochs without val improvement before early-stop')

    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)
