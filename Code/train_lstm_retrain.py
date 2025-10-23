# -*- coding: utf-8 -*-
"""
Pure RNN (BiLSTM) training on labeled windows (X.npy, y.npy, meta.csv).

- Stratified TIC-level split (prevents window leakage)
- Class weighting in BCEWithLogitsLoss
- AMP + GPU, TF32 enabled
- Early stopping on validation F1
- Saves: models/exo_bilstm_retrain.pt, models/metrics_retrain.txt, models/splits_retrain.json

Usage:
  conda activate exo-lstm-gpu
  cd C:\CS_4280_Project\Code
  python .\train_lstm_retrain.py ^
    --windows_dir "C:\CS_4280_Project\Code\data\windows_train" ^
    --epochs 40 --batch_size 256 --lr 3e-4 --hidden 128 --layers 2 --dropout 0.2
"""

import os, json, math, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------- Runtime utils ---------------------

def get_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return "cuda"
    return "cpu"

# --------------------- Data ------------------------------

class WindowDataset(Dataset):
    def __init__(self, X_path, y_path, meta_df, indices):
        # memory-mapped to keep RAM in check
        self.X = np.load(X_path, mmap_mode="r")  # shape (N, T) or (N, T, F)
        self.y = np.load(y_path, mmap_mode="r")  # shape (N,)
        self.meta = meta_df.reset_index(drop=True)
        self.indices = np.array(indices, dtype=np.int64)

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, idx):
        j = int(self.indices[idx])
        x = self.X[j]
        y = int(self.y[j])
        # ensure (T, F) tensor, F=1 if 1D
        if x.ndim == 1:
            x = x[:, None]
        x = torch.from_numpy(x).float()
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# --------------------- Model -----------------------------

class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size=1, hidden=128, layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        d = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(d*hidden),
            nn.Linear(d*hidden, d*hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(d*hidden, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.rnn(x)
        # use last time step
        h = out[:, -1, :]  # (B, H*dir)
        logits = self.head(h).squeeze(-1)
        return logits

# --------------------- Training loop ---------------------

def compute_class_weight(y):
    # weight positive more if imbalance
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    # avoid div by zero
    if pos == 0: return 1.0
    return neg / pos

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, probs, labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=("cuda" if device == "cuda" else "cpu"), dtype=torch.bfloat16 if device=="cuda" else torch.float32):
            logits = model(x)
        p = (logits >= 0).long()
        pr = torch.sigmoid(logits)
        preds.append(p.cpu())
        probs.append(pr.cpu())
        labels.append(y.cpu())
    preds = torch.cat(preds).numpy()
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1)}, preds, probs, labels

def train(args):
    device = get_device()
    print(f"[runtime] device={device}  amp=True  tf32=True  cudnn.benchmark=True")

    win_dir = Path(args.windows_dir)
    X_path = win_dir / "X.npy"
    y_path = win_dir / "y.npy"
    meta_path = win_dir / "meta.csv"
    assert X_path.exists() and y_path.exists() and meta_path.exists(), "windows_dir must contain X.npy, y.npy, meta.csv"

    meta = pd.read_csv(meta_path)
    # normalize tic_id
    if "tic_id" not in meta.columns:
        raise RuntimeError("meta.csv must include 'tic_id'")
    meta["tic_id"] = meta["tic_id"].astype(str)

    y_all = np.load(y_path, mmap_mode="r")
    n = meta.shape[0]
    assert y_all.shape[0] == n, "y and meta length mismatch"

    # group-aware stratified split at TIC level
    # make one label per TIC by majority vote (or mean) to stratify
    tic_lab = (meta[["tic_id"]].assign(y=y_all).groupby("tic_id")["y"].mean() > 0.0).astype(int).reset_index()
    groups = tic_lab["tic_id"].values
    y_tic = tic_lab["y"].values

    # 5-fold to pick one split; use first fold as val/test; train = the rest
    # (simpler: train/val/test 70/15/15 via grouped split cascade)
    rng = np.random.RandomState(42)
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(gkf.split(X=groups, y=y_tic, groups=groups))
    tr_idx, val_idx = folds[0]  # TIC-level indices
    # split val into val/test halves
    val_half = int(len(val_idx) * 0.5)
    rng.shuffle(val_idx)
    val_tic = val_idx[:val_half]
    test_tic = val_idx[val_half:]

    tic_ids = tic_lab["tic_id"].values
    tr_tics = set(tic_ids[tr_idx])
    va_tics = set(tic_ids[val_tic])
    te_tics = set(tic_ids[test_tic])

    # map window indices by TIC
    idx_all = np.arange(n)
    tr_win = idx_all[meta["tic_id"].isin(tr_tics).values]
    va_win = idx_all[meta["tic_id"].isin(va_tics).values]
    te_win = idx_all[meta["tic_id"].isin(te_tics).values]

    # class weight from TRAIN windows only
    y_train = np.load(y_path, mmap_mode="r")[tr_win]
    pos_w = compute_class_weight(y_train)
    print(f"[split] windows: train={len(tr_win)} val={len(va_win)} test={len(te_win)}  | class_weight_pos={pos_w:.3f}")

    # datasets
    ds_tr = WindowDataset(X_path, y_path, meta, tr_win)
    ds_va = WindowDataset(X_path, y_path, meta, va_win)
    ds_te = WindowDataset(X_path, y_path, meta, te_win)

    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    loader_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    loader_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    sample = ds_tr[0][0]
    input_size = sample.shape[-1]
    model = BiLSTMClassifier(input_size=input_size, hidden=args.hidden, layers=args.layers, dropout=args.dropout, bidirectional=True)
    model.to(device)

    # loss/opt
    pos_weight = torch.tensor([pos_w], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler(device if device=="cuda" else "cpu", enabled=True)

    best_f1 = -1.0
    patience = max(5, args.epochs // 5)
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in loader_tr:
            x = x.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=("cuda" if device=="cuda" else "cpu"),
                                     dtype=torch.bfloat16 if device=="cuda" else torch.float32):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        scheduler.step()
        va_metrics, _, _, _ = evaluate(model, loader_va, device)
        print(f"[epoch {epoch:02d}] loss={running/len(loader_tr):.4f}  val_f1={va_metrics['f1']:.3f}  val_p={va_metrics['precision']:.3f}  val_r={va_metrics['recall']:.3f}")

        # early stopping on val F1
        if va_metrics["f1"] > best_f1:
            best_f1 = va_metrics["f1"]
            bad = 0
            best_path = Path("C:/CS_4280_Project/Code/models/exo_bilstm_retrain.pt")
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "args": vars(args)},
                       best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[early stop] no val F1 improvement for {patience} epochs.")
                break

    # load best and evaluate test
    state = torch.load("C:/CS_4280_Project/Code/models/exo_bilstm_retrain.pt", map_location=device)
    model.load_state_dict(state["model"])
    va_metrics, _, _, _ = evaluate(model, loader_va, device)
    te_metrics, _, _, _ = evaluate(model, loader_te, device)
    print(f"[final] val: {va_metrics} | test: {te_metrics}")

    # save metrics + splits
    out_metrics = Path("C:/CS_4280_Project/Code/models/metrics_retrain.txt")
    with out_metrics.open("w", encoding="utf-8") as f:
        f.write(f"val: {va_metrics}\n")
        f.write(f"test: {te_metrics}\n")
        f.write(f"class_weight_pos: {pos_w:.3f}\n")

    splits = {
        "train_tics": sorted(list(tr_tics)),
        "val_tics": sorted(list(va_tics)),
        "test_tics": sorted(list(te_tics))
    }
    out_splits = Path("C:/CS_4280_Project/Code/models/splits_retrain.json")
    out_splits.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print("[done] Saved model, metrics, and splits.")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_dir", type=str, required=True, help="Folder containing X.npy, y.npy, meta.csv")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    return ap.parse_args()

if __name__ == "__main__":
    train(parse_args())
