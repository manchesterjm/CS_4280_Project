# -*- coding: utf-8 -*-
# Pure RNN (BiLSTM) training with TRAIN-ONLY clustering for balanced batches.
# - Auto-selects CUDA if available, uses AMP (fp16) for speed on GPU (PyTorch 2.5+ APIs).
# - Uses PCA+KMeans on TRAIN ONLY to build a cluster-balanced sampler.
# - Robust splits handling: accepts existing splits.json as row indices OR TIC IDs.
#   When creating new splits, saves TIC IDs for compatibility with evaluators.
#
# Outputs:
#   Code\models\exo_bilstm_cluster.pt
#   Code\reports\pr_curve.csv
#   Code\models\metrics_cluster_v3.txt

import os, re, json, math, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

from torch.amp import autocast, GradScaler  # <-- modern AMP APIs (PyTorch 2.5+)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# CONFIG (adjust as needed)
# -----------------------------
ROOT = Path(r"C:\CS_4280_Project")
CODE = ROOT / "Code"
DATA_DIR = CODE / "data" / "windows"
REPORTS_DIR = CODE / "reports"
MODELS_DIR = CODE / "models"

SEED = 42

# Auto device + runtime knobs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True                    # automatic mixed precision on GPU
AMP_DTYPE = torch.float16         # good default for RTX 30xx
CUDNN_BENCHMARK = True            # tune cuDNN for input sizes
ALLOW_TF32 = True                 # allow TF32 on Ampere+

# Data/model/training
SEQ_LEN = 512          # we will center-trim/pad windows to this length
FEATURES = 1           # 1 if just flux (expects X[N, T] or X[N, T, 1])
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 0.0
PATIENCE = 6           # early stop on val AP (PR-AUC)
USE_CLASS_WEIGHTS = True

# Clustering (TRAIN-ONLY)
K_CLUSTERS = 8
PCA_DIM = 16
BALANCE_MODE = "uniform"  # currently only 'uniform'

# Splits file
SPLITS_JSON = MODELS_DIR / "splits.json"

# -----------------------------
# Utilities / Repro
# -----------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def ensure_dirs():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

ensure_dirs()

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    if ALLOW_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

print(f"[runtime] device={DEVICE}  amp={USE_AMP and DEVICE=='cuda'}  tf32={ALLOW_TF32}  cudnn.benchmark={CUDNN_BENCHMARK and DEVICE=='cuda'}")

# -----------------------------
# Preprocessing helpers
# -----------------------------
def detrend_flux(flux: np.ndarray, poly_deg: int = 2) -> np.ndarray:
    t = np.arange(len(flux))
    m = np.isfinite(flux)
    if m.sum() < poly_deg + 1:
        return flux - np.nanmedian(flux)
    coeffs = np.polyfit(t[m], flux[m], deg=poly_deg)
    trend = np.polyval(coeffs, t)
    return flux - trend

def robust_zscore(x: np.ndarray) -> np.ndarray:
    m = np.isfinite(x)
    if m.sum() == 0:
        return np.zeros_like(x)
    mu = np.nanmedian(x[m])
    mad = np.nanmedian(np.abs(x[m] - mu)) + 1e-8
    return (x - mu) / (1.4826 * mad)

def pad_or_trim(x: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(x)
    if n == seq_len:
        return x
    if n > seq_len:
        start = (n - seq_len) // 2
        return x[start:start+seq_len]
    pad_left = (seq_len - n) // 2
    pad_right = seq_len - n - pad_left
    return np.pad(x, (pad_left, pad_right), constant_values=np.nan)

# -----------------------------
# Dataset
# -----------------------------
class LightCurveWindows(Dataset):
    """
    Loads from X.npy, y.npy, meta.csv. Applies detrend + robust zscore + pad/trim.
    Returns tensors: (T,F) float32 and label int64.
    """
    def __init__(self, data_dir: Path, seq_len=SEQ_LEN, features=FEATURES, indices=None):
        self.data_dir = Path(data_dir)
        self.seq_len = seq_len
        self.features = features

        X_path = self.data_dir / "X.npy"
        y_path = self.data_dir / "y.npy"
        meta_path = self.data_dir / "meta.csv"
        assert X_path.exists() and y_path.exists(), f"Missing {X_path} or {y_path}"

        self.X = np.load(X_path, mmap_mode="r")  # (N,T) or (N,T,F)
        self.y = np.load(y_path).astype(np.int64)
        self.meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame({"row": np.arange(len(self.y))})
        if "tic_id" in self.meta.columns:
            self.meta["tic_id"] = self.meta["tic_id"].astype(str)

        if indices is not None:
            self.idx = np.array(indices, dtype=int)
        else:
            self.idx = np.arange(len(self.y))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = self.X[j]  # (T,) or (T,F)
        y = int(self.y[j])

        if x.ndim == 1:
            x = x[:, None]  # (T,1)

        out = []
        for f in range(x.shape[1]):
            s = detrend_flux(x[:, f])
            s = robust_zscore(s)
            s = pad_or_trim(s, self.seq_len)
            out.append(s)
        Xp = np.stack(out, axis=1)           # (T,F)
        Xp = np.nan_to_num(Xp, nan=0.0)      # replace NaN padding with 0
        return torch.tensor(Xp, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# -----------------------------
# Splits helpers (row indices <-> TIC IDs)
# -----------------------------
def _extract_tic_from_path(p: str):
    m = re.search(r'(\d+)\s*_lightcurve\.csv$', str(p))
    return m.group(1) if m else None

def build_grouped_splits(meta: pd.DataFrame, y: np.ndarray, seed=SEED, val_frac=0.15, test_frac=0.15):
    """
    Build splits grouped by TIC ID (preferred). Falls back to other ID columns or random indices.
    Returns dict with lists of TIC strings for train/val/test (to avoid leakage).
    """
    rng = np.random.RandomState(seed)

    m = meta.copy()
    if "tic_id" not in m.columns:
        for c in ["curve_path", "source_path", "fname", "file", "path"]:
            if c in m.columns:
                derived = m[c].apply(_extract_tic_from_path)
                if derived.notna().any():
                    m["tic_id"] = derived
                    break

    id_cols = ["tic_id", "kic", "kepid", "star_id", "object_id", "target_id", "source_id"]
    chosen = None
    for col in id_cols:
        if col in m.columns:
            chosen = col
            break

    if chosen is None:
        idx = np.arange(len(y))
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(test_frac * n)
        n_val  = int(val_frac * n)
        test = idx[:n_test].tolist()
        val  = idx[n_test:n_test+n_val].tolist()
        train = idx[n_test+n_val:].tolist()
        return {"train": [str(i) for i in train],
                "val":   [str(i) for i in val],
                "test":  [str(i) for i in test],
                "_by":   "row_index"}
    else:
        groups = m[chosen].astype(str).to_numpy()
        uniq = np.array(sorted(pd.unique(groups)))
        rng.shuffle(uniq)
        n_test_g = int(len(uniq) * test_frac)
        n_val_g  = int(len(uniq) * val_frac)
        test_g = set(uniq[:n_test_g])
        val_g  = set(uniq[n_test_g:n_test_g+n_val_g])
        train_g = [g for g in uniq if g not in test_g and g not in val_g]
        return {"train": list(map(str, train_g)),
                "val":   list(map(str, val_g)),
                "test":  list(map(str, test_g)),
                "_by":   chosen}

def ids_to_row_indices(id_list, meta: pd.DataFrame):
    """
    Map a list of TIC (or other ID) strings to row indices using meta.
    """
    if not id_list:
        return []
    m = meta.copy()
    if "tic_id" not in m.columns:
        for c in ["curve_path", "source_path", "fname", "file", "path"]:
            if c in m.columns:
                derived = m[c].apply(_extract_tic_from_path)
                if derived.notna().any():
                    m["tic_id"] = derived
                    break
    for col in ["tic_id", "kic", "kepid", "star_id", "object_id", "target_id", "source_id"]:
        if col in m.columns:
            m[col] = m[col].astype(str)
            pos = {}
            for i, v in enumerate(m[col].to_numpy()):
                pos.setdefault(v, []).append(i)
            out, missing = [], []
            for v in id_list:
                vv = str(v)
                if vv in pos:
                    out.extend(pos[vv])
                else:
                    missing.append(vv)
            if missing:
                print(f"[warn] {len(missing)} IDs not found in meta[{col}] (e.g., {missing[:5]})")
            return sorted(set(out))
    print("[warn] No usable ID column found to map IDs to rows; returning empty.")
    return []

def normalize_indices(raw_list, meta: pd.DataFrame, n_rows: int):
    """
    Accept existing splits.json content as either:
      - row indices (ints < n_rows)
      - TIC/ID strings
    Returns a list of row indices.
    """
    if raw_list is None:
        return []
    items = list(raw_list)
    if not items:
        return []
    try:
        ints = [int(x) for x in items]
        if max(ints) < n_rows and min(ints) >= 0:
            return sorted(set(ints))
    except Exception:
        pass
    ids = [str(x) for x in items]
    return ids_to_row_indices(ids, meta)

# -----------------------------
# Model (Pure RNN)
# -----------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size=FEATURES, hidden1=128, hidden2=64, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True, bidirectional=bidirectional)
        self.do1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1*(2 if bidirectional else 1), hidden2, batch_first=True, bidirectional=bidirectional)
        self.do2 = nn.Dropout(dropout)
        out_dim = hidden2*(2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):  # x: (B,T,F)
        x, _ = self.lstm1(x)
        x = self.do1(x)
        x, (hn, _) = self.lstm2(x)
        if self.lstm2.bidirectional:
            h = torch.cat([hn[-2], hn[-1]], dim=1)  # (B, 2*hidden2)
        else:
            h = hn[-1]
        h = self.do2(h)
        logits = self.fc(h).squeeze(1)  # (B,)
        return logits

# -----------------------------
# Cluster-balanced sampler (TRAIN ONLY)
# -----------------------------
class ClusterBalancedSampler(Sampler):
    def __init__(self, cluster_labels: np.ndarray, batch_size: int, k_clusters: int, mode="uniform"):
        self.batch_size = batch_size
        self.mode = mode
        self.k = k_clusters
        self.cluster_to_indices = defaultdict(list)
        for i, c in enumerate(cluster_labels):
            self.cluster_to_indices[int(c)].append(i)
        self.deques = {}
        for c, lst in self.cluster_to_indices.items():
            random.shuffle(lst)
            self.deques[c] = deque(lst)
        self.num_samples = sum(len(v) for v in self.cluster_to_indices.values())

    def __len__(self):
        return max(1, math.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        per = max(1, self.batch_size // self.k)
        clusters = list(self.cluster_to_indices.keys())
        yielded = 0
        max_batches = self.__len__()
        while yielded < max_batches:
            batch = []
            random.shuffle(clusters)
            for c in clusters:
                dq = self.deques[c]
                take = min(per, len(dq))
                for _ in range(take):
                    batch.append(dq.popleft())
                if len(dq) == 0 and len(self.cluster_to_indices[c]) > 0:
                    dq.extend(self.cluster_to_indices[c])
                    random.shuffle(dq)
                if len(batch) >= self.batch_size:
                    break
            if not batch:
                return
            yielded += 1
            yield batch[:self.batch_size]

# -----------------------------
# Train / Eval helpers
# -----------------------------
def compute_metrics(y_true, y_score):
    ap = average_precision_score(y_true, y_score)
    try:
        roc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc = float("nan")
    return ap, roc

@torch.no_grad()
def run_inference(model, loader, device=DEVICE):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(prob)
    y = np.concatenate(ys) if ys else np.array([])
    p = np.concatenate(ps) if ps else np.array([])
    return y, p

def save_pr_curve_csv(y, p, out_csv: Path):
    if y.size == 0:
        print("[warn] No validation/test examples to write PR curve.")
        return
    prec, rec, thr = precision_recall_curve(y, p)
    df = pd.DataFrame({"threshold": np.r_[thr, 1.0], "precision": prec, "recall": rec})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

# -----------------------------
# Main
# -----------------------------
def main():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    X_path = DATA_DIR / "X.npy"
    y_path = DATA_DIR / "y.npy"
    meta_path = DATA_DIR / "meta.csv"
    assert X_path.exists() and y_path.exists(), f"Missing X/y in {DATA_DIR}"

    X = np.load(X_path, mmap_mode="r")       # (N,T) or (N,T,F)
    y = np.load(y_path).astype(np.int64)     # (N,)
    meta = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame({"row": np.arange(len(y))})
    if "tic_id" in meta.columns:
        meta["tic_id"] = meta["tic_id"].astype(str)

    print("[meta] columns:", list(meta.columns)[:12], "| rows:", len(meta), "| X shape:", X.shape, "| pos:", int((y==1).sum()))

    # Load or build splits (save as TIC IDs for compatibility)
    if SPLITS_JSON.exists():
        with open(SPLITS_JSON, "r") as f:
            old = json.load(f)
        n_rows = len(y)
        train_idx = normalize_indices(old.get("train", []), meta, n_rows)
        val_idx   = normalize_indices(old.get("val",   []), meta, n_rows)
        test_idx  = normalize_indices(old.get("test",  []), meta, n_rows)
        if not train_idx or not val_idx or not test_idx:
            print("[warn] Existing splits.json could not be normalized; rebuilding splits.")
            tic_splits = build_grouped_splits(meta, y, seed=SEED)
            with open(SPLITS_JSON, "w") as f:
                json.dump(tic_splits, f)
            train_idx = normalize_indices(tic_splits["train"], meta, len(y))
            val_idx   = normalize_indices(tic_splits["val"],   meta, len(y))
            test_idx  = normalize_indices(tic_splits["test"],  meta, len(y))
    else:
        tic_splits = build_grouped_splits(meta, y, seed=SEED)
        SPLITS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(SPLITS_JSON, "w") as f:
            json.dump(tic_splits, f)
        train_idx = normalize_indices(tic_splits["train"], meta, len(y))
        val_idx   = normalize_indices(tic_splits["val"],   meta, len(y))
        test_idx  = normalize_indices(tic_splits["test"],  meta, len(y))

    # Datasets
    ds_train = LightCurveWindows(DATA_DIR, indices=train_idx)
    ds_val   = LightCurveWindows(DATA_DIR, indices=val_idx)
    ds_test  = LightCurveWindows(DATA_DIR, indices=test_idx)

    # -----------------------------
    # TRAIN-ONLY clustering
    # -----------------------------
    def to_feat(ds: LightCurveWindows):
        Xf = []
        for i in range(len(ds)):
            x, _ = ds[i]     # tensor (T,F)
            Xf.append(x.numpy().reshape(-1))
        return np.stack(Xf, axis=0) if Xf else np.zeros((0, SEQ_LEN*FEATURES), dtype=np.float32)

    X_train_feat = to_feat(ds_train)
    if X_train_feat.shape[0] == 0:
        raise SystemExit("No training examples available after splits; check your data and meta.csv")

    pca = PCA(n_components=min(PCA_DIM, X_train_feat.shape[1]), random_state=SEED)
    X_train_pca = pca.fit_transform(X_train_feat)

    kmeans = KMeans(n_clusters=min(K_CLUSTERS, len(ds_train)), random_state=SEED, n_init="auto")
    train_clusters = kmeans.fit_predict(X_train_pca)

    # Class weights by label (optional)
    cw = None
    if USE_CLASS_WEIGHTS and len(ds_train) > 0:
        y_train = np.array([ds_train[i][1].item() for i in range(len(ds_train))])
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        if pos > 0 and neg > 0:
            w_pos = neg / (pos + 1e-8)
            w_neg = 1.0
            cw = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=DEVICE)

    # Loaders (Windows-friendly: num_workers=0)
    sampler = ClusterBalancedSampler(train_clusters, batch_size=BATCH_SIZE,
                                     k_clusters=len(np.unique(train_clusters)),
                                     mode=BALANCE_MODE)
    train_loader = DataLoader(ds_train, batch_sampler=sampler, num_workers=0)
    val_loader   = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model / Optim / Loss
    model = BiLSTMClassifier(input_size=FEATURES, hidden1=128, hidden2=64, dropout=0.3, bidirectional=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    # Modern AMP scaler (noop on CPU)
    scaler = GradScaler(device="cuda", enabled=(USE_AMP and DEVICE == "cuda"))

    best_ap = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, EPOCHS+1):
        model.train()
        running = 0.0
        steps = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE).float()

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(USE_AMP and DEVICE=="cuda")):
                logits = model(xb)
                loss = bce(logits, yb)  # per-sample
                if cw is not None:
                    weights = torch.where(yb > 0.5, cw[1], cw[0])
                    loss = (loss * weights).mean()
                else:
                    loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item())
            steps += 1

        # Validation
        yv, pv = run_inference(model, val_loader, device=DEVICE)
        ap, roc = compute_metrics(yv, pv)
        print(f"Epoch {epoch:02d} | train_loss={running/max(1,steps):.4f} | val_AP={ap:.4f} | val_ROC={roc:.4f}")

        # Early stopping on AP
        if ap > best_ap + 1e-5:
            best_ap = ap
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement in {PATIENCE} epochs).")
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Test
    yt, pt = run_inference(model, test_loader, device=DEVICE)
    ap_t, roc_t = compute_metrics(yt, pt)
    print(f"TEST: AP={ap_t:.4f}  ROC-AUC={roc_t:.4f}")

    # Save artifacts
    torch.save(model.state_dict(), MODELS_DIR / "exo_bilstm_cluster.pt")
    # PR curve using val+test for a smoother curve if both exist, else whichever exists
    if (yv.size and yt.size):
        y_all = np.concatenate([yv, yt])
        p_all = np.concatenate([pv, pt])
    elif yv.size:
        y_all, p_all = yv, pv
    else:
        y_all, p_all = yt, pt
    save_pr_curve_csv(y_all, p_all, REPORTS_DIR / "pr_curve.csv")

    metrics_txt = MODELS_DIR / "metrics_cluster_v3.txt"
    with open(metrics_txt, "w") as f:
        f.write(f"val_AP={best_ap:.6f}\n")
        f.write(f"test_AP={ap_t:.6f}\n")
        f.write(f"test_ROC_AUC={roc_t:.6f}\n")

    print("Saved:",
          MODELS_DIR / "exo_bilstm_cluster.pt",
          REPORTS_DIR / "pr_curve.csv",
          metrics_txt,
          sep="\n")

if __name__ == "__main__":
    main()
