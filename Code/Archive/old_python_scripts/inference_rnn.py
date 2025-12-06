# -*- coding: utf-8 -*-
"""
Inference for the trained BiLSTM (RNN-only) exoplanet detector.

Reads prebuilt windows from a folder (X.npy, meta.csv),
auto-uses CUDA with AMP if available, scores windows, aggregates per TIC,
and writes:
  - Code/reports/inference_scores.csv       (per-window scores + preds)
  - Code/reports/inference_aggregated.csv   (per-TIC aggregated score + pred)

Threshold selection:
  --threshold_mode F2 (default)  -> recall-heavy; picked from Code/reports/pr_curve.csv
  --threshold_mode F1            -> balanced; picked from Code/reports/pr_curve.csv
  --threshold_mode fixed --fixed_threshold 0.5 -> explicit value
"""

import argparse, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast  # PyTorch 2.5+ AMP API

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------- Defaults (edit if needed) ----------------
ROOT = Path(r"C:\CS_4280_Project")
CODE = ROOT / "Code"
DEFAULT_WINDOWS_DIR = CODE / "data" / "windows_infer"
DEFAULT_MODEL = CODE / "models" / "exo_bilstm_cluster.pt"
PR_CURVE_CSV = CODE / "reports" / "pr_curve.csv"
REPORTS_DIR = CODE / "reports"

SEQ_LEN = 512        # model expects windows padded/trimmed to 512 (handled here)
FEATURES = 1
BATCH_SIZE = 256
USE_AMP = True
AMP_DTYPE = torch.float16

# ---------------- Runtime helpers ----------------
def device_setup():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"[runtime] device={dev}  amp={USE_AMP and dev=='cuda'}")
    return dev

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

def _extract_tic_from_path(p: str):
    m = re.search(r'(\d+)\s*_lightcurve\.csv$', str(p))
    return m.group(1) if m else None

def ensure_tic_id(meta: pd.DataFrame) -> pd.DataFrame:
    m = meta.copy()
    if "tic_id" not in m.columns:
        for c in ["curve_path","source_path","fname","file","path"]:
            if c in m.columns:
                derived = m[c].astype(str).apply(_extract_tic_from_path)
                if derived.notna().any():
                    m["tic_id"] = derived
                    break
    if "tic_id" not in m.columns:
        m["tic_id"] = m.index.astype(str)
        print("[warn] tic_id not found/derived; using row index as surrogate.")
    m["tic_id"] = m["tic_id"].astype(str)
    return m

# ---------------- Model (must match training) ----------------
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
            h = torch.cat([hn[-2], hn[-1]], dim=1)
        else:
            h = hn[-1]
        h = self.do2(h)
        return self.fc(h).squeeze(1)

# ---------------- Threshold selection ----------------
def load_threshold(mode: str, fixed_value: float|None) -> float:
    if mode.lower() == "fixed":
        if fixed_value is None:
            raise SystemExit("--threshold_mode fixed requires --fixed_threshold")
        return float(fixed_value)
    if not PR_CURVE_CSV.exists():
        print(f"[warn] {PR_CURVE_CSV} not found; defaulting threshold=0.5")
        return 0.5
    df = pd.read_csv(PR_CURVE_CSV)
    if "precision" not in df.columns or "recall" not in df.columns:
        print("[warn] pr_curve.csv missing precision/recall; defaulting threshold=0.5")
        return 0.5
    if "threshold" not in df.columns:
        df["threshold"] = np.linspace(0, 1, len(df))
    def fbeta(p, r, beta):
        denom = (beta**2)*p + r
        return (1+beta**2) * (p*r) / np.where(denom==0, np.nan, denom)
    df["F1"] = fbeta(df["precision"].values, df["recall"].values, 1.0)
    df["F2"] = fbeta(df["precision"].values, df["recall"].values, 2.0)
    idx = int(df["F2"].idxmax()) if mode.upper()=="F2" else int(df["F1"].idxmax())
    thr = float(df.loc[idx, "threshold"])
    print(f"[threshold] mode={mode.upper()} threshold≈{thr:.6f}")
    return thr

# ---------------- Preprocess windows ----------------
def preprocess_X(X: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        X = X[:, :, None]
    N, T, F = X.shape
    out = np.zeros((N, SEQ_LEN, F), dtype=np.float32)
    for i in range(N):
        for f in range(F):
            s = X[i, :, f]
            s = detrend_flux(s)
            s = robust_zscore(s)
            s = pad_or_trim(s, SEQ_LEN)
            out[i, :, f] = np.nan_to_num(s, nan=0.0)
    return out

# ---------------- Inference core ----------------
@torch.no_grad()
def score_windows(model, Xnp: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    scores = []
    for i in range(0, Xnp.shape[0], BATCH_SIZE):
        xb = torch.from_numpy(Xnp[i:i+BATCH_SIZE]).to(device)
        with autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=(USE_AMP and device=="cuda")):
            logits = model(xb)
            prob = torch.sigmoid(logits).float().cpu().numpy()
        scores.append(prob)
    return np.concatenate(scores, axis=0) if scores else np.array([], dtype=np.float32)

def aggregate_by_tic(meta: pd.DataFrame, scores: np.ndarray, how: str = "max") -> pd.DataFrame:
    m = meta.copy()
    m["score"] = scores
    if "tic_id" not in m.columns:
        m = ensure_tic_id(m)
    if how == "mean":
        agg = m.groupby("tic_id")["score"].mean().reset_index().rename(columns={"score":"score_mean"})
    else:
        agg = m.groupby("tic_id")["score"].max().reset_index().rename(columns={"score":"score_max"})
    return agg

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Inference for BiLSTM exoplanet detector (RNN-only).")
    ap.add_argument("--windows_dir", type=str, default=str(DEFAULT_WINDOWS_DIR),
                    help="Folder containing X.npy and meta.csv for the dataset to score.")
    ap.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL),
                    help="Path to trained model .pt")
    ap.add_argument("--threshold_mode", type=str, default="F2", choices=["F2","F1","fixed"],
                    help="Pick threshold from pr_curve.csv (F1/F2) or use a fixed value.")
    ap.add_argument("--fixed_threshold", type=float, default=None, help="Used only when threshold_mode=fixed")
    ap.add_argument("--aggregate", type=str, default="max", choices=["max","mean"],
                    help="How to aggregate window scores by tic_id.")
    args = ap.parse_args()

    device = device_setup()

    windows_dir = Path(args.windows_dir)
    model_path = Path(args.model_path)
    X_path = windows_dir / "X.npy"
    meta_path = windows_dir / "meta.csv"
    assert X_path.exists() and meta_path.exists(), f"Expected {X_path} and {meta_path}"

    # Load windows
    X = np.load(X_path, mmap_mode="r")
    meta = pd.read_csv(meta_path)
    meta = ensure_tic_id(meta)
    print(f"[data] X shape={X.shape}, windows={len(meta)}, unique TIC={meta['tic_id'].nunique()}")

    # Preprocess like training
    Xp = preprocess_X(X).astype(np.float32)

    # Load model
    model = BiLSTMClassifier(input_size=FEATURES, hidden1=128, hidden2=64, dropout=0.3, bidirectional=True).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)

    # Score windows
    scores = score_windows(model, Xp, device=device)

    # Threshold
    thr = load_threshold(args.threshold_mode, args.fixed_threshold)
    preds = (scores >= thr).astype(int)

    # Save per-window
    out_win = meta.copy()
    out_win["score"] = scores
    out_win["pred"] = preds
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_win_path = REPORTS_DIR / "inference_scores.csv"
    out_win.to_csv(out_win_path, index=False)

    # Aggregate per TIC
    agg = aggregate_by_tic(meta, scores, how=args.aggregate)
    score_col = "score_mean" if args.aggregate=="mean" else "score_max"
    agg["pred"] = (agg[score_col] >= thr).astype(int)
    out_agg_path = REPORTS_DIR / "inference_aggregated.csv"
    agg.to_csv(out_agg_path, index=False)

    # Console summary
    n_win_pos = int((out_win["pred"]==1).sum())
    n_tic_pos = int((agg["pred"]==1).sum())
    print(f"[done] Saved:\n  {out_win_path}\n  {out_agg_path}")
    print(f"[summary] windows predicted planet: {n_win_pos} / {len(out_win)}")
    print(f"[summary] TICs predicted planet:    {n_tic_pos} / {agg.shape[0]}")
    print(f"[threshold] {thr:.6f}  (mode={args.threshold_mode})  aggregate={args.aggregate}")

if __name__ == "__main__":
    main()
