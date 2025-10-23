# -*- coding: utf-8 -*-
"""
Parallel window builder for *test* datasets (CPU-only, but multi-core).
- Reads CSV light curves from a "processed" directory
- Runs a quick BLS to find a candidate transit
- Builds positive/negative phase windows
- Writes X.npy, y.npy, meta.csv to an output folder

Usage (PowerShell):
  conda activate exo-lstm-gpu
  cd C:\CS_4280_Project\Code
  python .\build_windows_infer_v2.py ^
    --processed_dir "C:\CS_4280_Project\test_dataset\simulated_dataset\processed" ^
    --out_dir "C:\CS_4280_Project\Code\data\windows_infer" ^
    --neg_per_pos 5 --seq_len 2048 --n_jobs -1
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from joblib import Parallel, delayed
from typing import Optional, Tuple, Dict, Any

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Build phase windows for inference (parallel).")
    p.add_argument("--processed_dir", type=str,
                   default=r"C:\CS_4280_Project\test_dataset\simulated_dataset\processed",
                   help="Folder containing *_lightcurve.csv")
    p.add_argument("--out_dir", type=str,
                   default=r"C:\CS_4280_Project\Code\data\windows_infer",
                   help="Output folder for X.npy, y.npy, meta.csv")
    p.add_argument("--seq_len", type=int, default=2048, help="Length of each window")
    p.add_argument("--neg_per_pos", type=int, default=5, help="Negatives per positive")
    p.add_argument("--pos_width", type=float, default=0.26, help="Phase width for windows")
    p.add_argument("--pos_jitter", type=float, default=0.05, help="Jitter around phase 0 for positives")
    p.add_argument("--neg_min_sep", type=float, default=0.18, help="Min phase distance from 0 for negatives")
    p.add_argument("--min_period", type=float, default=0.30, help="BLS min period (days)")
    p.add_argument("--max_period_cap", type=float, default=30.0, help="Upper cap for BLS max period")
    p.add_argument("--n_jobs", type=int, default=-1, help="Parallel jobs (-1 = all cores)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()

# ---------------- helpers ----------------
def zscore(x: np.ndarray) -> np.ndarray:
    m = np.isfinite(x)
    if m.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)
    med = np.nanmedian(x[m])
    mad = np.nanmedian(np.abs(x[m] - med)) + 1e-8
    return ((x - med) / (1.4826 * mad)).astype(np.float32)

def load_curve(csv_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns or "flux" not in df.columns:
        return None
    t = df["time"].to_numpy(dtype=float)
    f = df["flux"].to_numpy(dtype=float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 10:
        return None
    # simple poly detrend
    tt = np.arange(t.size, dtype=float)
    mask = np.isfinite(f)
    if mask.sum() >= 3:
        coeffs = np.polyfit(tt[mask], f[mask], deg=2)
        trend = np.polyval(coeffs, tt)
        f = f - trend
    return t, f

def run_bls(time: np.ndarray, flux: np.ndarray,
            min_period: float, max_period_cap: float) -> Optional[Tuple[float, float, float, float, float]]:
    t, f = time, flux
    tspan = t.max() - t.min()
    if not np.isfinite(tspan) or tspan <= 0:
        return None
    max_period = float(min(max_period_cap, max(2.0, 0.9 * tspan)))
    if max_period <= min_period:
        return None
    durations = np.array([0.02, 0.04, 0.06, 0.08, 0.10], dtype=float)
    durations = durations[durations < min_period * 0.95]
    if durations.size == 0:
        return None
    bls = BoxLeastSquares(t, f)
    res = bls.autopower(durations, minimum_period=min_period, maximum_period=max_period,
                        objective="snr", frequency_factor=1.0)
    i = int(np.nanargmax(res.power))
    P = float(res.period[i]); dur = float(res.duration[i])
    depth = float(getattr(res, "depth", np.array([np.nan]))[i])
    snr = float(res.power[i])
    t0_arr = getattr(res, "transit_time", getattr(res, "t0", None))
    t0 = float(t0_arr[i]) if t0_arr is not None else float(t[0])
    return P, dur, depth, snr, t0

def phase_window(time: np.ndarray, flux: np.ndarray, period: float, t0: float,
                 center_phase: float, width: float, seq_len: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    phase = ((time - t0) % period) / period
    lo, hi = (center_phase - width/2) % 1.0, (center_phase + width/2) % 1.0
    sel = (phase >= lo) & (phase <= hi) if lo < hi else ((phase >= lo) | (phase <= hi))
    f = flux[sel]
    if len(f) == 0:
        return None
    if len(f) < seq_len:
        idx = rng.choice(len(f), seq_len, replace=True)
    else:
        idx = np.linspace(0, len(f)-1, seq_len).astype(int)
    return zscore(f[idx].astype(np.float32))

def process_file(csv_path: Path, params: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    # independent RNG per file (stable across runs)
    seed = (params["seed"] + hash(csv_path.stem) % (2**31 - 1)) & 0x7fffffff
    rng = np.random.default_rng(seed)

    lc = load_curve(csv_path)
    if lc is None:
        return None
    t, f = lc

    bls = run_bls(t, f, params["min_period"], params["max_period_cap"])
    if bls is None:
        return None
    period, duration, depth, snr, t0 = bls

    Xs, ys, metas = [], [], []

    # one positive near phase 0 with jitter
    pos = phase_window(t, f, period, t0,
                       center_phase=(0.0 + float(rng.uniform(-params["pos_jitter"], params["pos_jitter"]))) % 1.0,
                       width=params["pos_width"], seq_len=params["seq_len"], rng=rng)
    if pos is not None:
        Xs.append(pos); ys.append(1)

    # negatives away from transit center
    for _ in range(params["neg_per_pos"]):
        off = float(rng.uniform(params["neg_min_sep"], 1.0 - params["neg_min_sep"]))
        neg = phase_window(t, f, period, t0, center_phase=off,
                           width=params["pos_width"], seq_len=params["seq_len"], rng=rng)
        if neg is not None:
            Xs.append(neg); ys.append(0)

    if not Xs:
        return None

    tic = csv_path.stem.split('_')[0]
    for y in ys:
        metas.append({
            "tic_id": tic, "period": period, "duration": duration, "depth": depth,
            "t0": t0, "bls_power": snr, "label": int(y)
        })

    return np.stack(Xs), np.array(ys, dtype=np.int64), pd.DataFrame(metas)

# ---------------- main ----------------
def main():
    args = parse_args()

    processed = Path(args.processed_dir)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    params = dict(
        seq_len=args.seq_len,
        neg_per_pos=args.neg_per_pos,
        pos_width=args.pos_width,
        pos_jitter=args.pos_jitter,
        neg_min_sep=args.neg_min_sep,
        min_period=args.min_period,
        max_period_cap=args.max_period_cap,
        seed=args.seed
    )

    files = sorted(processed.glob("*_lightcurve.csv"))
    if not files:
        raise SystemExit(f"No CSVs found under {processed}")

    print(f"- PROCESSED points to: {processed}")
    print(f"- OUTDIR: {outdir}")
    print(f"- Files: {len(files)} | seq_len={args.seq_len} | neg/pos={args.neg_per_pos}")

    # Parallel pass over files
    results = Parallel(n_jobs=args.n_jobs, prefer="threads" if len(files) < 8 else "processes")(
        delayed(process_file)(p, params) for p in files
    )

    Xs, Ys, Ms = [], [], []
    ok = 0; skipped = 0
    for p, r in zip(files, results):
        if r is None:
            print(f"[skip] {p.name}")
            skipped += 1
        else:
            Xi, yi, mi = r
            Xs.append(Xi); Ys.append(yi); Ms.append(mi)
            ok += 1

    if not Xs:
        raise SystemExit("No examples built (all skipped).")

    X = np.concatenate(Xs, 0); y = np.concatenate(Ys, 0)
    meta = pd.concat(Ms, ignore_index=True)

    np.save(outdir / "X.npy", X)
    np.save(outdir / "y.npy", y)
    meta.to_csv(outdir / "meta.csv", index=False)

    print("\nSaved:")
    print(" ", outdir / "X.npy")
    print(" ", outdir / "y.npy")
    print(" ", outdir / "meta.csv")
    print(f"Shapes: X={X.shape}  y={y.shape}  | pos={int((y==1).sum())}  neg={int((y==0).sum())}")
    print(f"Files processed: {ok}  | skipped: {skipped}")

if __name__ == "__main__":
    main()
