# -*- coding: utf-8 -*-
"""
Parallel window builder for exoplanet light curves (Windows/PowerShell friendly).

- Preserves behavior from build_windows_v4.py (POS jitter, NEG sampling far from transit,
  robust detrend, BLS search, z-score, SEQ_LEN=2048 default).
- Adds CLI + parallel processing with ProcessPoolExecutor.
- Per-process RNG seeding for reproducibility across workers.
- Safe on Windows (uses if __name__ == "__main__").

Outputs:
  out_dir/X.npy, out_dir/y.npy, out_dir/meta.csv

Example:
  conda activate exo-lstm-gpu
  cd C:\CS_4280_Project\Code
  python .\build_windows_parallel_v5.py ^
    --processed_dir "C:\CS_4280_Project\test_dataset\simulated_dataset\processed" ^
    --out_dir "C:\CS_4280_Project\Code\data\windows_train" ^
    --seq_len 2048 ^
    --neg_per_pos 5 ^
    --n_jobs -1 ^
    --seed 42
"""
from __future__ import annotations

import argparse, os, math, sys
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

# ----------------------------- core config (defaults match your v4) -----------------------------
DEFAULT_SEQ_LEN   = 2048
DEFAULT_NEG_PER   = 5
DEFAULT_POS_WIDTH = 0.26
DEFAULT_POS_JITT  = 0.05
DEFAULT_NEG_SEP   = 0.18
DEFAULT_SEED      = 42

# ----------------------------- helpers (same math/flow as v4) ----------------------------------

def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    n = len(x)
    if n < 10:
        return x
    w = min(win, max(5, (n // 5) * 2 + 1))
    pad = w // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    mov = np.convolve(padded, np.ones(w) / w, mode="valid")
    return mov

def robust_detrend(flux: np.ndarray) -> np.ndarray:
    base = rolling_median(flux, 401)
    base = np.where(base == 0, np.nanmedian(flux), base)
    return flux / base

def zscore(arr: np.ndarray) -> np.ndarray:
    mu = np.nanmean(arr); sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return (arr - mu) / sd

def bls_best(time: np.ndarray, flux: np.ndarray) -> Optional[Tuple[float, float, float, float, float]]:
    mask = np.isfinite(time) & np.isfinite(flux)
    t, f = time[mask], flux[mask]
    if len(t) < 500:
        return None
    tspan = t.max() - t.min()
    if not np.isfinite(tspan) or tspan <= 0:
        return None

    min_period = 0.30
    max_period = float(min(30.0, max(2.0, 0.9 * tspan)))
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

def phase_window(rng: np.random.Generator, time: np.ndarray, flux: np.ndarray,
                 period: float, t0: float, center_phase: float, width: float,
                 seq_len: int) -> Optional[np.ndarray]:
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

def process_one(csv_path: Path, seq_len: int, neg_per_pos: int, pos_width: float,
                pos_jitter: float, neg_min_sep: float,
                seed: int, worker_id: int) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """Process a single lightcurve CSV → (X_i, y_i, meta_i). RNG unique per worker+file."""
    rng = np.random.default_rng(seed + worker_id * 1_000_003 + (hash(csv_path.name) & 0xffff))
    try:
        df = pd.read_csv(csv_path)
        if not {"time", "flux"}.issubset(df.columns):
            return None
        t = df["time"].to_numpy(float)
        f = df["flux"].to_numpy(float)
        m = np.isfinite(t) & np.isfinite(f) & (f != 0)
        t, f = t[m], f[m]
        if len(f) < 500:
            return None

        f = robust_detrend(f)
        f = f / np.nanmedian(f)

        got = bls_best(t, f)
        if got is None:
            return None
        period, duration, depth, snr, t0 = got

        # positive windows: jitter around 0 phase
        pos_centers = [0.0] + list(rng.uniform(-pos_jitter, pos_jitter, size=2))
        Xs: List[np.ndarray] = []
        ys: List[int] = []

        for c in pos_centers:
            pos = phase_window(rng, t, f, period, t0, center_phase=c % 1.0,
                               width=pos_width, seq_len=seq_len)
            if pos is not None:
                Xs.append(pos); ys.append(1)

        # negatives: far from transit
        for _ in range(neg_per_pos):
            off = float(rng.uniform(neg_min_sep, 1.0 - neg_min_sep))
            neg = phase_window(rng, t, f, period, t0, center_phase=off, width=pos_width, seq_len=seq_len)
            if neg is not None:
                Xs.append(neg); ys.append(0)

        if not Xs:
            return None

        tic = csv_path.stem.split('_')[0]
        meta_rows = [{
            "tic_id": tic, "period": period, "duration": duration, "depth": depth,
            "t0": t0, "bls_power": snr, "label": int(l)
        } for l in ys]

        return np.stack(Xs), np.array(ys, dtype=np.int64), pd.DataFrame(meta_rows)
    except Exception as e:
        # Return a lightweight error record via None; caller logs filename
        return None

# ----------------------------- main (parallel) ----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parallel window builder for light curves")
    ap.add_argument("--processed_dir", type=str, required=True, help="Folder of *_lightcurve.csv files")
    ap.add_argument("--out_dir",      type=str, required=True, help="Output folder for X.npy, y.npy, meta.csv")
    ap.add_argument("--seq_len",      type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--neg_per_pos",  type=int, default=DEFAULT_NEG_PER)
    ap.add_argument("--pos_width",    type=float, default=DEFAULT_POS_WIDTH)
    ap.add_argument("--pos_jitter",   type=float, default=DEFAULT_POS_JITT)
    ap.add_argument("--neg_min_sep",  type=float, default=DEFAULT_NEG_SEP)
    ap.add_argument("--n_jobs",       type=int, default=-1, help="-1→all cores; 1→sequential")
    ap.add_argument("--seed",         type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(processed.glob("*_lightcurve.csv"))
    if not files:
        raise SystemExit(f"No *_lightcurve.csv under {processed}")

    n_jobs = os.cpu_count() if args.n_jobs in (-1, 0) else max(1, args.n_jobs)
    print(f"- PROCESSED: {processed}")
    print(f"- OUTDIR:    {out_dir}")
    print(f"- Files:     {len(files)}  | seq_len={args.seq_len} | neg/pos={args.neg_per_pos} | jobs={n_jobs}")

    Xs: List[np.ndarray] = []
    Ys: List[np.ndarray] = []
    Ms: List[pd.DataFrame] = []

    t0 = time.time()
    if n_jobs == 1:
        # sequential (debug)
        for i, p in enumerate(files, 1):
            out = process_one(p, args.seq_len, args.neg_per_pos, args.pos_width,
                              args.pos_jitter, args.neg_min_sep, args.seed, worker_id=0)
            if out is None:
                print(f"[skip] {p.name}")
                continue
            Xi, yi, mi = out
            Xs.append(Xi); Ys.append(yi); Ms.append(mi)
            print(f"[ok]   {p.name}: {len(yi)} examples  ({i}/{len(files)})")
    else:
        # parallel
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as ex:
            futs = []
            for worker_id, p in enumerate(files):
                futs.append(ex.submit(
                    process_one, p, args.seq_len, args.neg_per_pos, args.pos_width,
                    args.pos_jitter, args.neg_min_sep, args.seed, worker_id
                ))
            done = 0
            for fut, p in zip(as_completed(futs), files):
                res = fut.result()
                done += 1
                if res is None:
                    print(f"[skip] {p.name}    ({done}/{len(files)})")
                    continue
                Xi, yi, mi = res
                Xs.append(Xi); Ys.append(yi); Ms.append(mi)
                print(f"[ok]   {p.name}: {len(yi)} examples   ({done}/{len(files)})")

    if not Xs:
        raise SystemExit("no examples built")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)
    meta = pd.concat(Ms, ignore_index=True)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y)
    meta.to_csv(out_dir / "meta.csv", index=False)

    dt = time.time() - t0
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    print("Saved:")
    print(" ", out_dir / "X.npy")
    print(" ", out_dir / "y.npy")
    print(" ", out_dir / "meta.csv")
    print(f"Shapes: X={X.shape}  y={y.shape}  | pos={pos} neg={neg} | elapsed={dt:.1f}s")

if __name__ == "__main__":
    # Important for Windows parallelism
    mp.freeze_support()
    main()
