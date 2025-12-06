# -*- coding: utf-8 -*-
r"""
Parallel window builder for exoplanet light curves (Windows/PowerShell friendly).

Outputs:
  out_dir/X.npy
  out_dir/y.npy
  out_dir/meta.csv

Examples (PowerShell):
  python .\build_windows_parallel_v6.py `
    --processed_dir "C:\CS_4280_Project\test_dataset\simulated_dataset\processed" `
    --out_dir "C:\CS_4280_Project\Code\data\windows_train" `
    --seq_len 2048 `
    --neg_per_pos 5 `
    --n_jobs -1 `
    --seed 42 `
    --manifest "C:\CS_4280_Project\test_dataset\simulated_dataset\manifest.csv"
"""
from __future__ import annotations

import argparse, os, sys, math, time
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

# ----------------------------- defaults -----------------------------
DEFAULT_SEQ_LEN   = 2048
DEFAULT_NEG_PER   = 5
DEFAULT_POS_WIDTH = 0.26     # fraction of phase to include in a window
DEFAULT_POS_JITT  = 0.05     # jitter around transit phase center for variety
DEFAULT_NEG_SEP   = 0.18     # keep negatives far from transit center
DEFAULT_MIN_PTS   = 500
DEFAULT_SEED      = 42
DEFAULT_CHUNK     = 2000     # write to disk every N examples
DEFAULT_JOBS      = -1
DEFAULT_USE_MAN   = 1        # use manifest labels if provided

# ----------------------------- helpers -----------------------------

def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    n = len(x)
    if n < 10:
        return x
    w = min(win, max(5, (n // 5) * 2 + 1))
    pad = w // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    mov = np.convolve(padded, np.ones(w)/w, mode="valid")
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
    """Return (period, duration, depth, snr(power), t0) or None if failed."""
    mask = np.isfinite(time) & np.isfinite(flux)
    t, f = time[mask], flux[mask]
    if len(t) < DEFAULT_MIN_PTS:
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
    lo = (center_phase - width/2) % 1.0
    hi = (center_phase + width/2) % 1.0
    sel = (phase >= lo) & (phase <= hi) if lo < hi else ((phase >= lo) | (phase <= hi))
    f = flux[sel]
    if len(f) == 0:
        return None
    if len(f) < seq_len:
        idx = rng.choice(len(f), seq_len, replace=True)
    else:
        idx = np.linspace(0, len(f)-1, seq_len).astype(int)
    win = f[idx].astype(np.float32)
    return zscore(win)

def safe_label_to_int(v) -> int:
    try:
        return int(v)
    except Exception:
        s = str(v).strip().lower()
        return 1 if s in ("1","true","planet","pos","positive","yes","y") else 0

# ----------------------------- worker -----------------------------

def process_one(csv_path: Path, seq_len: int, neg_per_pos: int, pos_width: float,
                pos_jitter: float, neg_min_sep: float,
                seed: int, worker_id: int,
                label_map: Optional[Dict[str, int]] = None,
                min_points: int = DEFAULT_MIN_PTS) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """
    Process single *_lightcurve.csv → (X_i, y_i, meta_i).
    If label_map is provided: controls whether we create positive windows.
      - label==1: build positive + negative windows
      - label==0: build negative-only
    """
    rng = np.random.default_rng(seed + worker_id * 1_000_003 + (hash(csv_path.name) & 0xffff))
    try:
        df = pd.read_csv(csv_path)
        if not {"time", "flux"}.issubset(df.columns):
            return None
        t = df["time"].to_numpy(float)
        f = df["flux"].to_numpy(float)
        m = np.isfinite(t) & np.isfinite(f) & (f != 0)
        t, f = t[m], f[m]
        if len(f) < min_points:
            return None

        f = robust_detrend(f)
        f = f / np.nanmedian(f)

        got = bls_best(t, f)
        if got is None:
            return None
        period, duration, depth, snr, t0 = got

        tic = csv_path.stem.split('_')[0]
        label_for_tic = None
        if label_map is not None:
            label_for_tic = label_map.get(tic, 0)

        Xs: List[np.ndarray] = []
        ys: List[int] = []
        meta_rows: List[Dict] = []

        # Build POS windows only if tic labeled 1 or label_map is None (unsupervised windowing)
        build_pos = (label_map is None) or (label_for_tic == 1)

        if build_pos:
            # 3 centers: exact transit + 2 jittered copies near phase 0
            pos_centers = [0.0, float(rng.uniform(-pos_jitter, pos_jitter)), float(rng.uniform(-pos_jitter, pos_jitter))]
            for c in pos_centers:
                arr = phase_window(rng, t, f, period, t0, center_phase=c % 1.0, width=pos_width, seq_len=seq_len)
                if arr is not None:
                    Xs.append(arr); ys.append(1)
                    meta_rows.append({
                        "tic_id": tic, "period": period, "duration": duration, "depth": depth,
                        "t0": t0, "bls_power": snr, "label": 1
                    })

        # Negatives far from transit
        n_negs = neg_per_pos if build_pos else max(neg_per_pos, 3)  # ensure some negatives even for label==0
        for _ in range(n_negs):
            off = float(rng.uniform(neg_min_sep, 1.0 - neg_min_sep))
            arr = phase_window(rng, t, f, period, t0, center_phase=off, width=pos_width, seq_len=seq_len)
            if arr is not None:
                Xs.append(arr); ys.append(0)
                meta_rows.append({
                    "tic_id": tic, "period": period, "duration": duration, "depth": depth,
                    "t0": t0, "bls_power": snr, "label": 0
                })

        if not Xs:
            return None

        return np.stack(Xs), np.array(ys, dtype=np.int64), pd.DataFrame(meta_rows)
    except Exception:
        return None

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Parallel window builder with manifest integration and chunked saving.")
    ap.add_argument("--processed_dir", type=str, required=True, help="Folder of *_lightcurve.csv files")
    ap.add_argument("--out_dir",      type=str, required=True, help="Output folder for X.npy, y.npy, meta.csv")
    ap.add_argument("--seq_len",      type=int, default=DEFAULT_SEQ_LEN)
    ap.add_argument("--neg_per_pos",  type=int, default=DEFAULT_NEG_PER)
    ap.add_argument("--pos_width",    type=float, default=DEFAULT_POS_WIDTH)
    ap.add_argument("--pos_jitter",   type=float, default=DEFAULT_POS_JITT)
    ap.add_argument("--neg_min_sep",  type=float, default=DEFAULT_NEG_SEP)
    ap.add_argument("--min_points",   type=int, default=DEFAULT_MIN_PTS)
    ap.add_argument("--n_jobs",       type=int, default=DEFAULT_JOBS, help="-1→all logical cores; 1→sequential")
    ap.add_argument("--seed",         type=int, default=DEFAULT_SEED)
    ap.add_argument("--chunk_size",   type=int, default=DEFAULT_CHUNK, help="Write to disk every N examples")
    ap.add_argument("--manifest",     type=str, default="", help="Optional manifest.csv with tic_id + label")
    ap.add_argument("--use_manifest_labels", type=int, default=DEFAULT_USE_MAN, help="1=honor manifest labels; 0=ignore")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(processed.glob("*_lightcurve.csv"))
    if not files:
        raise SystemExit(f"No *_lightcurve.csv under {processed}")

    # Build label map from manifest if provided
    label_map = None
    if args.manifest and int(args.use_manifest_labels) == 1:
        man_path = Path(args.manifest)
        if man_path.exists():
            man = pd.read_csv(man_path)
            if "tic_id" in man.columns:
                man["tic_id"] = man["tic_id"].astype(str)
                lab_col = next((c for c in ["label","labels","y","target","is_planet"] if c in man.columns), None)
                if lab_col:
                    tmp = man[["tic_id", lab_col]].copy()
                    tmp["label"] = tmp[lab_col].apply(safe_label_to_int)
                    label_map = dict(zip(tmp["tic_id"].tolist(), tmp["label"].tolist()))

    n_jobs = os.cpu_count() if args.n_jobs in (-1, 0) else max(1, args.n_jobs)
    print(f"- PROCESSED: {processed}")
    print(f"- OUTDIR:    {out_dir}")
    print(f"- Files:     {len(files)}  | seq_len={args.seq_len} | neg/pos={args.neg_per_pos} | jobs={n_jobs}")
    if label_map is not None:
        print(f"- Manifest:  using labels for {len(label_map)} TICs")

    # Prepare memmaps (we don't know final N upfront; we will grow in chunks)
    X_path = out_dir / "X.npy"
    y_path = out_dir / "y.npy"
    meta_path = out_dir / "meta.csv"

    # We'll accumulate in RAM until chunk_size, then append to disk memmaps.
    X_buf: List[np.ndarray] = []
    y_buf: List[np.ndarray] = []
    m_buf: List[pd.DataFrame] = []

    total = 0
    start = time.time()

    def flush_buffers(total_written: int) -> int:
        """Append buffered samples to disk; return new total_written."""
        nonlocal X_buf, y_buf, m_buf
        if not X_buf:
            return total_written
        X_cat = np.concatenate(X_buf, axis=0)  # (B, T) or (B, T, F)
        y_cat = np.concatenate(y_buf, axis=0)  # (B,)
        m_cat = pd.concat(m_buf, ignore_index=True)

        # If first write, create files; else append
        if total_written == 0:
            # Create new arrays
            np.save(X_path, X_cat)
            np.save(y_path, y_cat)
            m_cat.to_csv(meta_path, index=False)
        else:
            # Append by loading existing and concatenating (simple and safe at this scale)
            # If size gets huge later, switch to np.memmap and manual resizing.
            X_prev = np.load(X_path, mmap_mode=None)
            y_prev = np.load(y_path, mmap_mode=None)
            X_new = np.concatenate([X_prev, X_cat], axis=0)
            y_new = np.concatenate([y_prev, y_cat], axis=0)
            np.save(X_path, X_new)
            np.save(y_path, y_new)
            # append CSV
            m_cat.to_csv(meta_path, mode="a", header=False, index=False)

        total_written += y_cat.shape[0]
        X_buf, y_buf, m_buf = [], [], []
        return total_written

    # Run workers
    if n_jobs == 1:
        for i, p in enumerate(files, 1):
            out = process_one(p, args.seq_len, args.neg_per_pos, args.pos_width,
                              args.pos_jitter, args.neg_min_sep,
                              args.seed, worker_id=0,
                              label_map=label_map, min_points=args.min_points)
            if out is None:
                if i % 5 == 0 or i == len(files):
                    elapsed = time.time() - start
                    print(f"[skip] {p.name}   ({i}/{len(files)})  elapsed={elapsed:.1f}s")
                continue
            Xi, yi, mi = out
            X_buf.append(Xi); y_buf.append(yi); m_buf.append(mi)
            total += yi.shape[0]

            if total % args.chunk_size < yi.shape[0]:
                written_so_far = int(np.load(y_path).shape[0]) if y_path.exists() else 0
                written_so_far = flush_buffers(written_so_far)
                eta = (time.time() - start) / max(1, i) * (len(files) - i)
                print(f"[ok]   {p.name}: +{yi.shape[0]} ex  ({i}/{len(files)})  total={written_so_far}  ETA~{eta:.0f}s")
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
            futs = []
            for wid, p in enumerate(files):
                futs.append(ex.submit(
                    process_one, p, args.seq_len, args.neg_per_pos, args.pos_width,
                    args.pos_jitter, args.neg_min_sep, args.seed, wid,
                    label_map, args.min_points
                ))
            done = 0
            for fut, p in zip(as_completed(futs), files):
                res = fut.result()
                done += 1
                if res is None:
                    if done % 5 == 0 or done == len(files):
                        elapsed = time.time() - start
                        print(f"[skip] {p.name}   ({done}/{len(files)})  elapsed={elapsed:.1f}s")
                    continue
                Xi, yi, mi = res
                X_buf.append(Xi); y_buf.append(yi); m_buf.append(mi)
                total += yi.shape[0]

                # Flush when buffer crosses chunk boundary
                written_so_far = int(np.load(y_path).shape[0]) if y_path.exists() else 0
                if (written_so_far + sum(arr.shape[0] for arr in y_buf)) >= written_so_far + args.chunk_size:
                    written_so_far = flush_buffers(written_so_far)
                    eta = (time.time() - start) / max(1, done) * (len(files) - done)
                    print(f"[ok]   {p.name}: +{yi.shape[0]} ex  ({done}/{len(files)})  total={written_so_far}  ETA~{eta:.0f}s")

    # Final flush
    written = int(np.load(y_path).shape[0]) if y_path.exists() else 0
    written = flush_buffers(written)

    # Summary
    y_all = np.load(y_path, mmap_mode="r")
    pos = int((y_all == 1).sum()); neg = int((y_all == 0).sum())
    X_all = np.load(X_path, mmap_mode="r")
    elapsed = time.time() - start
    print("Saved:")
    print(" ", X_path)
    print(" ", y_path)
    print(" ", meta_path)
    print(f"Shapes: X={X_all.shape}  y={y_all.shape}  | pos={pos} neg={neg} | elapsed={elapsed:.1f}s")

if __name__ == "__main__":
    mp.freeze_support()
    main()
