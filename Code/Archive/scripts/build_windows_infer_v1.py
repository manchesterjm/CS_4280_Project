# -*- coding: utf-8 -*-
"""
Build windows for the *test* dataset so we can run trained RNN inference.
Same logic style as build_windows_v4.py, but:
- PROCESSED points to C:\CS_4280_Project\test_dataset\simulated_dataset\processed
- OUTDIR is C:\CS_4280_Project\Code\data\windows_infer
- Writes X.npy, y.npy (y will be 0/1 if labels exist; otherwise all 0), meta.csv (with tic_id, period, duration, depth, t0, bls_power, label)
"""

from pathlib import Path
import numpy as np, pandas as pd
from astropy.timeseries import BoxLeastSquares

# -------- Paths (adjust if you move folders) ----------
ROOT = Path(r"C:\CS_4280_Project")
PROCESSED = ROOT / "test_dataset" / "simulated_dataset" / "processed"
OUTDIR = ROOT / "Code" / "data" / "windows_infer"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -------- Window params (match v4 style) -------------
SEQ_LEN = 2048            # source window length; model will re-pad/trim to 512 at inference
NEG_PER_POS = 5           # negatives per positive
POS_WIDTH = 0.26
POS_JITTER = 0.05
NEG_MIN_SEP = 0.18
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# -------- Helpers (copied from v4 style) -------------
def zscore(x: np.ndarray) -> np.ndarray:
    m = np.isfinite(x)
    if m.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)
    med = np.nanmedian(x[m])
    mad = np.nanmedian(np.abs(x[m] - med)) + 1e-8
    return ((x - med) / (1.4826 * mad)).astype(np.float32)

def load_curve(csv_path: Path):
    df = pd.read_csv(csv_path)
    # expect columns time, flux
    t = df["time"].to_numpy(dtype=float)
    f = df["flux"].to_numpy(dtype=float)
    # basic cleaning
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    if t.size < 10:
        return None
    # detrend (simple polyfit of order 2)
    tt = np.arange(t.size, dtype=float)
    mask = np.isfinite(f)
    if mask.sum() >= 3:
        coeffs = np.polyfit(tt[mask], f[mask], deg=2)
        trend = np.polyval(coeffs, tt)
        f = f - trend
    return t, f

def run_bls(time: np.ndarray, flux: np.ndarray):
    t, f = time, flux
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

def phase_window(time, flux, period, t0, center_phase, width):
    phase = ((time - t0) % period) / period
    lo, hi = (center_phase - width/2) % 1.0, (center_phase + width/2) % 1.0
    sel = (phase >= lo) & (phase <= hi) if lo < hi else ((phase >= lo) | (phase <= hi))
    f = flux[sel]
    if len(f) == 0:
        return None
    if len(f) < SEQ_LEN:
        idx = rng.choice(len(f), SEQ_LEN, replace=True)
    else:
        idx = np.linspace(0, len(f)-1, SEQ_LEN).astype(int)
    return zscore(f[idx].astype(np.float32))

def process_one(csv_path: Path):
    lc = load_curve(csv_path)
    if lc is None:
        return None
    t, f = lc
    bls = run_bls(t, f)
    if bls is None:
        return None
    period, duration, depth, snr, t0 = bls

    Xs, ys = [], []

    # positives near transit center (phase ~ 0), add jitter
    for _ in range(1):  # one positive per TIC by default (matches earlier windows density)
        jitter = float(rng.uniform(-POS_JITTER, POS_JITTER))
        pos = phase_window(t, f, period, t0, center_phase=(0.0 + jitter) % 1.0, width=POS_WIDTH)
        if pos is not None:
            Xs.append(pos); ys.append(1)

    # negatives far from transit center
    for _ in range(NEG_PER_POS):
        off = float(rng.uniform(NEG_MIN_SEP, 1.0 - NEG_MIN_SEP))
        neg = phase_window(t, f, period, t0, center_phase=off, width=POS_WIDTH)
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

def main():
    Xs, Ys, Ms = [], [], []
    n_files = 0
    for p in sorted(PROCESSED.glob("*_lightcurve.csv")):
        n_files += 1
        try:
            out = process_one(p)
            if out is None:
                print(f"[skip] {p.name}")
                continue
            Xi, yi, mi = out
            Xs.append(Xi); Ys.append(yi); Ms.append(mi)
            print(f"[ok]   {p.name}: {len(yi)} examples")
        except Exception as e:
            print(f"[err]  {p.name}: {e}")

    if n_files == 0:
        raise SystemExit(f"No CSVs found under {PROCESSED}")

    if not Xs:
        raise SystemExit("No examples built (all skipped).")

    X = np.concatenate(Xs, 0); y = np.concatenate(Ys, 0)
    meta = pd.concat(Ms, ignore_index=True)

    np.save(OUTDIR / "X.npy", X)
    # If you truly have unlabeled data, you can comment out the next line; but keeping y helps debugging.
    np.save(OUTDIR / "y.npy", y)
    meta.to_csv(OUTDIR / "meta.csv", index=False)

    print("Saved:", OUTDIR / "X.npy", OUTDIR / "y.npy", OUTDIR / "meta.csv")
    print("Shapes:", X.shape, y.shape, "| pos:", int((y==1).sum()), "neg:", int((y==0).sum()))
    print(f"Unique TICs: {meta['tic_id'].nunique()} from {n_files} files")

if __name__ == "__main__":
    main()
