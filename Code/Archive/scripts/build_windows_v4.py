from pathlib import Path
import numpy as np, pandas as pd
from astropy.timeseries import BoxLeastSquares

ROOT = Path(r"C:\CS_4280_Project")
PROCESSED = ROOT / "Planet_LightCurve_Data" / "processed"
OUTDIR = ROOT / "Code" / "data" / "windows"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 2048
NEG_PER_POS = 5          # tougher classification
POS_WIDTH = 0.26         # slightly narrower than v3
POS_JITTER = 0.05        # <-- new: jitter positives ±5% phase around t0
NEG_MIN_SEP = 0.18       # keep negatives farther from transit
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

def rolling_median(x, win):
    n = len(x)
    if n < 10: return x
    w = min(win, max(5, (n // 5) * 2 + 1))
    pad = w // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    mov = np.convolve(padded, np.ones(w) / w, mode="valid")
    return mov

def robust_detrend(flux):
    base = rolling_median(flux, 401)
    base = np.where(base == 0, np.nanmedian(flux), base)
    return flux / base

def zscore(arr):
    mu = np.nanmean(arr); sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0: sd = 1.0
    return (arr - mu) / sd

def bls_best(time, flux):
    mask = np.isfinite(time) & np.isfinite(flux)
    t, f = time[mask], flux[mask]
    if len(t) < 500: return None
    tspan = t.max() - t.min()
    if not np.isfinite(tspan) or tspan <= 0: return None
    min_period = 0.30
    max_period = float(min(30.0, max(2.0, 0.9 * tspan)))
    if max_period <= min_period: return None
    durations = np.array([0.02, 0.04, 0.06, 0.08, 0.10], dtype=float)
    durations = durations[durations < min_period * 0.95]
    if durations.size == 0: return None
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
    if len(f) == 0: return None
    if len(f) < SEQ_LEN:
        idx = rng.choice(len(f), SEQ_LEN, replace=True)
    else:
        idx = np.linspace(0, len(f)-1, SEQ_LEN).astype(int)
    return zscore(f[idx].astype(np.float32))

def process_one(csv_path: Path):
    df = pd.read_csv(csv_path)
    if not {"time","flux"}.issubset(df.columns): return None
    t = df["time"].to_numpy(float)
    f = df["flux"].to_numpy(float)
    m = np.isfinite(t) & np.isfinite(f) & (f != 0)
    t, f = t[m], f[m]
    if len(f) < 500: return None
    f = robust_detrend(f); f = f / np.nanmedian(f)

    got = bls_best(t, f)
    if got is None: return None
    period, duration, depth, snr, t0 = got

    # positive windows: jitter around phase 0
    pos_centers = [0.0] + list(rng.uniform(-POS_JITTER, POS_JITTER, size=2))
    Xs, ys = [], []
    for c in pos_centers:
        pos = phase_window(t, f, period, t0, center_phase=c % 1.0, width=POS_WIDTH)
        if pos is not None:
            Xs.append(pos); ys.append(1)

    # negatives far from transit center
    for _ in range(NEG_PER_POS):
        off = float(rng.uniform(NEG_MIN_SEP, 1.0 - NEG_MIN_SEP))
        neg = phase_window(t, f, period, t0, center_phase=off, width=POS_WIDTH)
        if neg is not None:
            Xs.append(neg); ys.append(0)

    if not Xs: return None
    tic = csv_path.stem.split('_')[0]
    meta_rows = [{
        "tic_id": tic, "period": period, "duration": duration, "depth": depth,
        "t0": t0, "bls_power": snr, "label": int(l)
    } for l in ys]
    return np.stack(Xs), np.array(ys, dtype=np.int64), pd.DataFrame(meta_rows)

def main():
    Xs, Ys, Ms = [], [], []
    for p in sorted(PROCESSED.glob("*_lightcurve.csv")):
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
    if not Xs: raise SystemExit("no examples built")
    X = np.concatenate(Xs, 0); y = np.concatenate(Ys, 0)
    meta = pd.concat(Ms, ignore_index=True)
    np.save(OUTDIR/"X.npy", X); np.save(OUTDIR/"y.npy", y)
    meta.to_csv(OUTDIR/"meta.csv", index=False)
    print("Saved:", OUTDIR/"X.npy", OUTDIR/"y.npy", OUTDIR/"meta.csv")
    print("Shapes:", X.shape, y.shape, "| pos:", int((y==1).sum()), "neg:", int((y==0).sum()))

if __name__ == "__main__":
    main()
