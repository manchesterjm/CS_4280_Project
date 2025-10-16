# C:\CS_4280_Project\Code\build_windows_v2.py
from pathlib import Path
import numpy as np, pandas as pd
from astropy.timeseries import BoxLeastSquares

ROOT = Path(r"C:\CS_4280_Project")
PROCESSED = ROOT / "Planet_LightCurve_Data" / "processed"
OUTDIR = ROOT / "Code" / "data" / "windows"
OUTDIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 2048
NEG_PER_POS = 3          # negatives per positive
POS_WIDTH = 0.28         # phase width of each window
NEG_MIN_SEP = 0.12       # min phase distance from transit center for negatives
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---- detrend + scale ---------------------------------------------------------
def rolling_median(x, win):
    n = len(x)
    if n < 10: return x
    w = min(win, max(5, (n // 5) * 2 + 1))
    pad = w // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    mov = np.convolve(padded, np.ones(w) / w, mode="valid")
    return mov

def robust_detrend(flux):
    baseline = rolling_median(flux, 401)
    if np.any(baseline == 0):
        baseline = np.where(baseline == 0, np.nanmedian(flux), baseline)
    f = flux / baseline
    return f

def zscore(arr):
    mu = np.nanmean(arr); sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0: sd = 1.0
    return (arr - mu) / sd

# ---- BLS (fast, safe) --------------------------------------------------------
def bls_best(time, flux):
    """Use autopower with safe durations & period bounds."""
    mask = np.isfinite(time) & np.isfinite(flux)
    t, f = time[mask], flux[mask]
    if len(t) < 500:
        return None

    tspan = t.max() - t.min()
    if not np.isfinite(tspan) or tspan <= 0:
        return None

    # Period search range: at least a few cadences up to ~span
    min_period = 0.30   # days; keep > max(duration)
    max_period = float(min(30.0, max(2.0, 0.9 * tspan)))

    if max_period <= min_period:
        return None

    # Durations in days (must be strictly < min_period); 0.02–0.10 d = 0.5–2.4 h
    durations_days = np.array([0.02, 0.04, 0.06, 0.08, 0.10], dtype=float)
    durations_days = durations_days[durations_days < min_period * 0.95]
    if durations_days.size == 0:
        return None

    bls = BoxLeastSquares(t, f)
    # Use 'snr' objective; frequency_factor controls density (1.0 is fine)
    res = bls.autopower(durations_days,
                        minimum_period=min_period,
                        maximum_period=max_period,
                        objective='snr',
                        frequency_factor=1.0)

    i = int(np.nanargmax(res.power))
    # results fields are arrays; extract scalars
    period = float(res.period[i])
    duration = float(res.duration[i])
    depth = float(getattr(res, "depth", np.array([np.nan]))[i])
    snr = float(res.power[i])
    return period, duration, depth, snr

# ---- windowing ---------------------------------------------------------------
def phase_window(time, flux, period, center, width):
    phase = (time % period) / period
    lo, hi = (center - width/2) % 1.0, (center + width/2) % 1.0
    sel = (phase >= lo) & (phase <= hi) if lo < hi else ((phase >= lo) | (phase <= hi))
    f = flux[sel]
    if len(f) == 0:
        return None
    # pad/downsample to SEQ_LEN, then z-score per window
    if len(f) < SEQ_LEN:
        idx = rng.choice(len(f), SEQ_LEN, replace=True)
    else:
        idx = np.linspace(0, len(f) - 1, SEQ_LEN).astype(int)
    w = f[idx].astype(np.float32)
    return zscore(w).astype(np.float32)

def process_one(csv_path: Path):
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

    bp = bls_best(t, f)
    if bp is None:
        return None
    period, duration, depth, snr = bp

    pos = phase_window(t, f, period, center=0.0, width=POS_WIDTH)
    if pos is None:
        return None
    Xs = [pos]; ys = [1]

    # negatives well away from transit center
    for _ in range(NEG_PER_POS):
        off = float(rng.uniform(NEG_MIN_SEP, 1.0 - NEG_MIN_SEP))
        neg = phase_window(t, f, period, center=off, width=POS_WIDTH)
        if neg is not None:
            Xs.append(neg); ys.append(0)

    meta_rows = [{
        "tic_id": csv_path.stem.split('_')[0],
        "period": period, "duration": duration, "depth": depth,
        "bls_power": snr, "label": int(l)
    } for l in ys]

    return np.stack(Xs), np.array(ys, dtype=np.int64), pd.DataFrame(meta_rows)

def main():
    Xs, Ys, Ms = [], [], []
    files = sorted(PROCESSED.glob("*_lightcurve.csv"))
    for p in files:
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

    if not Xs:
        raise SystemExit("no examples built")

    X = np.concatenate(Xs, 0)
    y = np.concatenate(Ys, 0)
    meta = pd.concat(Ms, ignore_index=True)

    np.save(OUTDIR / "X.npy", X)
    np.save(OUTDIR / "y.npy", y)
    meta.to_csv(OUTDIR / "meta.csv", index=False)

    print("Saved:", OUTDIR / "X.npy", OUTDIR / "y.npy", OUTDIR / "meta.csv")
    print("Shapes:", X.shape, y.shape, "| pos:", int((y == 1).sum()), "neg:", int((y == 0).sum()))

if __name__ == "__main__":
    main()
