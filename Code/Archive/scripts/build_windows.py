# C:\CS_4280_Project\Code\build_windows.py
from pathlib import Path
import numpy as np, pandas as pd
from astropy.timeseries import BoxLeastSquares

# ---- paths (match your tree) ----
ROOT = Path(r"C:\CS_4280_Project")
PROCESSED = ROOT/"Planet_LightCurve_Data"/"processed"
OUTDIR = ROOT/"Code"/"data"/"windows"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- config ----
SEQ_LEN = 2048
NEG_PER_POS = 2
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

def robust_detrend(flux, win=401):
    # simple rolling median detrend, fall back if series is short
    n = len(flux)
    w = min(win, n//5*2+1) if n>=50 else max(5, n//5*2+1)
    if w < 5: return flux / np.nanmedian(flux)
    pad = w//2
    padded = np.pad(flux, (pad,pad), mode='edge')
    med = np.convolve(padded, np.ones(w)/w, mode='valid')
    med = np.where(med==0, np.nanmedian(flux), med)
    return flux / med

def bls_best_period(time, flux):
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    if len(time) < 300:
        return None
    # rough grid 0.5-30 d, one duration guess
    grid = np.linspace(0.5, 30.0, 2000)
    bls = BoxLeastSquares(time, flux)
    res = bls.power(grid, 0.1)
    i = np.nanargmax(res.power)
    return float(res.period[i]), float(res.duration[i]), float(res.depth[i]), float(res.power[i])

def phase_fold_window(time, flux, period, center=0.0, width=0.4):
    phase = (time % period) / period
    lo, hi = (center - width/2) % 1.0, (center + width/2) % 1.0
    sel = (phase >= lo) & (phase <= hi) if lo < hi else ((phase >= lo) | (phase <= hi))
    f = flux[sel]
    if len(f) == 0:
        return None
    # pad or downsample to SEQ_LEN
    if len(f) < SEQ_LEN:
        idx = rng.choice(len(f), SEQ_LEN, replace=True)
    else:
        idx = np.linspace(0, len(f)-1, SEQ_LEN).astype(int)
    return f[idx].astype(np.float32)

def process_file(csv_path: Path):
    df = pd.read_csv(csv_path)
    if not {'time','flux'}.issubset(df.columns):
        raise ValueError(f"{csv_path.name} missing 'time'/'flux' columns")
    # clean
    t = df['time'].to_numpy(dtype=float)
    f = df['flux'].to_numpy(dtype=float)
    # drop NaNs/zeros
    m = np.isfinite(t) & np.isfinite(f) & (f != 0)
    t, f = t[m], f[m]
    if len(f) < 400:
        return None
    # detrend + normalize
    f = robust_detrend(f)
    f = f / np.nanmedian(f)
    # period
    bp = bls_best_period(t, f)
    if bp is None:
        return None
    period, duration, depth, power = bp
    # windows
    pos = phase_fold_window(t, f, period, center=0.0, width=0.4)
    if pos is None: 
        return None
    windows = [pos]; labels = [1]
    for _ in range(NEG_PER_POS):
        off = float(rng.uniform(0.1, 0.9))
        neg = phase_fold_window(t, f, period, center=off, width=0.4)
        if neg is not None:
            windows.append(neg); labels.append(0)
    meta_rows = [{"tic_id": csv_path.stem.split('_')[0], "period": period, "duration": duration,
                  "depth": depth, "bls_power": power, "label": int(l)} for l in labels]
    return np.stack(windows, axis=0), np.array(labels, dtype=np.int64), pd.DataFrame(meta_rows)

def main():
    files = sorted(PROCESSED.glob("*_lightcurve.csv"))
    X_list, y_list, meta_list = [], [], []
    for p in files:
        try:
            out = process_file(p)
            if out is None:
                print(f"[skip] {p.name}: too short or BLS failed")
                continue
            X_i, y_i, m_i = out
            X_list.append(X_i); y_list.append(y_i); meta_list.append(m_i)
            print(f"[ok]   {p.name}: examples {len(y_i)}")
        except Exception as e:
            print(f"[err]  {p.name}: {e}")

    if not X_list:
        raise RuntimeError("No examples generated. Check input files/columns.")

    X = np.concatenate(X_list, axis=0)                    # (N, SEQ_LEN)
    y = np.concatenate(y_list, axis=0)                    # (N,)
    meta = pd.concat(meta_list, ignore_index=True)

    np.save(OUTDIR/"X.npy", X)
    np.save(OUTDIR/"y.npy", y)
    meta.to_csv(OUTDIR/"meta.csv", index=False)

    print("\nSaved:")
    print(" ", (OUTDIR/'X.npy').resolve())
    print(" ", (OUTDIR/'y.npy').resolve())
    print(" ", (OUTDIR/'meta.csv').resolve())
    print("Shapes:", X.shape, y.shape, "| positives:", int((y==1).sum()), "negatives:", int((y==0).sum()))

if __name__ == "__main__":
    main()
