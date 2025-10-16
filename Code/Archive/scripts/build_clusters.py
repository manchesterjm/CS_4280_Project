# C:\CS_4280_Project\Code\build_clusters.py
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Optional (recommended) libs:
import umap  # from umap-learn
import hdbscan

ROOT = Path(r"C:\CS_4280_Project")
PROC = ROOT / "Planet_LightCurve_Data" / "processed"
WIN  = ROOT / "Code" / "data" / "windows"
OUTD = ROOT / "Code" / "data"
OUTD.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = OUTD / "features.csv"
CLUSTERS_CSV = OUTD / "clusters.csv"

SEQ_STATS_MIN_POINTS = 400
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def robust_detrend(flux, win=401):
    n = len(flux)
    if n < 10:
        return flux
    w = min(win, max(5, (n // 5) * 2 + 1))
    pad = w // 2
    padded = np.pad(flux, (pad, pad), mode="edge")
    mov = np.convolve(padded, np.ones(w) / w, mode="valid")
    med = np.nanmedian(flux) if np.all(mov == 0) else mov
    return flux / med

def acf_peak(x, max_lag=200):
    x = x - np.nanmean(x)
    if np.allclose(x.std(), 0):
        return 0.0, 0
    acf = np.correlate(x, x, mode="full")[len(x)-1:len(x)-1+max_lag+1]
    acf = acf / (acf[0] + 1e-12)
    # ignore lag 0, take best lag and value
    if len(acf) <= 1:
        return 0.0, 0
    lag = int(np.argmax(acf[1:]) + 1)
    return float(acf[lag]), lag

def load_meta_star_level():
    """Collapse your windows/meta.csv to one row per tic_id for BLS stats."""
    meta = pd.read_csv(WIN / "meta.csv")
    # same stats repeated 3 times per star; take the first occurrence
    g = meta.groupby("tic_id", as_index=False).first()
    return g[["tic_id", "period", "duration", "depth", "bls_power"]]

def compute_curve_stats(csv_path: Path):
    df = pd.read_csv(csv_path)
    if not {"time", "flux"}.issubset(df.columns):
        raise ValueError(f"{csv_path.name} missing time/flux")
    t = df["time"].to_numpy(float)
    f = df["flux"].to_numpy(float)
    m = np.isfinite(t) & np.isfinite(f)
    t, f = t[m], f[m]
    frac_missing = 1.0 - (m.sum() / max(len(m), 1))
    # treat zeros as missing, drop them
    m2 = f != 0
    t, f = t[m2], f[m2]
    if len(f) < SEQ_STATS_MIN_POINTS:
        return dict(frac_missing=1.0, rms=np.nan, mad=np.nan, acf_val=np.nan, acf_lag=0, cadence=np.nan)
    f = robust_detrend(f)
    f = f / np.nanmedian(f)
    # RMS, MAD
    rms = float(np.sqrt(np.nanmean((f - 1.0) ** 2)))
    mad = float(np.nanmedian(np.abs(f - np.nanmedian(f))))
    acv, alag = acf_peak(f)
    # cadence (days)
    dt = np.diff(t)
    cadence = float(np.nanmedian(dt)) if len(dt) > 0 else np.nan
    return dict(frac_missing=float(frac_missing), rms=rms, mad=mad, acf_val=acv, acf_lag=int(alag), cadence=cadence)

def build_features():
    bls = load_meta_star_level()  # tic_id, period, duration, depth, bls_power
    rows = []
    for i, tic in enumerate(bls["tic_id"].astype(str).tolist(), 1):
        csv_path = PROC / f"{tic}_lightcurve.csv"
        if not csv_path.exists():
            print(f"[warn] missing {csv_path.name}")
            continue
        stats = compute_curve_stats(csv_path)
        rows.append({
            "tic_id": tic,
            **stats
        })
        if i % 20 == 0:
            print(f"[feat] {i}/{len(bls)}")
    feats = pd.DataFrame(rows)
    # join BLS fields
    df = bls.astype({"tic_id": str}).merge(feats.astype({"tic_id": str}), on="tic_id", how="left")
    # fill NAs cautiously
    for c in ["rms", "mad", "acf_val", "acf_lag", "cadence"]:
        if c in df:
            df[c] = df[c].fillna(df[c].median())
    df["frac_missing"] = df["frac_missing"].fillna(1.0)
    df.to_csv(FEATURES_CSV, index=False)
    print(f"[save] features -> {FEATURES_CSV.resolve()}")
    return df

def cluster(df: pd.DataFrame):
    # pick feature columns
    feat_cols = ["period", "duration", "depth", "bls_power", "rms", "mad", "acf_val", "acf_lag", "cadence", "frac_missing"]
    X = df[feat_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    Xs = StandardScaler().fit_transform(X)

    # reduce
    reducer = umap.UMAP(n_components=5, n_neighbors=20, min_dist=0.1, random_state=RANDOM_SEED)
    Xr = reducer.fit_transform(Xs)

    # cluster
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=8, min_samples=5, prediction_data=False)
        labels = clusterer.fit_predict(Xr)  # -1 = noise
        algo = "hdbscan"
    except Exception as e:
        print(f"[warn] HDBSCAN failed ({e}); falling back to KMeans(k=6)")
        km = KMeans(n_clusters=6, random_state=RANDOM_SEED, n_init="auto")
        labels = km.fit_predict(Xr)
        algo = "kmeans"

    out = df[["tic_id"]].copy()
    out["cluster_id"] = labels.astype(int)
    out["algo"] = algo
    # Normalize cluster ids to 0..K-1 with -1 reserved if present
    unique = sorted(set(labels) - {-1})
    remap = {c: i for i, c in enumerate(unique)}
    out["cluster_index"] = out["cluster_id"].map(lambda c: remap[c] if c != -1 else -1)

    out.to_csv(CLUSTERS_CSV, index=False)
    print(f"[save] clusters -> {CLUSTERS_CSV.resolve()}")
    print(out["cluster_id"].value_counts(dropna=False).sort_index())
    return out

def main():
    feats = build_features()
    cluster(feats)

if __name__ == "__main__":
    main()
