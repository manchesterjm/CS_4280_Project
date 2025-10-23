# -*- coding: utf-8 -*-
"""
Grid-sweep over post-filter settings to trade precision vs recall.
Reads:
  - Code/reports/inference_scores.csv
  - Code/data/windows_infer/meta.csv
  - test_dataset/.../manifest.csv  (labels)
Writes:
  - Code/reports/postfilter_sweep_results.csv
"""

from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(r"C:\CS_4280_Project")
CODE = ROOT / "Code"
SCORES = CODE / "reports" / "inference_scores.csv"
META   = CODE / "data" / "windows_infer" / "meta.csv"
MAN    = ROOT / "test_dataset" / "simulated_dataset" / "manifest.csv"
OUT    = CODE / "reports" / "postfilter_sweep_results.csv"

def to01(x):
    try: return int(x)
    except Exception:
        s = str(x).strip().lower()
        return 1 if s in ["1","true","planet","pos","positive","yes","y"] else 0

# --- load
win = pd.read_csv(SCORES)
meta = pd.read_csv(META)
man  = pd.read_csv(MAN)

win["tic_id"] = win["tic_id"].astype(str)
meta["tic_id"] = meta["tic_id"].astype(str)
man["tic_id"]  = man["tic_id"].astype(str)
lab_col = next((c for c in ["label","labels","y","target","is_planet"] if c in man.columns), None)
if lab_col is None:
    raise SystemExit("No label column found in manifest.csv")

labs = man[["tic_id", lab_col]].copy()
labs["label_true"] = labs[lab_col].map(to01)
labs = labs[["tic_id","label_true"]]

# TIC aggregates
g = win.groupby("tic_id")
tic = pd.DataFrame({
    "tic_id": g.size().index.astype(str),
    "n_windows": g.size().values,
    "score_max": g["score"].max().values,
    "score_mean": g["score"].mean().values,
    "n_ge_070": (g["score"].apply(lambda s: (s >= 0.70).sum())).values,
    "n_ge_075": (g["score"].apply(lambda s: (s >= 0.75).sum())).values,
})
meta_agg = (meta.groupby("tic_id")
                .agg(period=("period","median"),
                     duration=("duration","median"),
                     depth=("depth","median"),
                     bls_power=("bls_power","median"))
                .reset_index())
tic = tic.merge(meta_agg, on="tic_id", how="left").merge(labs, on="tic_id", how="left")
tic["label_true"] = tic["label_true"].fillna(0).astype(int)

# parameter grid (start focused; expand if needed)
base_thrs = [0.60, 0.65, 0.70, 0.75]
high_thrs = [0.70, 0.75]
min_highs = [0, 1, 2]   # 0 = no consistency gate
use_bls   = [0, 1]      # 0 to start; 1 later (NaN tolerant)
min_bls   = [4.0, 5.0]
nan_pass  = True

rows = []
for base_thr, high_thr, min_high, ub, mb in itertools.product(base_thrs, high_thrs, min_highs, use_bls, min_bls):
    # gates (NaN tolerant)
    m = tic.copy()
    m["pred"] = True
    # base
    m["pred"] &= (m["score_max"] >= base_thr)
    # consistency
    if min_high > 0:
        count_col = "n_ge_075" if abs(high_thr - 0.75) < 1e-6 else "n_ge_070"
        m["pred"] &= (m[count_col] >= min_high)
    # bls (optional)
    if ub == 1:
        cond = (m["bls_power"] >= mb)
        if nan_pass:
            cond = cond | m["bls_power"].isna()
        m["pred"] &= cond

    y_true = m["label_true"].to_numpy()
    y_pred = m["pred"].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    rows.append({
        "base_thr": base_thr, "high_thr": high_thr, "min_high": min_high,
        "use_bls": ub, "min_bls": mb,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": prec, "Recall": rec, "F1": f1,
        "kept": int(m["pred"].sum())
    })

res = pd.DataFrame(rows).sort_values(["F1","Precision","Recall"], ascending=[False,False,False])
OUT.parent.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT, index=False)
print("Saved:", OUT)
print(res.head(10).to_string(index=False))
