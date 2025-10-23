# -*- coding: utf-8 -*-
"""
Post-filter RNN inference (NaN-tolerant) with count- *and* percentage-based consistency gates.

Inputs
  - Code/reports/inference_scores.csv  (per-window; needs tic_id, score)
  - Code/data/windows_infer/meta.csv   (per-window meta: tic_id, period, duration, depth, bls_power)
  - (optional) manifest.csv with labels (tic_id + label column)

Outputs
  - Code/reports/inference_aggregated_post.csv
  - Code/reports/postfilter_summary.txt
  - Code/reports/postfilter_gate_passrates.csv
  - Code/reports/postfilter_gate_diagnostics.csv
  - (if labels) false_positives_post.csv / false_negatives_post.csv
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Post-filter RNN outputs with count/percentage consistency gates.")
    ap.add_argument("--scores",   required=True, help="Path to inference_scores.csv")
    ap.add_argument("--meta",     required=True, help="Path to meta.csv used at inference")
    ap.add_argument("--manifest", default="",    help="Optional manifest with labels for metrics")

    # Base + consistency
    ap.add_argument("--base_thr", type=float, default=0.73, help="Min TIC score (e.g., score_mean or score_max) to consider")
    ap.add_argument("--high_thr", type=float, default=0.75, help="Per-window score considered 'high'")
    ap.add_argument("--min_high", type=int,   default=-1,   help="Require at least this many high windows per TIC; -1 disables")
    ap.add_argument("--min_high_frac", type=float, default=0.30,
                    help="Require at least this fraction of windows to be high; set <0 to disable")

    # Optional physics gates
    ap.add_argument("--use_bls", type=int, default=0)
    ap.add_argument("--min_bls", type=float, default=4.0)
    ap.add_argument("--use_dur", type=int, default=0)
    ap.add_argument("--dur_min", type=float, default=0.01)
    ap.add_argument("--dur_max", type=float, default=0.15)
    ap.add_argument("--use_depth", type=int, default=0)
    ap.add_argument("--depth_min", type=float, default=0.0002)
    ap.add_argument("--depth_max", type=float, default=0.10)

    # NaN handling & toggles
    ap.add_argument("--use_base", type=int, default=1)
    ap.add_argument("--use_consistency", type=int, default=1)
    ap.add_argument("--nan_pass", type=int, default=1, help="Treat NaN meta values as pass (1) or fail (0)")
    return ap.parse_args()

def nan_ok(condition: pd.Series, nan_mask: pd.Series, nan_pass: bool) -> pd.Series:
    return condition | nan_mask if nan_pass else (condition & ~nan_mask)

def to01(x):
    try: return int(x)
    except Exception:
        s = str(x).strip().lower()
        return 1 if s in ["1","true","planet","pos","positive","yes","y"] else 0

def main():
    args = parse_args()
    scores_path = Path(args.scores)
    meta_path   = Path(args.meta)
    manifest_path = Path(args.manifest) if args.manifest else None

    assert scores_path.exists(), f"Missing {scores_path}"
    assert meta_path.exists(),   f"Missing {meta_path}"

    win  = pd.read_csv(scores_path)
    meta = pd.read_csv(meta_path)
    for df in (win, meta):
        if "tic_id" not in df.columns:
            raise SystemExit("Both scores and meta must include 'tic_id'")
        df["tic_id"] = df["tic_id"].astype(str)

    # Per-TIC aggregates from window scores
    g = win.groupby("tic_id")
    tic = pd.DataFrame({
        "tic_id": g.size().index.astype(str),
        "n_windows": g.size().values,
        "score_max": g["score"].max().values,
        "score_mean": g["score"].mean().values,
        "n_high": (g["score"].apply(lambda s: (s >= args.high_thr).sum())).values,
    })
    tic["frac_high"] = tic["n_high"] / tic["n_windows"].clip(lower=1)

    # Meta aggregates
    meta_agg = (meta.groupby("tic_id")
                  .agg(period=("period","median"),
                       duration=("duration","median"),
                       depth=("depth","median"),
                       bls_power=("bls_power","median"))
                  .reset_index())
    tic = tic.merge(meta_agg, on="tic_id", how="left")

    # Build gates
    active_gates = []
    if args.use_base:
        # Prefer score_mean if present in aggregated inference; fall back to score_max
        score_col = "score_mean" if "score_mean" in tic.columns else "score_max"
        tic["pass_base"] = tic[score_col] >= args.base_thr
        active_gates.append("pass_base")

    if args.use_consistency:
        pass_cons = pd.Series(True, index=tic.index)
        if args.min_high >= 0:
            pass_cons &= (tic["n_high"] >= int(args.min_high))
        if args.min_high_frac >= 0:
            pass_cons &= (tic["frac_high"] >= float(args.min_high_frac))
        tic["pass_consistency"] = pass_cons
        active_gates.append("pass_consistency")

    if args.use_bls:
        cond = tic["bls_power"] >= args.min_bls
        tic["pass_bls"] = nan_ok(cond, tic["bls_power"].isna(), bool(args.nan_pass))
        active_gates.append("pass_bls")

    if args.use_dur:
        cond = (tic["duration"] >= args.dur_min) & (tic["duration"] <= args.dur_max)
        tic["pass_dur"] = nan_ok(cond, tic["duration"].isna(), bool(args.nan_pass))
        active_gates.append("pass_dur")

    if args.use_depth:
        cond = (tic["depth"] >= args.depth_min) & (tic["depth"] <= args.depth_max)
        tic["pass_depth"] = nan_ok(cond, tic["depth"].isna(), bool(args.nan_pass))
        active_gates.append("pass_depth")

    # Final decision
    tic["pred_post"] = True
    for k in active_gates:
        tic["pred_post"] &= tic[k]

    # Diagnostics
    passrates = pd.DataFrame({
        "gate": active_gates,
        "rate": [float(tic[k].mean()) for k in active_gates]
    })

    out_dir = Path(r"C:\CS_4280_Project\Code\reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inference_aggregated_post.csv").write_text(tic.to_csv(index=False), encoding="utf-8")
    (out_dir / "postfilter_gate_passrates.csv").write_text(passrates.to_csv(index=False), encoding="utf-8")
    (out_dir / "postfilter_gate_diagnostics.csv").write_text(tic.to_csv(index=False), encoding="utf-8")

    kept = int((tic["pred_post"]==1).sum())
    lines = [f"TICs predicted planet after filters: {kept} / {len(tic)}",
             "Active gates: " + (", ".join(active_gates) if active_gates else "(none)")]
    for _, r in passrates.iterrows():
        lines.append(f"{r['gate']}: pass_rate={r['rate']:.3f}")

    # If labels: compute metrics
    if manifest_path and manifest_path.exists():
        man = pd.read_csv(manifest_path)
        if "tic_id" in man.columns:
            man["tic_id"] = man["tic_id"].astype(str)
            lab_col = next((c for c in ["label","labels","y","target","is_planet"] if c in man.columns), None)
            if lab_col:
                labs = man[["tic_id", lab_col]].copy()
                labs["label_true"] = labs[lab_col].map(to01)
                m = tic.merge(labs[["tic_id","label_true"]], on="tic_id", how="left")
                m["label_true"] = m["label_true"].fillna(0).astype(int)
                y_true = m["label_true"].to_numpy()
                y_pred = m["pred_post"].astype(int).to_numpy()

                from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
                lines.append(f"TP={tp} FP={fp} TN={tn} FN={fn} | Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

                # FP/FN lists
                m[(m["label_true"]==0) & (m["pred_post"]==1)].to_csv(out_dir/"false_positives_post.csv", index=False)
                m[(m["label_true"]==1) & (m["pred_post"]==0)].to_csv(out_dir/"false_negatives_post.csv", index=False)

    (out_dir / "postfilter_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Saved:")
    print(" ", out_dir / "inference_aggregated_post.csv")
    print(" ", out_dir / "postfilter_summary.txt")
    print(" ", out_dir / "postfilter_gate_passrates.csv")
    print(" ", out_dir / "postfilter_gate_diagnostics.csv")

if __name__ == "__main__":
    main()
