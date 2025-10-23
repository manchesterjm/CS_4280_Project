# -*- coding: utf-8 -*-
"""
Post-filter RNN inference to reduce false positives (non-planets), with NaN-tolerant gates and diagnostics.

Inputs:
  - Code/reports/inference_scores.csv     (per-window scores/preds; must include tic_id, score)
  - Code/data/windows_infer/meta.csv      (per-window meta: tic_id, period, duration, depth, bls_power, etc.)
  - (optional) manifest.csv with labels   (tic_id + label column) for metrics

Outputs:
  - Code/reports/inference_aggregated_post.csv
  - Code/reports/postfilter_summary.txt
  - (if labels) false_positives_post.csv / false_negatives_post.csv
  - Code/reports/postfilter_gate_passrates.csv  (diagnostics)
  - Code/reports/postfilter_gate_diagnostics.csv (per-TIC details)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="NaN-tolerant post-filter for RNN exoplanet inference.")
    ap.add_argument("--scores",   type=str, required=True, help="Path to inference_scores.csv")
    ap.add_argument("--meta",     type=str, required=True, help="Path to meta.csv used for inference windows")
    ap.add_argument("--manifest", type=str, default="", help="Optional manifest with labels for metrics")

    # thresholds / ranges
    ap.add_argument("--base_thr", type=float, default=0.53, help="Min TIC max score to consider")
    ap.add_argument("--high_thr", type=float, default=0.70, help="Score considered 'high' for consistency")
    ap.add_argument("--min_high", type=int,   default=1,    help="Min # of high-scoring windows per TIC")
    ap.add_argument("--min_bls",  type=float, default=4.0,  help="Minimum BLS power")
    ap.add_argument("--dur_min",  type=float, default=0.01, help="Min transit duration (days)")
    ap.add_argument("--dur_max",  type=float, default=0.15, help="Max transit duration (days)")
    ap.add_argument("--depth_min",type=float, default=0.0002, help="Min depth (fraction)")
    ap.add_argument("--depth_max",type=float, default=0.10,   help="Max depth (fraction)")

    # gate toggles
    ap.add_argument("--use_base", type=int, default=1, help="Use base score gate (1/0)")
    ap.add_argument("--use_consistency", type=int, default=1, help="Use consistency gate (1/0)")
    ap.add_argument("--use_bls",  type=int, default=1, help="Use BLS power gate (1/0)")
    ap.add_argument("--use_dur",  type=int, default=1, help="Use duration gate (1/0)")
    ap.add_argument("--use_depth",type=int, default=1, help="Use depth gate (1/0)")

    # NaN handling
    ap.add_argument("--nan_pass", type=int, default=1, help="Treat NaN in meta fields as pass (1) or fail (0)")
    return ap.parse_args()

def nan_ok(cond, nan_mask, nan_pass=True):
    if nan_pass:
        # where value is NaN, override to True
        return cond | nan_mask
    return cond & (~nan_mask)

def main():
    args = parse_args()
    scores_path = Path(args.scores)
    meta_path   = Path(args.meta)
    manifest_path = Path(args.manifest) if args.manifest else None

    assert scores_path.exists(), f"Missing {scores_path}"
    assert meta_path.exists(),   f"Missing {meta_path}"

    win = pd.read_csv(scores_path)
    meta = pd.read_csv(meta_path)

    if "tic_id" not in win.columns or "score" not in win.columns:
        raise SystemExit("inference_scores.csv must have 'tic_id' and 'score' columns.")

    win["tic_id"] = win["tic_id"].astype(str)
    meta["tic_id"] = meta["tic_id"].astype(str)

    # Per-TIC aggregates
    g = win.groupby("tic_id")
    tic = pd.DataFrame({
        "tic_id": g.size().index.astype(str),
        "n_windows": g.size().values,
        "score_max": g["score"].max().values,
        "score_mean": g["score"].mean().values,
        "n_high": (g["score"].apply(lambda s: (s >= args.high_thr).sum())).values,
    })

    # Meta aggregates (median per TIC)
    meta_agg = (meta.groupby("tic_id")
                    .agg(period=("period","median"),
                         duration=("duration","median"),
                         depth=("depth","median"),
                         bls_power=("bls_power","median"))
                    .reset_index())
    tic = tic.merge(meta_agg, on="tic_id", how="left")

    # Build gates (NaN-tolerant)
    gates = {}
    if args.use_base:
        gates["pass_base"] = tic["score_max"] >= args.base_thr
    if args.use_consistency:
        gates["pass_consistency"] = tic["n_high"] >= args.min_high
    if args.use_bls:
        bls_nan = tic["bls_power"].isna()
        cond = tic["bls_power"] >= args.min_bls
        gates["pass_bls"] = nan_ok(cond, bls_nan, nan_pass=bool(args.nan_pass))
    if args.use_dur:
        dur_nan = tic["duration"].isna()
        cond = (tic["duration"] >= args.dur_min) & (tic["duration"] <= args.dur_max)
        gates["pass_dur"] = nan_ok(cond, dur_nan, nan_pass=bool(args.nan_pass))
    if args.use_depth:
        depth_nan = tic["depth"].isna()
        cond = (tic["depth"] >= args.depth_min) & (tic["depth"] <= args.depth_max)
        gates["pass_depth"] = nan_ok(cond, depth_nan, nan_pass=bool(args.nan_pass))

    for k, v in gates.items():
        tic[k] = v.astype(bool)

    # Final decision = AND of all active gates; if no gates active, everything passes
    active = list(gates.keys())
    tic["pred_post"] = True
    for k in active:
        tic["pred_post"] = tic["pred_post"] & tic[k]

    # Diagnostics: per-gate pass rates
    passrates = []
    for k in active:
        passrates.append({"gate": k, "rate": float(tic[k].mean())})
    pass_df = pd.DataFrame(passrates)

    # Save outputs
    out_dir = Path(r"C:\CS_4280_Project\Code\reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inference_aggregated_post.csv").write_text(tic.to_csv(index=False), encoding="utf-8")
    (out_dir / "postfilter_gate_passrates.csv").write_text(pass_df.to_csv(index=False), encoding="utf-8")
    (out_dir / "postfilter_gate_diagnostics.csv").write_text(tic.to_csv(index=False), encoding="utf-8")

    # Summary
    n_kept = int((tic["pred_post"]==1).sum())
    lines = []
    lines.append(f"TICs predicted planet after filters: {n_kept} / {len(tic)}")
    lines.append("Active gates: " + (", ".join(active) if active else "(none)"))
    for r in passrates:
        lines.append(f"{r['gate']}: pass_rate={r['rate']:.3f}")
    # If labels are provided, compute metrics
    if manifest_path and manifest_path.exists():
        man = pd.read_csv(manifest_path)
        if "tic_id" in man.columns:
            man["tic_id"] = man["tic_id"].astype(str)
            lab_col = next((c for c in ["label","labels","y","target","is_planet"] if c in man.columns), None)
            if lab_col:
                def to01(x):
                    try: return int(x)
                    except Exception:
                        s = str(x).strip().lower()
                        return 1 if s in ["1","true","planet","pos","positive","yes","y"] else 0
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

                # Save FP/FN lists
                m[(m["label_true"]==0) & (m["pred_post"]==1)].to_csv(out_dir / "false_positives_post.csv", index=False)
                m[(m["label_true"]==1) & (m["pred_post"]==0)].to_csv(out_dir / "false_negatives_post.csv", index=False)

    (out_dir / "postfilter_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("Saved:")
    print(" ", out_dir / "inference_aggregated_post.csv")
    print(" ", out_dir / "postfilter_summary.txt")
    print(" ", out_dir / "postfilter_gate_passrates.csv")
    print(" ", out_dir / "postfilter_gate_diagnostics.csv")

if __name__ == "__main__":
    main()
