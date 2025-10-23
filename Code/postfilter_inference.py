# -*- coding: utf-8 -*-
"""
Post-filter RNN inference to reduce false positives (non-planets).

Inputs:
  - Code/reports/inference_scores.csv     (per-window scores/preds)
  - Code/data/windows_infer/meta.csv      (window metadata with tic_id, period, duration, depth, bls_power)
  - (optional) test_dataset/.../manifest.csv   (labels for metrics)

Outputs:
  - Code/reports/inference_aggregated_post.csv   (per-TIC after filters)
  - Code/reports/postfilter_summary.txt          (counts + metrics if labels provided)
  - Code/reports/false_positives_post.csv        (if labels provided)
  - Code/reports/false_negatives_post.csv        (if labels provided)

Usage (examples):
  # conservative but still practical defaults
  python postfilter_inference.py ^
    --scores "C:\CS_4280_Project\Code\reports\inference_scores.csv" ^
    --meta   "C:\CS_4280_Project\Code\data\windows_infer\meta.csv" ^
    --manifest "C:\CS_4280_Project\test_dataset\simulated_dataset\manifest.csv" ^
    --base_thr 0.53 --high_thr 0.70 --min_high 2 --min_bls 6.0 --dur_min 0.02 --dur_max 0.10 --depth_min 0.0005 --depth_max 0.05

  # stricter (fewer FPs, more TNR; may drop some recall)
  python postfilter_inference.py --base_thr 0.60 --high_thr 0.75 --min_high 3 --min_bls 7.5
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Post-filter RNN inference to prune non-planets.")
    ap.add_argument("--scores",   type=str, required=True, help="Path to inference_scores.csv")
    ap.add_argument("--meta",     type=str, required=True, help="Path to meta.csv used for inference windows")
    ap.add_argument("--manifest", type=str, default="", help="Optional manifest with labels for metrics")
    # filtering knobs
    ap.add_argument("--base_thr", type=float, default=0.53, help="Minimum TIC max score to even consider (e.g., F1 threshold)")
    ap.add_argument("--high_thr", type=float, default=0.70, help="Count windows above this 'high' score")
    ap.add_argument("--min_high", type=int,   default=2,    help="Require at least this many high-scoring windows per TIC")
    ap.add_argument("--min_bls",  type=float, default=6.0,  help="Minimum BLS power for the TIC")
    ap.add_argument("--dur_min",  type=float, default=0.02, help="Minimum BLS duration (days)")
    ap.add_argument("--dur_max",  type=float, default=0.10, help="Maximum BLS duration (days)")
    ap.add_argument("--depth_min",type=float, default=0.0005, help="Minimum BLS depth (fractional, e.g., 0.001 = 1000 ppm)")
    ap.add_argument("--depth_max",type=float, default=0.05,   help="Maximum BLS depth (e.g., 5%)")
    return ap.parse_args()

def main():
    args = parse_args()
    scores_path = Path(args.scores)
    meta_path   = Path(args.meta)
    manifest_path = Path(args.manifest) if args.manifest else None

    assert scores_path.exists(), f"Missing {scores_path}"
    assert meta_path.exists(),   f"Missing {meta_path}"

    win = pd.read_csv(scores_path)  # columns: includes tic_id, score, pred
    meta = pd.read_csv(meta_path)   # columns: tic_id, period, duration, depth, t0, bls_power, label
    if "tic_id" not in win.columns:
        raise SystemExit("inference_scores.csv must contain 'tic_id' column.")
    # ensure string ids
    win["tic_id"] = win["tic_id"].astype(str)
    meta["tic_id"] = meta["tic_id"].astype(str)

    # Per-TIC aggregate info from meta (same values repeated per TIC)
    agg_meta = (meta.groupby("tic_id")
                    .agg(period=("period", "median"),
                         duration=("duration", "median"),
                         depth=("depth", "median"),
                         bls_power=("bls_power", "median"))
                    .reset_index())

    # Per-TIC score aggregates
    g = win.groupby("tic_id")
    tic = pd.DataFrame({
        "tic_id": g.size().index.astype(str),
        "n_windows": g.size().values,
        "score_max": g["score"].max().values,
        "score_mean": g["score"].mean().values,
        "n_high": (g["score"].apply(lambda s: (s >= args.high_thr).sum())).values,
    })

    # Merge aggregates
    tic = tic.merge(agg_meta, on="tic_id", how="left")

    # Base pass: must exceed base_thr on max score
    tic["pass_base"] = tic["score_max"] >= args.base_thr
    # Consistency: must have at least min_high windows above high_thr
    tic["pass_consistency"] = tic["n_high"] >= args.min_high
    # BLS / physics gates
    tic["pass_bls"] = tic["bls_power"] >= args.min_bls
    tic["pass_dur"] = (tic["duration"] >= args.dur_min) & (tic["duration"] <= args.dur_max)
    tic["pass_depth"] = (tic["depth"] >= args.depth_min) & (tic["depth"] <= args.depth_max)

    # Final decision: AND of all gates
    gates = ["pass_base", "pass_consistency", "pass_bls", "pass_dur", "pass_depth"]
    tic["pred_post"] = tic[gates].all(axis=1).astype(int)

    # Save aggregated post-filtered predictions
    out_dir = Path(r"C:\CS_4280_Project\Code\reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_agg = out_dir / "inference_aggregated_post.csv"
    tic.to_csv(out_agg, index=False)

    # Print summary
    n_pos = int((tic["pred_post"]==1).sum())
    summary_lines = []
    summary_lines.append(f"TICs predicted planet after filters: {n_pos} / {len(tic)}")
    summary_lines.append(f"Gates: base_thr={args.base_thr}, high_thr={args.high_thr}, min_high={args.min_high}, "
                         f"min_bls={args.min_bls}, dur=[{args.dur_min},{args.dur_max}], depth=[{args.depth_min},{args.depth_max}]")

    # If labels provided, compute metrics
    if manifest_path and manifest_path.exists():
        man = pd.read_csv(manifest_path)
        if "tic_id" in man.columns:
            man["tic_id"] = man["tic_id"].astype(str)
            lab_col = None
            for c in ["label","labels","y","target","is_planet"]:
                if c in man.columns:
                    lab_col = c; break
            if lab_col is not None:
                def to01(x):
                    try: return int(x)
                    except Exception:
                        s = str(x).strip().lower()
                        if s in ["1","true","planet","pos","positive","yes","y"]: return 1
                        return 0
                labs = man[["tic_id", lab_col]].copy()
                labs["label_true"] = labs[lab_col].map(to01)
                labs = labs.drop(columns=[lab_col])

                m = tic.merge(labs, on="tic_id", how="left")
                m["label_true"] = m["label_true"].fillna(0).astype(int)
                y_true = m["label_true"].to_numpy()
                y_pred = m["pred_post"].to_numpy()

                from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
                cm = confusion_matrix(y_true, y_pred, labels=[0,1])
                tn, fp, fn, tp = cm.ravel()
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

                summary_lines.append(f"TP={tp} FP={fp} TN={tn} FN={fn}  |  Precision={prec:.3f} Recall={rec:.3f} F1={f1:.3f}")

                # Save FP/FN lists for review
                fp_rows = m[(m["label_true"]==0) & (m["pred_post"]==1)].sort_values("score_max", ascending=False)
                fn_rows = m[(m["label_true"]==1) & (m["pred_post"]==0)].sort_values("score_max", ascending=False)
                fp_rows.to_csv(out_dir / "false_positives_post.csv", index=False)
                fn_rows.to_csv(out_dir / "false_negatives_post.csv", index=False)

    out_txt = out_dir / "postfilter_summary.txt"
    out_txt.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))
    print("Saved:")
    print(" ", out_agg)
    print(" ", out_txt)
    if manifest_path and manifest_path.exists():
        print(" ", out_dir / "false_positives_post.csv")
        print(" ", out_dir / "false_negatives_post.csv")

if __name__ == "__main__":
    main()
