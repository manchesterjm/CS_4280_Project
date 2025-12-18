import torch
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "runs/sector1_incremental/best.pt"
ckpt = torch.load(path, map_location="cpu")

print(f"Checkpoint: {path}")
print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")

m = ckpt.get('val_metrics', {})
print(f"  AUC: {m.get('auc', 0):.4f}")
print(f"  F1: {m.get('f1', 0):.4f}")
print(f"  Recall: {m.get('recall', 0):.4f}")
print(f"  Precision: {m.get('precision', 0):.4f}")
