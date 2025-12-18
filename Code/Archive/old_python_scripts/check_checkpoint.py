import torch
import sys

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'runs/sector1_experiments/baseline_corrected_posweight/best.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"Val metrics: {ckpt.get('val_metrics', 'N/A')}")
