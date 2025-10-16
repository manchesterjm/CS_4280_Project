from pathlib import Path
import json
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = Path(r"C:\CS_4280_Project")
WIN  = ROOT/"Code"/"data"/"windows"
MOD  = ROOT/"Code"/"models"
OUT  = ROOT/"Code"/"reports"; OUT.mkdir(parents=True, exist_ok=True)

X = np.load(WIN/"X.npy").astype(np.float32)
y = np.load(WIN/"y.npy").astype(np.int64)
meta = pd.read_csv(WIN/"meta.csv"); meta["tic_id"]=meta["tic_id"].astype(str)
tic = meta["tic_id"].to_numpy()

# exact test split
splits = json.loads((MOD/"splits.json").read_text())
test_ids = set(splits["test"])
idx_te = np.where(np.isin(tic, list(test_ids)))[0]

# reuse saved scores if present; otherwise re-score
sc_path, lb_path = MOD/"test_scores.npy", MOD/"test_labels.npy"
if sc_path.exists() and lb_path.exists():
    ps = np.load(sc_path)
    yt = np.load(lb_path)
else:
    import torch
    from torch import nn
    class ConvBiGRU(nn.Module):
        def __init__(self, nfeat=1, h=64):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(nfeat, 8, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(8, 16, 9, padding=4),    nn.ReLU(), nn.MaxPool1d(2),
            )
            self.gru = nn.GRU(16, h, batch_first=True, bidirectional=True)
            self.fc  = nn.Sequential(nn.Linear(2*h, 64), nn.ReLU(), nn.Dropout(0.35),
                                     nn.Linear(64,1), nn.Sigmoid())
        def forward(self, x):
            x = x.transpose(1,2); x = self.conv(x); x = x.transpose(1,2)
            x,_ = self.gru(x); x = x[:,-1,:]
            return self.fc(x).squeeze(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvBiGRU().to(device)
    state = torch.load(MOD/"exo_conv_bigru_v2b.pt", map_location=device, weights_only=True)
    model.load_state_dict(state); model.eval()
    Xt = torch.from_numpy(X[idx_te][:, :, None]).to(device)
    with torch.no_grad():
        ps = model(Xt).cpu().numpy()
    yt = y[idx_te]
    np.save(sc_path, ps); np.save(lb_path, yt)

prec, rec, thr = precision_recall_curve(yt, ps)
ap = float(average_precision_score(yt, ps))
df = pd.DataFrame({"threshold": np.r_[thr, 1.0], "precision": prec, "recall": rec})
df.to_csv(OUT/"pr_curve.csv", index=False)
print("AP:", ap, "| saved:", OUT/"pr_curve.csv")
