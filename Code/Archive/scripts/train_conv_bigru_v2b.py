from pathlib import Path
import json
import numpy as np, pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(r"C:\CS_4280_Project")
WIN = ROOT/"Code"/"data"/"windows"
MODEL_DIR = ROOT/"Code"/"models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH, EPOCHS = 64, 200
LR, WD = 5e-4, 1e-3
PATIENCE = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X = np.load(WIN/"X.npy").astype(np.float32)
y = np.load(WIN/"y.npy").astype(np.int64)
meta = pd.read_csv(WIN/"meta.csv")
meta["tic_id"] = meta["tic_id"].astype(str)
tic = meta["tic_id"].to_numpy()

# ----- deterministic star-level split (saved to disk) -----
stars = np.unique(tic)
rng = np.random.default_rng(123); rng.shuffle(stars)
n = len(stars); n_tr = int(0.7*n); n_va = int(0.15*n)
train_ids = list(stars[:n_tr]); val_ids = list(stars[n_tr:n_tr+n_va]); test_ids = list(stars[n_tr+n_va:])

with open(MODEL_DIR/"splits.json", "w") as f:
    json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

idx_tr = np.where(np.isin(tic, train_ids))[0]
idx_va = np.where(np.isin(tic, val_ids))[0]
idx_te = np.where(np.isin(tic, test_ids))[0]

def subset(idx):
    Xs = torch.from_numpy(X[idx][:, :, None])
    ys = torch.from_numpy(y[idx])
    return Xs, ys

Xtr, ytr = subset(idx_tr); Xva, yva = subset(idx_va); Xte, yte = subset(idx_te)

class DS(Dataset):
    def __init__(self, X,y): self.X=X; self.y=y
    def __len__(self): return self.y.shape[0]
    def __getitem__(self,i): return self.X[i], self.y[i]

train_loader = DataLoader(DS(Xtr,ytr), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(DS(Xva,yva), batch_size=BATCH)
test_loader  = DataLoader(DS(Xte,yte), batch_size=BATCH)

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.55, gamma=2.0, eps=1e-7):
        super().__init__(); self.a=alpha; self.g=gamma; self.eps=eps
    def forward(self, p, y):
        y=y.float(); p=torch.clamp(p,self.eps,1-self.eps)
        ce=-(y*torch.log(p)+(1-y)*torch.log(1-p)); pt=y*p+(1-y)*(1-p)
        w=self.a*y+(1-self.a)*(1-y)
        return (w*((1-pt)**self.g)*ce).mean()

model=ConvBiGRU().to(DEVICE)
opt=torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40, eta_min=1e-5)
crit=FocalLoss()

def eval_metrics(m,loader):
    m.eval(); Y=[]; P=[]
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE).float()
            p=model(xb)
            P.append(p.cpu().numpy()); Y.append(yb.cpu().numpy())
    y_true=np.concatenate(Y); y_score=np.concatenate(P)
    ap=average_precision_score(y_true,y_score)
    try: roc=roc_auc_score(y_true,y_score)
    except: roc=float("nan")
    return ap, roc, y_true, y_score

best, patience = -1.0, PATIENCE
for ep in range(1, EPOCHS+1):
    model.train(); losses=[]
    for xb,yb in train_loader:
        xb=xb.to(DEVICE); yb=yb.to(DEVICE).float()
        p=model(xb); loss=crit(p,yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    sched.step()
    ap, roc, _, _ = eval_metrics(model, val_loader)
    print(f"Epoch {ep:03d} | loss {np.mean(losses):.4f} | val AP {ap:.4f} | val ROC {roc:.4f}")
    if ap>best:
        best=ap; patience=PATIENCE
        torch.save(model.state_dict(), MODEL_DIR/"exo_conv_bigru_v2b.pt")
    else:
        patience-=1
        if patience<=0:
            print("Early stopping."); break

state=torch.load(MODEL_DIR/"exo_conv_bigru_v2b.pt", map_location=DEVICE, weights_only=True)
model.load_state_dict(state)

tap, troc, y_true, y_score = eval_metrics(model, test_loader)
print(f"TEST AP {tap:.4f} | ROC {troc:.4f}")
np.save(MODEL_DIR/"test_labels.npy", y_true)
np.save(MODEL_DIR/"test_scores.npy", y_score)
with open(MODEL_DIR/"metrics_conv_bigru_v2b.txt","w") as f:
    f.write(f"TEST_AP={tap:.6f}\nTEST_ROC={troc:.6f}\n")
print("Saved:", MODEL_DIR/"splits.json", MODEL_DIR/"test_labels.npy", MODEL_DIR/"test_scores.npy")
