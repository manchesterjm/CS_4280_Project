from pathlib import Path
import numpy as np, pandas as pd, math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(r"C:\CS_4280_Project")
WIN = ROOT/"Code"/"data"/"windows"
DATA = ROOT/"Code"/"data"
MODEL_DIR = ROOT/"Code"/"models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH, EPOCHS, LR, PATIENCE = 64, 150, 5e-4, 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

X = np.load(WIN/"X.npy").astype(np.float32)
y = np.load(WIN/"y.npy").astype(np.int64)
meta = pd.read_csv(WIN/"meta.csv")
clusters = pd.read_csv(DATA/"clusters.csv")
meta["tic_id"] = meta["tic_id"].astype(str); clusters["tic_id"] = clusters["tic_id"].astype(str)

# ensure cluster_index
if "cluster_index" not in clusters.columns:
    labels = clusters["cluster_id"].astype(int).tolist()
    uniq = sorted(set(labels) - {-1})
    rem = {c:i for i,c in enumerate(uniq)}
    clusters["cluster_index"] = clusters["cluster_id"].map(lambda c: rem.get(int(c), -1)).astype(int)

rowc = meta.merge(clusters[["tic_id","cluster_index"]], on="tic_id", how="left").fillna(-1)
row_cluster = rowc["cluster_index"].astype(int).to_numpy()

# star-level frame
stars = np.unique(meta["tic_id"].to_numpy())
star_clusters = pd.DataFrame({"tic_id": stars}).merge(
    clusters[["tic_id","cluster_index"]], on="tic_id", how="left").fillna(-1)
star_clusters["cluster_index"] = star_clusters["cluster_index"].astype(int)

# embed mapping (noise -> last)
uniq_nonneg = sorted(set(int(c) for c in star_clusters["cluster_index"] if int(c)>=0))
rem_e = {c:i for i,c in enumerate(uniq_nonneg)}
K_EMB = len(uniq_nonneg) + 1
def to_emb(a): return torch.tensor([rem_e.get(int(v), K_EMB-1) for v in a], dtype=torch.long)

# stratified split by cluster (guard tiny clusters)
def split_by_cluster(df, train=0.7, val=0.15, seed=123):
    rng = np.random.default_rng(seed)
    tr=[]; va=[]; te=[]
    for c in sorted(df["cluster_index"].unique()):
        g = df[df["cluster_index"]==c]["tic_id"].to_numpy()
        rng.shuffle(g); n=len(g)
        if n==0: continue
        if n<4: tr += g.tolist(); continue
        ntr=int(math.floor(train*n)); nva=int(math.floor(val*n))
        tr += g[:ntr].tolist(); va += g[ntr:ntr+nva].tolist(); te += g[ntr+nva:].tolist()
    return set(tr), set(va), set(te)

train_ids, val_ids, test_ids = split_by_cluster(star_clusters)

tic = meta["tic_id"].to_numpy()
idx_tr = np.where(np.isin(tic, list(train_ids)))[0]
idx_va = np.where(np.isin(tic, list(val_ids)))[0]
idx_te = np.where(np.isin(tic, list(test_ids)))[0]

def subset(idx):
    return (torch.from_numpy(X[idx][:, :, None]),
            torch.from_numpy(y[idx]),
            row_cluster[idx])

Xtr, ytr, ctr = subset(idx_tr)
Xva, yva, cva = subset(idx_va)
Xte, yte, cte = subset(idx_te)

ctr_e, cva_e, cte_e = to_emb(ctr), to_emb(cva), to_emb(cte)

class DS(torch.utils.data.Dataset):
    def __init__(self,X,y,c): self.X=X; self.y=y; self.c=c
    def __len__(self): return self.y.shape[0]
    def __getitem__(self,i): return self.X[i], self.y[i], self.c[i]

train_loader = DataLoader(DS(Xtr,ytr,ctr_e), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(DS(Xva,yva,cva_e), batch_size=BATCH)
test_loader  = DataLoader(DS(Xte,yte,cte_e), batch_size=BATCH)

class ConvBiGRUCluster(nn.Module):
    def __init__(self, nfeat=1, h=64, K_emb=8, e=8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(nfeat, 8, 9, padding=4), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8,16, 9, padding=4),     nn.ReLU(), nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(16, h, batch_first=True, bidirectional=True)
        self.emb = nn.Embedding(K_emb, e)
        self.fc  = nn.Sequential(nn.Linear(2*h+e,64), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(64,1), nn.Sigmoid())
    def forward(self, x, c):
        x = x.transpose(1,2)
        x = self.conv(x)
        x = x.transpose(1,2)
        x,_ = self.gru(x)
        x = x[:,-1,:]
        e = self.emb(c)
        z = torch.cat([x,e], dim=1)
        return self.fc(z).squeeze(1)

model = ConvBiGRUCluster(K_emb=K_EMB).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, eps=1e-7):
        super().__init__(); self.a=alpha; self.g=gamma; self.eps=eps
    def forward(self, p, y):
        y=y.float(); p=torch.clamp(p,self.eps,1-self.eps)
        ce=-(y*torch.log(p)+(1-y)*torch.log(1-p)); pt=y*p+(1-y)*(1-p)
        w=self.a*y+(1-self.a)*(1-y)
        return (w*((1-pt)**self.g)*ce).mean()

crit = FocalLoss()
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

def evaluate(m, loader):
    m.eval(); Y=[]; P=[]
    with torch.no_grad():
        for xb,yb,cb in loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE).float(); cb=cb.to(DEVICE)
            p=model(xb,cb)
            P.append(p.cpu().numpy()); Y.append(yb.cpu().numpy())
    y_true=np.concatenate(Y); y_score=np.concatenate(P)
    ap=average_precision_score(y_true,y_score)
    try: roc=roc_auc_score(y_true,y_score)
    except: roc=float("nan")
    return ap, roc

best, patience = -1.0, PATIENCE
for ep in range(1,EPOCHS+1):
    model.train(); losses=[]
    for xb,yb,cb in train_loader:
        xb=xb.to(DEVICE); yb=yb.to(DEVICE).float(); cb=cb.to(DEVICE)
        p=model(xb,cb); loss=crit(p,yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    ap, roc = evaluate(model, val_loader); sched.step(1.0-ap)
    print(f"Epoch {ep:03d} | loss {np.mean(losses):.4f} | val AP {ap:.4f} | val ROC {roc:.4f}")
    if ap>best: best=ap; patience=PATIENCE; torch.save(model.state_dict(), MODEL_DIR/"exo_conv_bigru_cluster_v3.pt")
    else:
        patience-=1
        if patience<=0: print("Early stopping."); break

state=torch.load(MODEL_DIR/"exo_conv_bigru_cluster_v3.pt", map_location=DEVICE); model.load_state_dict(state)
tap, troc = evaluate(model, test_loader)
print(f"TEST AP {tap:.4f} | ROC {troc:.4f}")
with open(MODEL_DIR/"metrics_cluster_v3.txt","w") as f: f.write(f"TEST_AP={tap:.6f}\nTEST_ROC={troc:.6f}\n")
