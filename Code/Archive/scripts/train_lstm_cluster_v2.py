# C:\CS_4280_Project\Code\train_lstm_cluster_v2.py
from pathlib import Path
import numpy as np, pandas as pd, math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(r"C:\CS_4280_Project")
WIN_DIR = ROOT/"Code"/"data"/"windows"
DATA_DIR = ROOT/"Code"/"data"
MODEL_DIR = ROOT/"Code"/"models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 2048
BATCH = 64
EPOCHS = 150
LR = 5e-4
PATIENCE = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(123)

# ---------- load ----------
X = np.load(WIN_DIR/"X.npy").astype(np.float32)  # (N, L)
y = np.load(WIN_DIR/"y.npy").astype(np.int64)
meta = pd.read_csv(WIN_DIR/"meta.csv")
clusters = pd.read_csv(DATA_DIR/"clusters.csv")
meta["tic_id"] = meta["tic_id"].astype(str)
clusters["tic_id"] = clusters["tic_id"].astype(str)

# ensure cluster_index
if "cluster_index" not in clusters.columns:
    labels = clusters["cluster_id"].astype(int).tolist()
    uniq = sorted(set(labels) - {-1})
    remap = {c:i for i,c in enumerate(uniq)}
    clusters["cluster_index"] = clusters["cluster_id"].map(lambda c: remap.get(int(c), -1)).astype(int)

# merge row-level cluster
rowc = meta.merge(clusters[["tic_id","cluster_index"]], on="tic_id", how="left").fillna(-1)
row_cluster = rowc["cluster_index"].astype(int).to_numpy()

# star-level table (3 rows per star)
stars = meta["tic_id"].astype(str).groupby(meta.index // 4).first().to_numpy()  # ~4 windows/star now
star_clusters = pd.DataFrame({"tic_id": stars}).merge(
    clusters[["tic_id","cluster_index"]], on="tic_id", how="left").fillna(-1)
star_clusters["cluster_index"] = star_clusters["cluster_index"].astype(int)

# remap clusters to embedding ids (noise -> last)
uniq_nonneg = sorted(set(int(c) for c in star_clusters["cluster_index"].tolist() if int(c)>=0))
remap_e = {c:i for i,c in enumerate(uniq_nonneg)}
K_EMB = len(uniq_nonneg) + 1
def map_to_emb_idx(arr):
    return torch.tensor([remap_e.get(int(v), K_EMB-1) for v in arr], dtype=torch.long)

# ---------- stratified split with tiny-cluster guard ----------
def split_by_cluster(df, train=0.7, val=0.15, seed=123):
    rng = np.random.default_rng(seed)
    train_ids=[]; val_ids=[]; test_ids=[]
    for c in sorted(df["cluster_index"].unique()):
        group = df[df["cluster_index"]==c]["tic_id"].to_numpy()
        rng.shuffle(group)
        n = len(group)
        if n==0: continue
        if n<4:
            # put small clusters only in train to stabilize splits
            train_ids += group.tolist()
            continue
        n_tr = int(math.floor(train*n))
        n_va = int(math.floor(val*n))
        train_ids += group[:n_tr].tolist()
        val_ids   += group[n_tr:n_tr+n_va].tolist()
        test_ids  += group[n_tr+n_va:].tolist()
    return set(train_ids), set(val_ids), set(test_ids)

train_ids, val_ids, test_ids = split_by_cluster(star_clusters)

tic_per_row = meta["tic_id"].astype(str).to_numpy()
idx_tr = np.where(np.isin(tic_per_row, list(train_ids)))[0]
idx_va = np.where(np.isin(tic_per_row, list(val_ids)))[0]
idx_te = np.where(np.isin(tic_per_row, list(test_ids)))[0]

def subset(idx):
    Xs = torch.from_numpy(X[idx][:, :, None])  # (N,L,1)
    ys = torch.from_numpy(y[idx])
    cs = row_cluster[idx]
    return Xs, ys, cs

Xtr, ytr, ctr = subset(idx_tr)
Xva, yva, cva = subset(idx_va)
Xte, yte, cte = subset(idx_te)

ctr_e = map_to_emb_idx(ctr)
cva_e = map_to_emb_idx(cva)
cte_e = map_to_emb_idx(cte)

class DS(Dataset):
    def __init__(self, X,y,c): self.X=X; self.y=y; self.c=c
    def __len__(self): return self.y.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i], self.c[i]

train_loader = DataLoader(DS(Xtr,ytr,ctr_e), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(DS(Xva,yva,cva_e), batch_size=BATCH)
test_loader  = DataLoader(DS(Xte,yte,cte_e), batch_size=BATCH)

# ---------- model: Conv1d -> BiGRU + cluster embedding ----------
class ConvBiGRU(nn.Module):
    def __init__(self, nfeat=1, embed_dim=8, K_emb=8, h=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(nfeat, 8, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(16, h, num_layers=1, bidirectional=True, batch_first=True)
        self.emb = nn.Embedding(K_emb, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(2*h + embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, c):
        # x: (B,L,1) -> (B,1,L)
        x = x.transpose(1,2)
        x = self.conv(x)          # (B,16,L/4)
        x = x.transpose(1,2)      # (B,L/4,16)
        x,_ = self.gru(x)         # (B,L/4,2h)
        x = x[:, -1, :]           # (B,2h)
        e = self.emb(c)           # (B,embed_dim)
        z = torch.cat([x,e], dim=1)
        return self.fc(z).squeeze(1)

model = ConvBiGRU(nfeat=1, embed_dim=8, K_emb=K_EMB, h=64).to(DEVICE)

# focal loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha=alpha; self.gamma=gamma; self.eps=eps
    def forward(self, p, y):
        y = y.float()
        p = torch.clamp(p, self.eps, 1.0-self.eps)
        ce = - (y*torch.log(p) + (1-y)*torch.log(1-p))
        pt = y*p + (1-y)*(1-p)
        w = self.alpha*(y) + (1-self.alpha)*(1-y)
        return (w * (1-pt).pow(self.gamma) * ce).mean()

opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = FocalLoss(alpha=0.5, gamma=2.0)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

def evaluate(m, loader):
    m.eval(); Ys, Ps = [], []
    with torch.no_grad():
        for xb, yb, cb in loader:
            xb=xb.to(DEVICE); yb=yb.to(DEVICE).float(); cb=cb.to(DEVICE)
            p = m(xb, cb)
            Ps.append(p.detach().cpu().numpy()); Ys.append(yb.cpu().numpy())
    y_true = np.concatenate(Ys); y_score = np.concatenate(Ps)
    ap = average_precision_score(y_true, y_score)
    try: roc = roc_auc_score(y_true, y_score)
    except: roc = float("nan")
    return ap, roc

best_ap, patience = -1.0, PATIENCE
for ep in range(1, EPOCHS+1):
    model.train(); losses=[]
    for xb, yb, cb in train_loader:
        xb=xb.to(DEVICE); yb=yb.to(DEVICE).float(); cb=cb.to(DEVICE)
        p = model(xb, cb)
        loss = crit(p, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    val_ap, val_roc = evaluate(model, val_loader)
    sched.step(1.0 - val_ap)
    print(f"Epoch {ep:03d} | loss {np.mean(losses):.4f} | val AP {val_ap:.4f} | val ROC {val_roc:.4f}")
    if val_ap > best_ap:
        best_ap = val_ap; patience = PATIENCE
        torch.save(model.state_dict(), MODEL_DIR/"exo_conv_bigru_cluster.pt")
    else:
        patience -= 1
        if patience <= 0:
            print("Early stopping."); break

# test
state = torch.load(MODEL_DIR/"exo_conv_bigru_cluster.pt", map_location=DEVICE)
model.load_state_dict(state)
test_ap, test_roc = evaluate(model, test_loader)
print(f"TEST AP {test_ap:.4f} | ROC {test_roc:.4f}")
with open(MODEL_DIR/"metrics_cluster_v2.txt","w") as f:
    f.write(f"TEST_AP={test_ap:.6f}\nTEST_ROC={test_roc:.6f}\n")
