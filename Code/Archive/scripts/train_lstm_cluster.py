from pathlib import Path
import numpy as np
import pandas as pd
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(r"C:\CS_4280_Project")
WIN_DIR = ROOT / "Code" / "data" / "windows"
DATA_DIR = ROOT / "Code" / "data"
MODEL_DIR = ROOT / "Code" / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 2048
BATCH_SIZE = 64
EPOCHS = 120
LR = 5e-4
PATIENCE = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(123)

# ---- Load windows and meta ----
X = np.load(WIN_DIR / "X.npy").astype(np.float32)            # (N, seq_len)
y = np.load(WIN_DIR / "y.npy").astype(np.int64)              # (N,)
meta = pd.read_csv(WIN_DIR / "meta.csv")                     # rows align with X/y

# ---- Load clusters (one row per tic_id) ----
clusters = pd.read_csv(DATA_DIR / "clusters.csv")            # columns: tic_id, cluster_id, algo, cluster_index
clusters["tic_id"] = clusters["tic_id"].astype(str)
meta["tic_id"] = meta["tic_id"].astype(str)

# Ensure cluster_index exists; if not, derive from cluster_id
if "cluster_index" not in clusters.columns:
    # Map non-negative cluster_id to 0..K-1, keep -1 as noise
    labels = clusters["cluster_id"].astype(int).to_list()
    uniq = sorted(set(labels) - {-1})
    remap_tmp = {c: i for i, c in enumerate(uniq)}
    clusters["cluster_index"] = clusters["cluster_id"].map(lambda c: remap_tmp.get(int(c), -1)).astype(int)

# Build global mapping ONCE:
# non-negative clusters -> [0..K-1], noise (-1) -> K (extra bucket)
uniq_nonneg = sorted(set(int(c) for c in clusters["cluster_index"].tolist() if int(c) >= 0))
remap = {c: i for i, c in enumerate(uniq_nonneg)}
K_embed = len(uniq_nonneg) + 1  # +1 for noise bucket

def map_cluster_array(arr_np: np.ndarray) -> torch.Tensor:
    # arr_np contains original cluster_index values (may include -1)
    mapped = [remap.get(int(v), K_embed - 1) for v in arr_np]  # -1 -> last index
    return torch.tensor(mapped, dtype=torch.long)

# Map each example row to its cluster index
row_clusters = meta.merge(clusters[["tic_id","cluster_index"]], on="tic_id", how="left")
row_clusters["cluster_index"] = row_clusters["cluster_index"].fillna(-1).astype(int)
cluster_index = row_clusters["cluster_index"].to_numpy(dtype=int)  # (N,)

# Prepare star-level table (there are 3 rows per star from build_windows)
stars = meta["tic_id"].astype(str).groupby(meta.index // 3).first().to_numpy()
star_clusters = pd.DataFrame({"tic_id": stars})
star_clusters = star_clusters.merge(clusters[["tic_id","cluster_index"]], on="tic_id", how="left").fillna(-1)
star_clusters["cluster_index"] = star_clusters["cluster_index"].astype(int)

# ---- Stratified split by cluster ----
def stratified_star_split(df_star_clusters, train=0.7, val=0.15, seed=123):
    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = [], [], []
    for c in sorted(df_star_clusters["cluster_index"].unique()):
        group = df_star_clusters[df_star_clusters["cluster_index"] == c]["tic_id"].to_numpy()
        rng.shuffle(group)
        n = len(group)
        if n == 0:
            continue
        n_train = int(math.floor(train * n))
        n_val   = int(math.floor(val * n))
        train_ids += group[:n_train].tolist()
        val_ids   += group[n_train:n_train+n_val].tolist()
        test_ids  += group[n_train+n_val:].tolist()
    return set(train_ids), set(val_ids), set(test_ids)

train_ids, val_ids, test_ids = stratified_star_split(star_clusters)

# Assign examples to splits (by tic_id)
tic_per_row = meta["tic_id"].to_numpy(str)
idx_train = np.where(np.isin(tic_per_row, list(train_ids)))[0]
idx_val   = np.where(np.isin(tic_per_row, list(val_ids)))[0]
idx_test  = np.where(np.isin(tic_per_row, list(test_ids)))[0]

def subset(idx):
    Xs = torch.from_numpy(X[idx][:, :, None])   # (N, seq_len, 1)
    ys = torch.from_numpy(y[idx])
    cs = cluster_index[idx]
    return Xs, ys, cs

Xtr, ytr, ctr = subset(idx_train)
Xva, yva, cva = subset(idx_val)
Xte, yte, cte = subset(idx_test)

# Map clusters to embedding indices once per split
ctr_map = map_cluster_array(ctr)
cva_map = map_cluster_array(cva)
cte_map = map_cluster_array(cte)

# ---- Dataset / Loaders ----
class WinDS(Dataset):
    def __init__(self, X, y, c):
        self.X, self.y, self.c = X, y, c
    def __len__(self): return self.y.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i], self.c[i]

train_loader = DataLoader(WinDS(Xtr, ytr, ctr_map), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(WinDS(Xva, yva, cva_map), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(WinDS(Xte, yte, cte_map), batch_size=BATCH_SIZE, shuffle=False)

# ---- Model with cluster embedding ----
class BiLSTMWithCluster(nn.Module):
    def __init__(self, nfeat=1, h1=128, h2=64, embed_dim=8, K_embed=8):
        super().__init__()
        self.l1 = nn.LSTM(nfeat, h1, batch_first=True, bidirectional=True)
        self.do1 = nn.Dropout(0.3)
        self.l2 = nn.LSTM(2*h1, h2, batch_first=True, bidirectional=True)
        self.do2 = nn.Dropout(0.3)
        self.emb = nn.Embedding(K_embed, embed_dim)
        self.fc1 = nn.Linear(2*h2 + embed_dim, 64)
        self.do3 = nn.Dropout(0.3)
        self.out = nn.Linear(64, 1)
    def forward(self, x, cidx):
        x,_ = self.l1(x); x = self.do1(x)
        x,_ = self.l2(x); x = self.do2(x)
        x = x[:, -1, :]                 # (B, 2*h2)
        e = self.emb(cidx)              # (B, embed_dim)
        z = torch.cat([x, e], dim=1)
        z = torch.relu(self.fc1(z)); z = self.do3(z)
        return torch.sigmoid(self.out(z)).squeeze(1)

model = BiLSTMWithCluster(nfeat=1, h1=128, h2=64, embed_dim=8, K_embed=K_embed).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
bce = nn.BCELoss(reduction="none")
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4)

# class weights from train
pos = int((ytr.numpy() == 1).sum()); neg = int((ytr.numpy() == 0).sum())
w_pos = (len(ytr) / (2*pos)) if pos else 1.0
w_neg = (len(ytr) / (2*neg)) if neg else 1.0
cw = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=DEVICE)

def evaluate(m, loader):
    m.eval(); Ys, Ps = [], []
    with torch.no_grad():
        for xb, yb, cb in loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE).float()
            cb = cb.to(DEVICE)
            p = m(xb, cb)
            Ps.append(p.detach().cpu().numpy()); Ys.append(yb.cpu().numpy())
    y_true = np.concatenate(Ys); y_score = np.concatenate(Ps)
    ap = average_precision_score(y_true, y_score)
    try: roc = roc_auc_score(y_true, y_score)
    except: roc = float("nan")
    return ap, roc

best_ap, patience = -1.0, PATIENCE
for epoch in range(1, EPOCHS+1):
    model.train(); losses=[]
    for xb, yb, cb in train_loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE).float(); cb = cb.to(DEVICE)
        p = model(xb, cb)
        weights = torch.where(yb==1, cw[1], cw[0])
        loss = (bce(p, yb) * weights).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    val_ap, val_roc = evaluate(model, val_loader)
    sched.step(1.0 - val_ap)
    print(f"Epoch {epoch:03d} | loss {np.mean(losses):.4f} | val AP {val_ap:.4f} | val ROC {val_roc:.4f}")
    if val_ap > best_ap:
        best_ap = val_ap; patience = PATIENCE
        torch.save(model.state_dict(), MODEL_DIR / "exo_bilstm_cluster.pt")
    else:
        patience -= 1
        if patience <= 0:
            print("Early stopping.")
            break

# Final test
state = torch.load(MODEL_DIR / "exo_bilstm_cluster.pt", map_location=DEVICE)
model.load_state_dict(state)  # safe: this is a plain state_dict
test_ap, test_roc = evaluate(model, test_loader)
print(f"TEST AP {test_ap:.4f} | ROC {test_roc:.4f}")
with open(MODEL_DIR / "metrics_cluster.txt", "w") as f:
    f.write(f"TEST_AP={test_ap:.6f}\nTEST_ROC={test_roc:.6f}\n")
