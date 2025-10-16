# C:\CS_4280_Project\Code\train_lstm.py
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------- paths / params ----------------
ROOT = Path(r"C:\CS_4280_Project")
WIN_DIR = ROOT / "Code" / "data" / "windows"
MODEL_DIR = ROOT / "Code" / "models"; MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 2048
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3
PATIENCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(123)

# ---------------- data ----------------
X = np.load(WIN_DIR / "X.npy").astype(np.float32)            # (N, seq_len)
y = np.load(WIN_DIR / "y.npy").astype(np.int64)              # (N,)
meta = pd.read_csv(WIN_DIR / "meta.csv")                     # has tic_id, label, period, ...

# split by star (tic_id) to avoid leakage
stars = meta["tic_id"].astype(str).groupby(meta.index // 3).first().unique()  # 3 rows per star in build_windows
RNG.shuffle(stars)
n = len(stars)
n_train = int(0.7 * n); n_val = int(0.15 * n)
train_ids = set(stars[:n_train])
val_ids   = set(stars[n_train:n_train + n_val])
test_ids  = set(stars[n_train + n_val:])

# map example -> star id (each star contributed 3 rows in order)
star_id_per_row = meta["tic_id"].astype(str).to_numpy()
idx_train = np.where(np.isin(star_id_per_row, list(train_ids)))[0]
idx_val   = np.where(np.isin(star_id_per_row, list(val_ids)))[0]
idx_test  = np.where(np.isin(star_id_per_row, list(test_ids)))[0]

def subset(idx):
    Xs = torch.from_numpy(X[idx][:, :, None])   # (N, seq_len, 1)
    ys = torch.from_numpy(y[idx])
    return Xs, ys

Xtr, ytr = subset(idx_train)
Xva, yva = subset(idx_val)
Xte, yte = subset(idx_test)

class WinDS(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return self.y.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(WinDS(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(WinDS(Xva, yva), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(WinDS(Xte, yte), batch_size=BATCH_SIZE, shuffle=False)

# ---------------- model ----------------
class BiLSTM(nn.Module):
    def __init__(self, nfeat=1, h1=64, h2=32):
        super().__init__()
        self.l1 = nn.LSTM(nfeat, h1, batch_first=True, bidirectional=True)
        self.do1 = nn.Dropout(0.3)
        self.l2 = nn.LSTM(2*h1, h2, batch_first=True, bidirectional=True)
        self.do2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(2*h2, 64)
        self.do3 = nn.Dropout(0.3)
        self.out = nn.Linear(64, 1)
    def forward(self, x):
        x,_ = self.l1(x); x = self.do1(x)
        x,_ = self.l2(x); x = self.do2(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x)); x = self.do3(x)
        return torch.sigmoid(self.out(x)).squeeze(1)

model = BiLSTM().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
bce = nn.BCELoss(reduction="none")
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

# class weights (for imbalance)
pos = (ytr.numpy() == 1).sum(); neg = (ytr.numpy() == 0).sum()
w_pos = (len(ytr) / (2*pos)) if pos else 1.0
w_neg = (len(ytr) / (2*neg)) if neg else 1.0
cw = torch.tensor([w_neg, w_pos], dtype=torch.float32, device=DEVICE)

def eval_ap_roc(m, loader):
    m.eval(); Y, P = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE).float()
            p = m(xb)
            P.append(p.detach().cpu().numpy()); Y.append(yb.cpu().numpy())
    y_true = np.concatenate(Y); y_score = np.concatenate(P)
    ap = average_precision_score(y_true, y_score)
    try: roc = roc_auc_score(y_true, y_score)
    except: roc = float("nan")
    return ap, roc

best_ap, patience_left = -1.0, PATIENCE
for epoch in range(1, EPOCHS+1):
    model.train(); losses=[]
    for xb, yb in train_loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE).float()
        p = model(xb)
        weights = torch.where(yb==1, cw[1], cw[0])
        loss = (bce(p, yb) * weights).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    val_ap, val_roc = eval_ap_roc(model, val_loader)
    sched.step(1.0 - val_ap)
    print(f"Epoch {epoch:02d} | loss {np.mean(losses):.4f} | val AP {val_ap:.4f} | val ROC {val_roc:.4f}")
    if val_ap > best_ap:
        best_ap, patience_left = val_ap, PATIENCE
        torch.save(model.state_dict(), MODEL_DIR/"exo_bilstm.pt")
    else:
        patience_left -= 1
        if patience_left <= 0:
            print("Early stopping."); break

# ---- final test ----
model.load_state_dict(torch.load(MODEL_DIR/"exo_bilstm.pt", map_location=DEVICE))
test_ap, test_roc = eval_ap_roc(model, test_loader)
print(f"TEST AP {test_ap:.4f} | ROC {test_roc:.4f}")
with open(MODEL_DIR/"metrics.txt", "w") as f:
    f.write(f"TEST_AP={test_ap:.6f}\nTEST_ROC={test_roc:.6f}\n")
