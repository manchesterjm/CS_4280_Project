"""Evaluate Simple BiLSTM (no clustering) on any dataset."""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


class SimpleBiLSTM(nn.Module):
    """Simple BiLSTM without clustering"""
    def __init__(self, input_size=1, hidden_size=256, num_layers=4, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        lstm_out_size = hidden_size * 2
        self.fc1 = nn.Linear(lstm_out_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_fwd = h_n[-2, :, :]
        h_bwd = h_n[-1, :, :]
        h = torch.cat([h_fwd, h_bwd], dim=1)

        out = self.fc1(h)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        return out.squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] {device}")

    # Load checkpoint
    print(f"[loading model] {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    # Load data
    print(f"[loading data] {args.data_dir}")
    X = np.load(f"{args.data_dir}/X.npy")

    # Check for labels
    has_labels = True
    try:
        y = np.load(f"{args.data_dir}/y.npy")
        print(f"[data] X.shape={X.shape} y.shape={y.shape}")
        print(f"[data] pos={y.sum()} neg={len(y) - y.sum()}")
    except FileNotFoundError:
        has_labels = False
        y = None
        print(f"[data] X.shape={X.shape} (no labels)")

    # Load metadata if available
    try:
        meta = pd.read_csv(f"{args.data_dir}/meta.csv")
        if len(meta) != len(X):
            meta = meta.iloc[:len(X)]
    except FileNotFoundError:
        meta = pd.DataFrame({'idx': range(len(X))})

    # Build model
    model = SimpleBiLSTM(
        input_size=1,
        hidden_size=config.get('hidden', 256),
        num_layers=config.get('layers', 4),
        dropout=config.get('dropout', 0.3)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"[model] loaded, params={sum(p.numel() for p in model.parameters()):,}")

    # Run inference
    print("[inference] Running...")
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)

    batch_size = 128
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x_batch = X_tensor[i:i+batch_size].to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x_batch)
                probs = torch.sigmoid(logits)

            all_probs.extend(probs.cpu().numpy())

    probs = np.array(all_probs)
    preds = (probs >= 0.5).astype(int)

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)

    if has_labels:
        auc = roc_auc_score(y, probs)
        f1 = f1_score(y, preds)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        accuracy = accuracy_score(y, preds)

        print(f"  AUC:       {auc:.4f}")
        print(f"  F1:        {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")

        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        print(f"\n  Confusion Matrix:")
        print(f"    TP={tp:4d}  FP={fp:4d}")
        print(f"    FN={fn:4d}  TN={tn:4d}")
    else:
        print(f"  Total samples: {len(probs)}")
        print(f"  Predicted planets: {preds.sum()} ({100*preds.sum()/len(preds):.1f}%)")
        print(f"  Predicted non-planets: {len(preds) - preds.sum()}")
        print(f"\n  Probability distribution:")
        print(f"    Mean: {probs.mean():.4f}")
        print(f"    Median: {np.median(probs):.4f}")
        print(f"    Min: {probs.min():.4f}")
        print(f"    Max: {probs.max():.4f}")

    print("="*50)

    # Show top candidates
    top_k = min(20, len(probs))
    top_idx = np.argsort(probs)[::-1][:top_k]

    print(f"\n[top {top_k} candidates]")
    tic_col = 'tic_id' if 'tic_id' in meta.columns else meta.columns[0]
    print(f"{'Rank':>4} {'TIC ID':>12} {'Prob':>8}")
    print("-" * 30)

    for rank, idx in enumerate(top_idx, 1):
        tic = meta.iloc[idx].get(tic_col, idx)
        print(f"{rank:4d} {str(tic):>12} {probs[idx]:8.4f}")

    # Save predictions
    if args.output_file:
        results = meta.copy()
        results['probability'] = probs
        results['predicted'] = preds
        if has_labels:
            results['actual'] = y
        results.to_csv(args.output_file, index=False)
        print(f"\n[saved] {args.output_file}")


if __name__ == '__main__':
    main()
