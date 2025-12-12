"""Evaluate models on Kepler test data to measure cross-mission generalization."""
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import skew
import torch.nn as nn


class ClusterBiLSTMOriginal(nn.Module):
    """Original BiLSTM architecture (uses 'classifier' naming)."""

    def __init__(self, n_clusters, cluster_embed_dim, hidden, layers, dropout):
        super().__init__()
        self.cluster_embed = nn.Embedding(n_clusters, cluster_embed_dim)
        self.lstm = nn.LSTM(
            1, hidden, layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0
        )
        fc_in = hidden * 2 + cluster_embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x, cluster_ids):
        """Forward pass with flux windows and cluster IDs."""
        emb = self.cluster_embed(cluster_ids)
        _, (h, _) = self.lstm(x)
        h_fwd = h[-2]
        h_bwd = h[-1]
        combined = torch.cat([h_fwd, h_bwd, emb], dim=1)
        return self.classifier(combined).squeeze(-1)


class ClusterBiLSTMFinetune(nn.Module):
    """Fine-tuned BiLSTM architecture (uses 'fc' naming)."""

    def __init__(self, n_clusters, cluster_embed_dim, hidden, layers, dropout):
        super().__init__()
        self.cluster_embed = nn.Embedding(n_clusters, cluster_embed_dim)
        self.lstm = nn.LSTM(
            1, hidden, layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if layers > 1 else 0
        )
        fc_in = hidden * 2 + cluster_embed_dim
        self.fc = nn.Sequential(
            nn.Linear(fc_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x, cluster_ids):
        """Forward pass with flux windows and cluster IDs."""
        emb = self.cluster_embed(cluster_ids)
        _, (h, _) = self.lstm(x)
        h_fwd = h[-2]
        h_bwd = h[-1]
        combined = torch.cat([h_fwd, h_bwd, emb], dim=1)
        return self.fc(combined).squeeze(-1)


def compute_stats(flux_windows):
    """Compute statistical features for K-means clustering."""
    features = []
    for window in flux_windows:
        features.append([
            np.mean(window),
            np.std(window),
            np.var(window),
            skew(window),
            np.max(window) - np.min(window),
            np.median(window),
            np.median(np.abs(window - np.median(window))),
            np.max(window) - np.min(window)
        ])
    return np.array(features)


def evaluate_model(model_path, name, flux_data, labels, features, device,
                   use_original_arch=False):
    """Evaluate a model on the given data."""
    print(f'\n{"=" * 50}')
    print(f'{name}')
    print(f'{"=" * 50}')

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt['config']

    # Choose architecture based on model type
    if use_original_arch:
        model = ClusterBiLSTMOriginal(
            n_clusters=config['n_clusters'],
            cluster_embed_dim=config['cluster_embed_dim'],
            hidden=config['hidden'],
            layers=config['layers'],
            dropout=config['dropout']
        ).to(device)
    else:
        model = ClusterBiLSTMFinetune(
            n_clusters=config['n_clusters'],
            cluster_embed_dim=config['cluster_embed_dim'],
            hidden=config['hidden'],
            layers=config['layers'],
            dropout=config['dropout']
        ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Get expected feature dimension from saved scaler
    n_features_expected = len(ckpt['scaler_params']['mean'])

    if features.shape[1] == n_features_expected:
        # Features match, use saved scaler and kmeans
        scaler = StandardScaler()
        scaler.mean_ = np.array(ckpt['scaler_params']['mean'])
        scaler.scale_ = np.array(ckpt['scaler_params']['scale'])

        kmeans = KMeans(
            n_clusters=config['n_clusters'], random_state=42, n_init=10
        )
        kmeans.cluster_centers_ = np.array(ckpt['kmeans_centers'])

        features_scaled = scaler.transform(features)
        cluster_ids = kmeans.predict(features_scaled)
    else:
        # Feature dimension mismatch - refit clustering on new features
        print(f'  (Feature mismatch: {features.shape[1]} vs {n_features_expected})')
        print(f'  (Re-fitting K-means on new features)')
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        kmeans = KMeans(
            n_clusters=config['n_clusters'], random_state=42, n_init=10
        )
        cluster_ids = kmeans.fit_predict(features_scaled)

    # Process in batches to avoid OOM
    batch_size = 128
    n_samples = len(flux_data)
    all_probs = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            x_batch = torch.tensor(
                flux_data[i:end_idx], dtype=torch.float32
            ).unsqueeze(-1).to(device)
            c_batch = torch.tensor(
                cluster_ids[i:end_idx], dtype=torch.long
            ).to(device)
            logits = model(x_batch, c_batch)
            probs_batch = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs_batch)

    probs = np.concatenate(all_probs)

    preds = (probs > 0.5).astype(int)

    print(f'Predictions: {preds.sum()} positive out of {len(preds)}')
    print(f'Mean probability: {probs.mean():.4f}')
    print(f'Accuracy: {accuracy_score(labels, preds):.4f}')
    print(f'Precision: {precision_score(labels, preds, zero_division=0):.4f}')
    print(f'Recall: {recall_score(labels, preds, zero_division=0):.4f}')
    print(f'F1: {f1_score(labels, preds, zero_division=0):.4f}')

    return probs


def main():
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Use full Kepler dataset (1473 windows from 491 light curves)
    x_kepler = np.load(r'D:\CS_4280_Project\Code\data\windows_kepler_5pct\X.npy')
    y_kepler = np.load(r'D:\CS_4280_Project\Code\data\windows_kepler_5pct\y.npy')
    print(f'\nKepler test data: {x_kepler.shape[0]} windows')
    print(f'Positive samples: {y_kepler.sum()} ({100*y_kepler.mean():.1f}%)')

    features = compute_stats(x_kepler)
    print(f'Features shape: {features.shape}')

    models = [
        (r'D:\CS_4280_Project\Code\runs\sector1_final_0918\best.pt',
         'Original TESS Model (AUC 0.9261)', True),
        (r'D:\CS_4280_Project\Code\runs\finetuned_kepler\best.pt',
         'Fine-tuned (144 Kepler, 0.5%)', True),
        (r'D:\CS_4280_Project\Code\runs\finetuned_kepler_279\best.pt',
         'Fine-tuned (279 Kepler, 1.0%)', True),
        (r'D:\CS_4280_Project\Code\runs\finetuned_kepler_5pct\best.pt',
         'Fine-tuned (1393 Kepler, 5.0%)', True)
    ]

    results = {}
    for model_path, name, use_orig in models:
        probs = evaluate_model(
            model_path, name, x_kepler, y_kepler, features, device,
            use_original_arch=use_orig
        )
        results[name] = probs

    n_windows = len(y_kepler)
    print('\n' + '=' * 50)
    print('SUMMARY: Kepler Planet Detection Rate')
    print('=' * 50)
    print(f'(All {n_windows} Kepler samples are confirmed planets)')
    print()
    for name, probs in results.items():
        detected = (probs > 0.5).sum()
        rate = 100 * detected / len(probs)
        print(f'{name}:')
        print(f'  Detected: {detected}/{n_windows} ({rate:.1f}%)')
        print(f'  Mean prob: {probs.mean():.4f}')
        print()


if __name__ == '__main__':
    main()
