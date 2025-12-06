"""
Simple training script for TESS Sector 1 dataset.
"""
import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Import from existing training script
from train_bilstm_cluster import ClusterBiLSTM, LightCurveDataset

def main():
    print("="*70)
    print("SECTOR 1 TRAINING SCRIPT")
    print("="*70)

    # Check GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("WARNING: No GPU, using CPU")

    # Configuration
    config = {
        'hidden': 256,
        'layers': 4,
        'n_clusters': 5,
        'cluster_embed_dim': 32,
        'dropout': 0.311,
        'lr': 0.000225,
        'weight_decay': 7.56e-6,
        'batch_size': 128,
        'pos_weight': 3.367,
        'epochs': 80,
        'patience': 15
    }

    # Paths - use absolute paths
    base_dir = Path('C:/CS_4280_Project/Code')
    train_dir = base_dir / 'runs/sector1_experiments/data_w2048/train'
    save_dir = base_dir / 'runs/sector1_experiments/model_w2048'
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading data from {train_dir}...")

    # Load data
    X = np.load(train_dir / 'X.npy')
    y = np.load(train_dir / 'y.npy')
    meta = pd.read_csv(train_dir / 'meta.csv')

    print(f"Data shape: {X.shape}")
    print(f"Labels: {np.sum(y==1)} positive, {np.sum(y==0)} negative")
    print(f"Positive rate: {np.mean(y):.1%}")

    # K-means clustering
    feature_cols = ['mean', 'std', 'var', 'skew', 'range', 'median', 'mad', 'peak_to_peak']
    available_cols = [c for c in feature_cols if c in meta.columns]

    if len(available_cols) >= 4:
        print(f"\nClustering on: {available_cols}")
        features = meta[available_cols].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=config['n_clusters'], random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(features_scaled)
        print(f"Cluster distribution: {np.bincount(cluster_ids)}")
    else:
        print("No clustering features, using single cluster")
        cluster_ids = np.zeros(len(X), dtype=np.int64)
        scaler = None
        kmeans = None

    # Train/val split
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.15, stratify=y, random_state=42
    )

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    c_train, c_val = cluster_ids[train_idx], cluster_ids[val_idx]

    print(f"\nTraining: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")

    # Create dataloaders
    train_dataset = LightCurveDataset(X_train, y_train, c_train)
    val_dataset = LightCurveDataset(X_val, y_val, c_val)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=0, pin_memory=True
    )

    # Create model
    model = ClusterBiLSTM(
        input_size=1,
        hidden_size=config['hidden'],
        num_layers=config['layers'],
        n_clusters=config['n_clusters'],
        cluster_embed_dim=config['cluster_embed_dim'],
        dropout=config['dropout']
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    pos_weight = torch.tensor([config['pos_weight']]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Mixed precision
    scaler_amp = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    amp_dtype = torch.float16 if device.type == 'cuda' else torch.float32

    # Training loop
    best_auc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_auc': [], 'val_f1': []}

    print(f"\nStarting training for {config['epochs']} epochs...")
    start_time = time.time()

    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        for x_batch, y_batch, c_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            c_batch = c_batch.to(device)

            optimizer.zero_grad()

            if scaler_amp:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits = model(x_batch, c_batch)
                    loss = criterion(logits.squeeze(), y_batch.float())

                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                logits = model(x_batch, c_batch)
                loss = criterion(logits.squeeze(), y_batch.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x_batch, y_batch, c_batch in val_loader:
                x_batch = x_batch.to(device)
                c_batch = c_batch.to(device)

                if scaler_amp:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        logits = model(x_batch, c_batch)
                else:
                    logits = model(x_batch, c_batch)

                probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_batch.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        val_auc = roc_auc_score(all_labels, all_probs)
        val_preds = (all_probs > 0.5).astype(int)
        val_f1 = f1_score(all_labels, val_preds)

        history['train_loss'].append(train_loss)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)

        scheduler.step(val_auc)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config['epochs']}: "
                  f"Loss={train_loss:.4f}, AUC={val_auc:.4f}, F1={val_f1:.4f}")

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': config,
                'val_metrics': {'auc': val_auc, 'f1': val_f1},
                'scaler_params': {
                    'mean': scaler.mean_.tolist() if scaler else None,
                    'scale': scaler.scale_.tolist() if scaler else None
                },
                'kmeans_centers': kmeans.cluster_centers_.tolist() if kmeans else None,
                'epoch': epoch
            }
            torch.save(checkpoint, save_dir / 'best.pt')
            print(f"  -> New best! Saved to {save_dir / 'best.pt'}")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time/60:.1f} minutes")
    print(f"Best validation AUC: {best_auc:.4f}")

    # Save final checkpoint and history
    torch.save(checkpoint, save_dir / 'last.pt')
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to {save_dir}")

    # Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)

    test_dir = base_dir / 'runs/sector1_experiments/data_w2048/test'
    X_test = np.load(test_dir / 'X.npy')
    y_test = np.load(test_dir / 'y.npy')
    meta_test = pd.read_csv(test_dir / 'meta.csv')

    print(f"Test data: {len(X_test)} samples")

    # Compute cluster IDs for test data
    if kmeans and scaler:
        features_test = meta_test[available_cols].values
        features_test_scaled = scaler.transform(features_test)
        cluster_ids_test = kmeans.predict(features_test_scaled)
    else:
        cluster_ids_test = np.zeros(len(X_test), dtype=np.int64)

    test_dataset = LightCurveDataset(X_test, y_test, cluster_ids_test)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # Load best model
    checkpoint = torch.load(save_dir / 'best.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_probs = []
    with torch.no_grad():
        for x_batch, _, c_batch in test_loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)

            if device.type == 'cuda':
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits = model(x_batch, c_batch)
            else:
                logits = model(x_batch, c_batch)

            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            all_probs.extend(probs)

    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)

    # Metrics
    test_auc = roc_auc_score(y_test, all_probs)
    test_f1 = f1_score(y_test, all_preds)
    test_prec = precision_score(y_test, all_preds, zero_division=0)
    test_rec = recall_score(y_test, all_preds, zero_division=0)
    test_acc = accuracy_score(y_test, all_preds)
    cm = confusion_matrix(y_test, all_preds)

    print(f"\nTEST SET RESULTS:")
    print(f"  AUC:       {test_auc:.4f}")
    print(f"  F1:        {test_f1:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    print(f"\nPredicted positive: {np.sum(all_preds)}/{len(all_preds)} ({np.mean(all_preds)*100:.1f}%)")

    # Save test results
    results = {
        'train_auc': best_auc,
        'train_f1': max(history['val_f1']),
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_accuracy': test_acc,
        'confusion_matrix': cm.tolist(),
        'training_time_min': training_time / 60
    }
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTest results saved to {save_dir / 'test_results.json'}")
    print("="*70)


if __name__ == '__main__':
    main()
