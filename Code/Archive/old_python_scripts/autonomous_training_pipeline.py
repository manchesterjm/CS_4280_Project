"""
Autonomous Training Pipeline for TESS Sector 1 Exoplanet Detection.
Builds dataset, trains BiLSTM+Clustering model, evaluates, and experiments with window sizes.
"""
import os
import sys

# Limit CPU threads BEFORE importing numpy/torch to prevent system overload
# When using parallel workers, each worker should use minimal threads
os.environ['OMP_NUM_THREADS'] = '2'  # Each worker uses max 2 threads
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# Add Code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from build_sector1_dataset_v2 import build_dataset


def check_gpu():
    """Check GPU availability and print info."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device
    else:
        print("WARNING: No GPU available, using CPU (will be slow)")
        return torch.device('cpu')


def train_model(windows_dir, save_dir, config, device):
    """
    Train BiLSTM+Clustering model.

    Args:
        windows_dir: Directory with X.npy, y.npy, meta.csv
        save_dir: Where to save model checkpoints
        config: Training configuration dict
        device: torch device (cuda/cpu)

    Returns:
        dict with training results
    """
    from train_bilstm_cluster import ClusterBiLSTM, LightCurveDataset
    from torch.utils.data import DataLoader

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X = np.load(Path(windows_dir) / 'X.npy')
    y = np.load(Path(windows_dir) / 'y.npy')
    meta = pd.read_csv(Path(windows_dir) / 'meta.csv')

    print(f"\nLoaded data: {X.shape[0]} windows, {X.shape[1]} timesteps")
    print(f"Positive rate: {np.mean(y):.1%}")

    # K-means clustering on statistical features
    feature_cols = ['mean', 'std', 'var', 'skew', 'range', 'median', 'mad', 'peak_to_peak']
    available_cols = [c for c in feature_cols if c in meta.columns]

    if len(available_cols) >= 4:
        print(f"Clustering on features: {available_cols}")
        features = meta[available_cols].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=config['n_clusters'], random_state=42, n_init=10)
        cluster_ids = kmeans.fit_predict(features_scaled)
    else:
        print("Warning: No clustering features found, using single cluster")
        cluster_ids = np.zeros(len(X), dtype=np.int64)
        scaler = None
        kmeans = None

    print(f"Cluster distribution: {np.bincount(cluster_ids)}")

    # Train/val split (stratified)
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.15, stratify=y, random_state=42
    )

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    c_train, c_val = cluster_ids[train_idx], cluster_ids[val_idx]

    print(f"Training: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")

    # Create datasets and loaders
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
        cluster_embed_dim=config.get('cluster_embed_dim', 32),
        dropout=config['dropout']
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 1e-5)
    )

    # Class weighting for imbalanced data
    pos_weight = torch.tensor([config['pos_weight']]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Mixed precision training
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
                y_batch = y_batch.to(device)
                c_batch = c_batch.to(device)

                if scaler_amp:
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        logits = model(x_batch, c_batch)
                else:
                    logits = model(x_batch, c_batch)

                probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(y_batch.cpu().numpy())

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
                'val_metrics': {
                    'auc': val_auc,
                    'f1': val_f1
                },
                'scaler_params': {
                    'mean': scaler.mean_.tolist() if scaler else None,
                    'scale': scaler.scale_.tolist() if scaler else None
                },
                'kmeans_centers': kmeans.cluster_centers_.tolist() if kmeans else None,
                'epoch': epoch
            }
            torch.save(checkpoint, save_dir / 'best.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 15):
                print(f"Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time/60:.1f} minutes")
    print(f"Best validation AUC: {best_auc:.4f}")

    # Save final model and history
    torch.save(checkpoint, save_dir / 'last.pt')
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    return {
        'best_auc': best_auc,
        'best_f1': max(history['val_f1']),
        'training_time': training_time,
        'epochs_trained': len(history['train_loss'])
    }


def evaluate_model(model_path, test_dir, device):
    """Evaluate trained model on test set."""
    from train_bilstm_cluster import ClusterBiLSTM, LightCurveDataset
    from torch.utils.data import DataLoader

    # Load test data
    X_test = np.load(Path(test_dir) / 'X.npy')
    y_test = np.load(Path(test_dir) / 'y.npy')
    meta_test = pd.read_csv(Path(test_dir) / 'meta.csv')

    print(f"\nLoaded test data: {len(X_test)} windows")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']

    model = ClusterBiLSTM(
        input_size=1,
        hidden_size=config['hidden'],
        num_layers=config['layers'],
        n_clusters=config['n_clusters'],
        cluster_embed_dim=config.get('cluster_embed_dim', 32),
        dropout=config['dropout']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Compute cluster IDs for test data
    feature_cols = ['mean', 'std', 'var', 'skew', 'range', 'median', 'mad', 'peak_to_peak']
    available_cols = [c for c in feature_cols if c in meta_test.columns]

    if checkpoint['kmeans_centers'] and len(available_cols) >= 4:
        features = meta_test[available_cols].values
        scaler = StandardScaler()
        scaler.mean_ = np.array(checkpoint['scaler_params']['mean'])
        scaler.scale_ = np.array(checkpoint['scaler_params']['scale'])
        features_scaled = scaler.transform(features)

        kmeans = KMeans(n_clusters=len(checkpoint['kmeans_centers']))
        kmeans.cluster_centers_ = np.array(checkpoint['kmeans_centers'])
        cluster_ids = kmeans.predict(features_scaled)
    else:
        cluster_ids = np.zeros(len(X_test), dtype=np.int64)

    # Create dataset and loader
    test_dataset = LightCurveDataset(X_test, y_test, cluster_ids)
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True
    )

    # Run inference
    all_probs = []
    amp_dtype = torch.float16 if device.type == 'cuda' else torch.float32

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

    # Compute metrics
    auc = roc_auc_score(y_test, all_probs)
    f1 = f1_score(y_test, all_preds)
    precision = precision_score(y_test, all_preds, zero_division=0)
    recall = recall_score(y_test, all_preds, zero_division=0)
    accuracy = accuracy_score(y_test, all_preds)
    cm = confusion_matrix(y_test, all_preds)

    results = {
        'auc': auc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'n_samples': len(y_test),
        'n_positive': int(np.sum(y_test == 1)),
        'n_predicted_positive': int(np.sum(all_preds == 1))
    }

    print(f"\nTEST SET RESULTS:")
    print(f"  AUC: {auc:.4f}")
    print(f"  F1:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    print(f"  Predicted positive: {np.sum(all_preds == 1)}/{len(all_preds)} ({np.mean(all_preds)*100:.1f}%)")

    return results


def run_pipeline(data_dir, output_base_dir, window_sizes=[2048]):
    """
    Run the complete training pipeline.

    Args:
        data_dir: Path to TESS Sector 1 ground truth
        output_base_dir: Base directory for outputs
        window_sizes: List of window sizes to try
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    device = check_gpu()
    results_summary = []

    # Best hyperparameters from previous Optuna optimization
    base_config = {
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

    for seq_len in window_sizes:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: Window Size = {seq_len}")
        print(f"{'='*70}")

        # Create directories for this experiment
        data_output_dir = output_base_dir / f'data_w{seq_len}'
        model_output_dir = output_base_dir / f'model_w{seq_len}'

        # Step 1: Build dataset
        print(f"\n[Step 1] Building dataset with window size {seq_len}...")
        try:
            dataset_info = build_dataset(
                data_dir=data_dir,
                output_dir=data_output_dir,
                seq_len=seq_len,
                n_windows=3,
                test_size=0.2,
                n_jobs=-1,  # Will use cpu_count()-1 (leaves 1 core free)
                seed=42
            )
        except Exception as e:
            print(f"ERROR building dataset: {e}")
            continue

        # Step 2: Train model
        print(f"\n[Step 2] Training model...")
        config = base_config.copy()
        config['seq_len'] = seq_len

        try:
            train_results = train_model(
                windows_dir=data_output_dir / 'train',
                save_dir=model_output_dir,
                config=config,
                device=device
            )
        except Exception as e:
            print(f"ERROR training model: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Step 3: Evaluate on test set
        print(f"\n[Step 3] Evaluating on test set...")
        try:
            test_results = evaluate_model(
                model_path=model_output_dir / 'best.pt',
                test_dir=data_output_dir / 'test',
                device=device
            )
        except Exception as e:
            print(f"ERROR evaluating model: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Record results
        experiment_result = {
            'window_size': seq_len,
            'train_windows': dataset_info['train_windows'],
            'test_windows': dataset_info['test_windows'],
            'train_auc': train_results['best_auc'],
            'train_f1': train_results['best_f1'],
            'test_auc': test_results['auc'],
            'test_f1': test_results['f1'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'training_time_min': train_results['training_time'] / 60
        }
        results_summary.append(experiment_result)

        # Save intermediate results
        with open(output_base_dir / 'results_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)

    # Print final summary
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Window':<10} {'Train AUC':<12} {'Test AUC':<12} {'Test F1':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)
    for r in results_summary:
        print(f"{r['window_size']:<10} {r['train_auc']:<12.4f} {r['test_auc']:<12.4f} "
              f"{r['test_f1']:<10.4f} {r['test_precision']:<10.4f} {r['test_recall']:<10.4f}")

    # Find best result
    if results_summary:
        best = max(results_summary, key=lambda x: x['test_auc'])
        print(f"\nBest model: Window size {best['window_size']} with Test AUC {best['test_auc']:.4f}")

    return results_summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Autonomous Training Pipeline')
    parser.add_argument('--data_dir', type=str,
                       default=r'E:\lilith4_sector-1_groundtruth\sector-1\ground-truth')
    parser.add_argument('--output_dir', type=str,
                       default=r'C:\CS_4280_Project\Code\runs\sector1_experiments')
    parser.add_argument('--window_sizes', type=int, nargs='+',
                       default=[2048],
                       help='Window sizes to experiment with')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"AUTONOMOUS TRAINING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    results = run_pipeline(
        data_dir=args.data_dir,
        output_base_dir=args.output_dir,
        window_sizes=args.window_sizes
    )

    print(f"\nPipeline completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
