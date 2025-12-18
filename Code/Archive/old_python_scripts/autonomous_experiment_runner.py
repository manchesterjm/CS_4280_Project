"""
Autonomous Experiment Runner for BiLSTM+Clustering Exoplanet Detection
Runs multiple training configurations and logs results for comparison.

Author: Claude Code
Date: November 27, 2025
"""
import os
import sys

# Limit CPU threads BEFORE importing numpy/torch to prevent system overload
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Configuration
BASE_DIR = Path("C:/CS_4280_Project/Code")
TRAIN_DIR = BASE_DIR / "data/windows_sector1_full/train"
TEST_DIR = BASE_DIR / "data/windows_sector1_full/test"
RESULTS_DIR = BASE_DIR / "runs/sector1_experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Calculate correct pos_weight from actual class distribution
# Training: 3147 positive, 23325 negative -> ratio = 7.41
ACTUAL_POS_WEIGHT = 23325 / 3147  # = 7.41

# Experiment configurations (will run sequentially)
EXPERIMENTS = [
    {
        "name": "baseline_corrected_posweight",
        "description": "Baseline with corrected pos_weight (7.41 instead of 3.367)",
        "params": {
            "n_clusters": 5,
            "epochs": 60,
            "batch_size": 128,
            "lr": 0.000225,
            "hidden": 256,
            "layers": 4,
            "dropout": 0.311,
            "pos_weight": 7.41,
        }
    },
    {
        "name": "clusters_3",
        "description": "Fewer clusters (3) for simpler grouping",
        "params": {
            "n_clusters": 3,
            "epochs": 60,
            "batch_size": 128,
            "lr": 0.000225,
            "hidden": 256,
            "layers": 4,
            "dropout": 0.311,
            "pos_weight": 7.41,
        }
    },
    {
        "name": "clusters_7",
        "description": "More clusters (7) for finer grouping",
        "params": {
            "n_clusters": 7,
            "epochs": 60,
            "batch_size": 128,
            "lr": 0.000225,
            "hidden": 256,
            "layers": 4,
            "dropout": 0.311,
            "pos_weight": 7.41,
        }
    },
    {
        "name": "higher_lr",
        "description": "Higher learning rate (0.0005) for faster convergence",
        "params": {
            "n_clusters": 5,
            "epochs": 60,
            "batch_size": 128,
            "lr": 0.0005,
            "hidden": 256,
            "layers": 4,
            "dropout": 0.311,
            "pos_weight": 7.41,
        }
    },
    {
        "name": "deeper_model",
        "description": "Deeper model (5 layers) with more capacity",
        "params": {
            "n_clusters": 5,
            "epochs": 60,
            "batch_size": 128,
            "lr": 0.000225,
            "hidden": 256,
            "layers": 5,
            "dropout": 0.35,
            "pos_weight": 7.41,
        }
    },
    {
        "name": "wider_model",
        "description": "Wider model (384 hidden) with more capacity",
        "params": {
            "n_clusters": 5,
            "epochs": 60,
            "batch_size": 64,  # Smaller batch for memory
            "lr": 0.0002,
            "hidden": 384,
            "layers": 4,
            "dropout": 0.35,
            "pos_weight": 7.41,
        }
    },
]


def run_experiment(exp_config):
    """Run a single training experiment."""
    name = exp_config["name"]
    params = exp_config["params"]
    save_dir = RESULTS_DIR / name

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"Description: {exp_config['description']}")
    print(f"{'='*70}")

    # Build command
    cmd = [
        sys.executable, "train_bilstm_cluster.py",
        "--windows_dir", str(TRAIN_DIR),
        "--n_clusters", str(params["n_clusters"]),
        "--epochs", str(params["epochs"]),
        "--batch_size", str(params["batch_size"]),
        "--lr", str(params["lr"]),
        "--hidden", str(params["hidden"]),
        "--layers", str(params["layers"]),
        "--dropout", str(params["dropout"]),
        "--pos_weight", str(params["pos_weight"]),
        "--save_dir", str(save_dir),
        "--amp_dtype", "fp16",
        "--num_workers", "0",
        "--seed", "42"
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = time.time()

    try:
        # Run training
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout per experiment
        )

        elapsed = time.time() - start_time

        # Save output log
        log_file = save_dir / "training_log.txt"
        with open(log_file, 'w') as f:
            f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")

        # Parse results from output
        metrics = parse_metrics(result.stdout)
        metrics["elapsed_seconds"] = elapsed
        metrics["return_code"] = result.returncode

        print(f"\nCompleted in {elapsed/60:.1f} minutes")
        print(f"Return code: {result.returncode}")
        if metrics.get("best_auc"):
            print(f"Best AUC: {metrics['best_auc']:.4f}")
        if metrics.get("best_f1"):
            print(f"Best F1: {metrics['best_f1']:.4f}")

        return metrics

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after 2 hours")
        return {"error": "timeout", "elapsed_seconds": 7200}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"error": str(e)}


def parse_metrics(stdout):
    """Parse training metrics from stdout."""
    metrics = {}

    for line in stdout.split('\n'):
        # Look for validation metrics
        if 'val_auc=' in line.lower() or 'auc=' in line.lower():
            try:
                # Extract AUC value
                if 'best' in line.lower():
                    parts = line.split('auc=')
                    if len(parts) > 1:
                        val = parts[1].split()[0].strip(',')
                        metrics['best_auc'] = float(val)
            except:
                pass

        if 'val_f1=' in line.lower() or 'f1=' in line.lower():
            try:
                parts = line.split('f1=')
                if len(parts) > 1:
                    val = parts[1].split()[0].strip(',')
                    if 'best_f1' not in metrics:
                        metrics['best_f1'] = float(val)
            except:
                pass

        # Look for final metrics
        if 'Best checkpoint' in line or 'best.pt' in line:
            metrics['checkpoint_saved'] = True

    return metrics


def run_test_evaluation(model_path, test_dir, output_file):
    """Run inference on test set."""
    cmd = [
        sys.executable, "inference_cluster_model.py",
        "--model_path", str(model_path),
        "--windows_dir", str(test_dir),
        "--output_file", str(output_file)
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=600
        )
        return result.returncode == 0
    except:
        return False


def main():
    print("="*70)
    print("AUTONOMOUS EXPERIMENT RUNNER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training data: {TRAIN_DIR}")
    print(f"Test data: {TEST_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Number of experiments: {len(EXPERIMENTS)}")
    print(f"Corrected pos_weight: {ACTUAL_POS_WEIGHT:.2f}")
    print("="*70)

    all_results = []

    for i, exp in enumerate(EXPERIMENTS):
        print(f"\n[{i+1}/{len(EXPERIMENTS)}] Running experiment: {exp['name']}")

        metrics = run_experiment(exp)

        result = {
            "name": exp["name"],
            "description": exp["description"],
            "params": exp["params"],
            "metrics": metrics
        }
        all_results.append(result)

        # Save intermediate results
        results_file = RESULTS_DIR / "experiment_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to {results_file}")

    # Final summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    # Sort by AUC
    sorted_results = sorted(
        [r for r in all_results if r["metrics"].get("best_auc")],
        key=lambda x: x["metrics"].get("best_auc", 0),
        reverse=True
    )

    print(f"\n{'Experiment':<30} {'AUC':>10} {'F1':>10} {'Time (min)':>12}")
    print("-"*65)

    for r in sorted_results:
        auc = r["metrics"].get("best_auc", 0)
        f1 = r["metrics"].get("best_f1", 0)
        time_min = r["metrics"].get("elapsed_seconds", 0) / 60
        print(f"{r['name']:<30} {auc:>10.4f} {f1:>10.4f} {time_min:>12.1f}")

    if sorted_results:
        best = sorted_results[0]
        print(f"\nBEST MODEL: {best['name']}")
        print(f"  AUC: {best['metrics'].get('best_auc', 'N/A')}")
        print(f"  Config: {best['params']}")

        # Run test evaluation on best model
        best_model = RESULTS_DIR / best['name'] / "best.pt"
        if best_model.exists():
            print(f"\nRunning test evaluation on best model...")
            test_output = RESULTS_DIR / f"{best['name']}_test_predictions.csv"
            if run_test_evaluation(best_model, TEST_DIR, test_output):
                print(f"Test predictions saved to: {test_output}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    main()
