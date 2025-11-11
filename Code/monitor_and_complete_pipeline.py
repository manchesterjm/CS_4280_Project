"""
Monitor Optuna completion and automatically run remaining pipeline steps
"""
import time
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
OPTUNA_RESULTS_DIR = Path(r"C:\CS_4280_Project\Code\optuna_results_balanced")
WINDOWS_DIR = Path(r"C:\CS_4280_Project\Code\data\windows_train_400")
TESS_TEST_DIR = Path(r"C:\CS_4280_Project\Code\data\windows_planet_test")
CONDA_ENV = "exo-lstm-gpu"
CHECK_INTERVAL = 300  # Check every 5 minutes

def check_optuna_complete():
    """Check if Optuna has completed by looking for result files"""
    if not OPTUNA_RESULTS_DIR.exists():
        return False, None

    # Look for best_params JSON file
    json_files = list(OPTUNA_RESULTS_DIR.glob("best_params_*.json"))
    if json_files:
        # Get most recent
        latest = max(json_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            params = json.load(f)
        return True, params
    return False, None

def train_final_model(params):
    """Train final model with optimized hyperparameters"""
    print(f"\n{'='*80}")
    print("TRAINING FINAL MODEL WITH OPTIMIZED HYPERPARAMETERS")
    print(f"{'='*80}\n")

    best_params = params['best_params']

    # Calculate pos_weight for 1522 windows with 30.3% positive
    # pos_weight = n_neg / n_pos = 1061 / 461 = 2.302
    pos_weight = 2.302

    cmd = [
        "conda", "run", "-n", CONDA_ENV, "python",
        r"C:\CS_4280_Project\Code\train_bilstm_cluster.py",
        "--windows_dir", str(WINDOWS_DIR),
        "--n_clusters", str(best_params['n_clusters']),
        "--epochs", "80",
        "--batch_size", str(best_params['batch_size']),
        "--lr", str(best_params['lr']),
        "--hidden", str(best_params['hidden_size']),
        "--layers", str(best_params['num_layers']),
        "--dropout", str(best_params['dropout']),
        "--save_dir", r"C:\CS_4280_Project\Code\runs\bilstm_cluster_balanced_final",
        "--amp_dtype", "fp16",
        "--pos_weight", str(pos_weight),
        "--num_workers", "0"
    ]

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Training completed successfully!")
        return True
    else:
        print(f"❌ Training failed with error:\n{result.stderr}")
        return False

def test_on_tess_data():
    """Run inference on TESS test data"""
    print(f"\n{'='*80}")
    print("TESTING ON TESS DATA (100 CONFIRMED PLANETS)")
    print(f"{'='*80}\n")

    cmd = [
        "conda", "run", "-n", CONDA_ENV, "python",
        r"C:\CS_4280_Project\Code\inference_cluster_model.py",
        "--model_path", r"C:\CS_4280_Project\Code\runs\bilstm_cluster_balanced_final\best.pt",
        "--windows_dir", str(TESS_TEST_DIR),
        "--output_file", r"C:\CS_4280_Project\Code\reports\balanced_model_planet_predictions.csv"
    ]

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ TESS testing completed successfully!")
        return True
    else:
        print(f"❌ TESS testing failed with error:\n{result.stderr}")
        return False

def generate_comparison_report():
    """Generate comparison report between old and new models"""
    print(f"\n{'='*80}")
    print("GENERATING COMPARISON REPORT")
    print(f"{'='*80}\n")

    # Create a simple comparison report
    report_path = Path(r"C:\CS_4280_Project\COMPARISON_REPORT.md")

    report = f"""# Model Comparison Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

Comparison of OLD model (trained on imbalanced data) vs NEW model (trained on balanced synthetic data).

### OLD Model
- Training data: 655 windows (23% positive, imbalanced)
- Dataset: Real TESS data with class imbalance
- Best AUC: 0.6947 (Nov 9 Optuna optimization)
- TESS testing: 0/300 windows predicted positive (too conservative)

### NEW Model
- Training data: 1,522 windows (30.3% positive, balanced synthetic)
- Dataset: Balanced synthetic (200 planets + 200 non-planets)
- Optuna optimization: Completed on balanced dataset
- TESS testing: Results in balanced_model_planet_predictions.csv

## Files
- OLD model predictions: `Code/reports/optimized_planet_predictions.csv`
- NEW model predictions: `Code/reports/balanced_model_planet_predictions.csv`
- OLD model: `Code/runs/bilstm_cluster_optimized/best.pt`
- NEW model: `Code/runs/bilstm_cluster_balanced_final/best.pt`

## Analysis
Compare the prediction files to see:
1. Number of positive predictions (NEW should be > 0)
2. Prediction confidence distribution
3. Which planets each model identifies correctly

## Next Steps for Presentation
1. Load both prediction CSV files
2. Calculate metrics for each model
3. Create visualization comparing ROC curves
4. Show example predictions on specific planets
"""

    report_path.write_text(report)
    print(f"✅ Comparison report generated: {report_path}")
    return True

def main():
    """Main monitoring loop"""
    print(f"\n{'='*80}")
    print("AUTOMATED PIPELINE MONITOR")
    print(f"{'='*80}\n")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checking every {CHECK_INTERVAL} seconds for Optuna completion...")
    print(f"Output directory: {OPTUNA_RESULTS_DIR}\n")

    while True:
        complete, params = check_optuna_complete()

        if complete:
            print(f"\n✅ OPTUNA COMPLETED! Found results at {datetime.now().strftime('%H:%M:%S')}")
            print(f"Best AUC: {params.get('best_auc', 'N/A')}")
            print(f"Best trial: {params.get('best_trial', 'N/A')}\n")

            # Run remaining pipeline steps
            print("Starting automated pipeline execution...\n")

            # Step 1: Train final model
            if train_final_model(params):
                # Step 2: Test on TESS data
                if test_on_tess_data():
                    # Step 3: Generate comparison report
                    generate_comparison_report()

                    print(f"\n{'='*80}")
                    print("✅ PIPELINE COMPLETE!")
                    print(f"{'='*80}\n")
                    print(f"Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print("\nAll results ready for presentation!")
                    break

            print("\n⚠️ Pipeline encountered errors. Check logs above.")
            break
        else:
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"[{current_time}] Optuna still running... (checking again in {CHECK_INTERVAL}s)")
            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    except Exception as e:
        print(f"\n❌ Error in monitoring script: {e}")
