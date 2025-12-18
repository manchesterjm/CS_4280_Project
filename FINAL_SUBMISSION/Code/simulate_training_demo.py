# -*- coding: utf-8 -*-
"""
Simulated BiLSTM Training Visualization for Video Recording
Creates a 20-second visual progress display showing model "learning"

FINAL MODEL RESULTS (December 2025):
- Dataset: 33,051 windows (7,686 planets, 25,365 non-planets)
- Architecture: 4-layer BiLSTM, 192 hidden, 7 clusters, 64-dim embeddings
- Final: AUC 0.9261, Recall 100%, F1 0.5708

Run in PowerShell and record the screen for presentation:
    python simulate_training_demo.py
"""

import time
import sys
import random
from datetime import datetime


class Colors:
    """ANSI color codes for Windows PowerShell"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    WHITE = '\033[97m'


def progress_bar(current, total, width=40, label="Progress", color=Colors.GREEN):
    """Create a visual progress bar"""
    filled = int(width * current / total)
    bar = '#' * filled + '-' * (width - filled)
    percent = 100 * current / total
    return f"{color}{label}: [{bar}] {percent:6.2f}%{Colors.RESET}"


def print_header():
    """Print the header banner"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}   BiLSTM + K-means Clustering - Exoplanet Transit Detection{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
    print(f"{Colors.WHITE}Dataset: TESS Sector 1 Ground Truth{Colors.RESET}")
    print(f"{Colors.WHITE}  - 33,051 windows (7,686 planets, 25,365 non-planets){Colors.RESET}")
    print(f"{Colors.WHITE}  - Train: 26,472 | Test: 6,579{Colors.RESET}")
    print(f"{Colors.WHITE}Architecture: 4-layer BiLSTM + 7 Cluster Embeddings (64-dim){Colors.RESET}")
    print(f"{Colors.WHITE}Optimization: AdamW, LR=1e-4, Batch=128, FP16 Mixed Precision{Colors.RESET}\n")


def simulate_data_loading():
    """Simulate data loading phase"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}Phase 1: Loading TESS Sector 1 Data{Colors.RESET}")
    print(f"{Colors.WHITE}{'-'*70}{Colors.RESET}\n")

    total_windows = 33051
    duration = 3  # seconds
    steps = 30

    for i in range(steps + 1):
        current = int(total_windows * i / steps)
        bar = progress_bar(current, total_windows, width=50,
                          label=f"Loading ({current:,}/{total_windows:,})",
                          color=Colors.CYAN)
        sys.stdout.write('\r' + bar)
        sys.stdout.flush()
        time.sleep(duration / steps)

    print(f"\n{Colors.GREEN}[OK] Data loaded: 26,472 train / 6,579 test{Colors.RESET}\n")
    time.sleep(0.3)


def simulate_clustering():
    """Simulate K-means clustering phase"""
    print(f"{Colors.BOLD}{Colors.YELLOW}Phase 2: K-means Clustering (7 clusters){Colors.RESET}")
    print(f"{Colors.WHITE}{'-'*70}{Colors.RESET}\n")

    iterations = 15
    duration = 2  # seconds

    for i in range(iterations + 1):
        bar = progress_bar(i, iterations, width=50,
                          label=f"Iteration {i:2d}/{iterations}",
                          color=Colors.MAGENTA)
        inertia = 2500 - (i * 120) + random.uniform(-30, 30)
        status = f"{Colors.WHITE}Inertia: {inertia:.1f}{Colors.RESET}"
        sys.stdout.write('\r' + bar + '  ' + status)
        sys.stdout.flush()
        time.sleep(duration / iterations)

    print(f"\n{Colors.GREEN}[OK] Clustering complete! 7 clusters assigned{Colors.RESET}\n")
    time.sleep(0.3)


def simulate_training():
    """Simulate model training with metrics"""
    print(f"{Colors.BOLD}{Colors.YELLOW}Phase 3: BiLSTM Training (3.07M parameters){Colors.RESET}")
    print(f"{Colors.WHITE}{'-'*70}{Colors.RESET}\n")

    epochs = 8
    duration = 12  # seconds

    # Starting values (realistic progression)
    train_loss = 0.680
    val_loss = 0.690
    val_auc = 0.550
    val_recall = 0.600

    best_auc = 0

    for epoch in range(1, epochs + 1):
        # Simulate improving metrics
        train_loss = max(0.180, train_loss - random.uniform(0.050, 0.080))
        val_loss = max(0.200, val_loss - random.uniform(0.040, 0.070))
        val_auc = min(0.926, val_auc + random.uniform(0.040, 0.060))
        val_recall = min(1.000, val_recall + random.uniform(0.040, 0.060))

        # Progress bar
        bar = progress_bar(epoch, epochs, width=35,
                          label=f"Epoch {epoch}/{epochs}",
                          color=Colors.GREEN)

        # Metrics
        print(bar)
        print(f"  {Colors.CYAN}Train Loss: {train_loss:.4f}{Colors.RESET}  "
              f"{Colors.MAGENTA}Val Loss: {val_loss:.4f}{Colors.RESET}  "
              f"{Colors.GREEN}Val AUC: {val_auc:.4f}{Colors.RESET}  "
              f"{Colors.YELLOW}Recall: {val_recall:.2%}{Colors.RESET}")

        # Best model indicator
        if val_auc > best_auc:
            best_auc = val_auc
            print(f"  {Colors.GREEN}* New best model! (AUC: {val_auc:.4f}){Colors.RESET}")

        print()
        time.sleep(duration / epochs)

    print(f"{Colors.BOLD}{Colors.GREEN}[OK] Training complete!{Colors.RESET}\n")


def print_summary():
    """Print final summary with actual results"""
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}   FINAL RESULTS{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

    print(f"{Colors.WHITE}Test Set Performance (6,579 windows):{Colors.RESET}")
    print(f"  {Colors.GREEN}AUC:       0.9261  (92.61%){Colors.RESET}")
    print(f"  {Colors.GREEN}Recall:    1.0000  (100% - all planets detected!){Colors.RESET}")
    print(f"  {Colors.CYAN}F1 Score:  0.5708{Colors.RESET}")
    print(f"  {Colors.CYAN}Precision: 0.3993  (39.93%){Colors.RESET}")
    print(f"  {Colors.CYAN}Accuracy:  0.8326  (83.26%){Colors.RESET}\n")

    print(f"{Colors.WHITE}Model Architecture:{Colors.RESET}")
    print(f"  {Colors.YELLOW}Parameters: 3,068,801 (3.07M){Colors.RESET}")
    print(f"  {Colors.YELLOW}BiLSTM: 4 layers x 192 hidden (bidirectional){Colors.RESET}")
    print(f"  {Colors.YELLOW}Clusters: 7 -> 64-dim embeddings{Colors.RESET}")
    print(f"  {Colors.YELLOW}Dropout: 0.334{Colors.RESET}\n")

    print(f"{Colors.BOLD}{Colors.GREEN}* 100% Recall: No planets missed!{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}* +17% AUC improvement over midterm{Colors.RESET}\n")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")


def main():
    """Main simulation function - ~20 seconds total"""
    import os
    os.system('color')  # Enable ANSI colors on Windows

    print_header()
    time.sleep(0.5)

    # Phase 1: Data loading (~3 seconds)
    simulate_data_loading()

    # Phase 2: Clustering (~2 seconds)
    simulate_clustering()

    # Phase 3: Training (~12 seconds)
    simulate_training()

    # Summary (~2 seconds)
    print_summary()
    time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted{Colors.RESET}\n")
        sys.exit(0)
