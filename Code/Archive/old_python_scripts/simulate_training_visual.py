"""
Simulated BiLSTM Training Visualization for Video Recording
Creates a 20-second visual progress display showing model "learning"
Run in PowerShell and record the screen for presentation
"""

import time
import sys
import random
from datetime import datetime

# ANSI color codes for Windows PowerShell
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    MAGENTA = '\033[95m'
    RED = '\033[91m'
    WHITE = '\033[97m'

def clear_line():
    """Clear the current line"""
    sys.stdout.write('\r' + ' ' * 100 + '\r')
    sys.stdout.flush()

def print_header():
    """Print the header banner"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}   BiLSTM + K-means Clustering - Exoplanet Detection Training{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")
    print(f"{Colors.WHITE}Dataset: 655 windows (150 planets, 505 non-planets){Colors.RESET}")
    print(f"{Colors.WHITE}Architecture: 3-layer BiLSTM + Cluster Embeddings (5 clusters){Colors.RESET}")
    print(f"{Colors.WHITE}Optimization: Adam, LR=1e-4, Batch Size=64, FP16 Mixed Precision{Colors.RESET}\n")

def progress_bar(current, total, width=40, label="Progress", color=Colors.GREEN):
    """Create a visual progress bar"""
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    percent = 100 * current / total
    return f"{color}{label}: [{bar}] {percent:6.2f}%{Colors.RESET}"

def simulate_window_processing():
    """Simulate window building/loading phase"""
    print(f"\n{Colors.BOLD}{Colors.YELLOW}Phase 1: Building Training Windows{Colors.RESET}")
    print(f"{Colors.WHITE}{'─'*70}{Colors.RESET}\n")

    total_windows = 655
    duration = 5  # seconds
    steps = 50

    for i in range(steps + 1):
        current_windows = int(total_windows * i / steps)
        elapsed = i / steps * duration

        # Progress bar
        bar = progress_bar(current_windows, total_windows, width=50,
                          label=f"Windows ({current_windows}/{total_windows})",
                          color=Colors.CYAN)

        # Status info
        status = f"{Colors.WHITE}[{elapsed:.1f}s] Extracting features: BLS, period, depth...{Colors.RESET}"

        # Print everything on one line and update
        sys.stdout.write('\r' + bar + '  ' + status)
        sys.stdout.flush()

        time.sleep(duration / steps)

    print(f"\n{Colors.GREEN}✓ Window extraction complete!{Colors.RESET}\n")
    time.sleep(0.5)

def simulate_clustering():
    """Simulate K-means clustering phase"""
    print(f"{Colors.BOLD}{Colors.YELLOW}Phase 2: K-means Clustering (5 clusters){Colors.RESET}")
    print(f"{Colors.WHITE}{'─'*70}{Colors.RESET}\n")

    clusters = 5
    iterations = 20
    duration = 2  # seconds

    for i in range(iterations + 1):
        bar = progress_bar(i, iterations, width=50,
                          label=f"Clustering Iteration {i}/{iterations}",
                          color=Colors.MAGENTA)

        inertia = 1500 - (i * 50) + random.uniform(-20, 20)
        status = f"{Colors.WHITE}Inertia: {inertia:.1f}{Colors.RESET}"

        sys.stdout.write('\r' + bar + '  ' + status)
        sys.stdout.flush()

        time.sleep(duration / iterations)

    print(f"\n{Colors.GREEN}✓ Clustering complete! Cluster distribution: [145, 132, 128, 125, 125]{Colors.RESET}\n")
    time.sleep(0.5)

def simulate_training():
    """Simulate model training with metrics"""
    print(f"{Colors.BOLD}{Colors.YELLOW}Phase 3: BiLSTM Training{Colors.RESET}")
    print(f"{Colors.WHITE}{'─'*70}{Colors.RESET}\n")

    epochs = 10
    duration = 11  # seconds

    # Starting values
    train_loss = 0.693
    val_loss = 0.695
    train_auc = 0.520
    val_auc = 0.515

    for epoch in range(1, epochs + 1):
        # Simulate improving metrics with some noise
        train_loss = max(0.200, train_loss - random.uniform(0.030, 0.060))
        val_loss = max(0.220, val_loss - random.uniform(0.025, 0.055))
        train_auc = min(0.950, train_auc + random.uniform(0.025, 0.050))
        val_auc = min(0.805, val_auc + random.uniform(0.020, 0.045))

        # Progress bar for epoch
        bar = progress_bar(epoch, epochs, width=40,
                          label=f"Epoch {epoch:2d}/{epochs}",
                          color=Colors.GREEN)

        # Metrics display
        metrics = (f"{Colors.CYAN}Train Loss: {train_loss:.4f}{Colors.RESET}  "
                  f"{Colors.CYAN}Train AUC: {train_auc:.4f}{Colors.RESET}  "
                  f"{Colors.MAGENTA}Val Loss: {val_loss:.4f}{Colors.RESET}  "
                  f"{Colors.MAGENTA}Val AUC: {val_auc:.4f}{Colors.RESET}")

        # Print progress
        print(bar)
        print(f"  {metrics}")

        # Best model indicator
        if epoch > 1 and val_auc > 0.750:
            print(f"  {Colors.GREEN}★ New best model saved! (Val AUC improved){Colors.RESET}")

        print()

        time.sleep(duration / epochs)

    print(f"{Colors.BOLD}{Colors.GREEN}✓ Training complete!{Colors.RESET}\n")

def print_summary():
    """Print final summary"""
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}   Training Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

    print(f"{Colors.WHITE}Final Performance:{Colors.RESET}")
    print(f"  {Colors.GREEN}Validation AUC:  0.7572  (+9.0% improvement){Colors.RESET}")
    print(f"  {Colors.GREEN}Validation Loss: 0.2234{Colors.RESET}")
    print(f"  {Colors.CYAN}Training AUC:    0.9250{Colors.RESET}")
    print(f"  {Colors.CYAN}Training Loss:   0.2015{Colors.RESET}\n")

    print(f"{Colors.WHITE}Model Architecture:{Colors.RESET}")
    print(f"  {Colors.YELLOW}Parameters: 3,892,097 (~3.9M){Colors.RESET}")
    print(f"  {Colors.YELLOW}BiLSTM Layers: 3 (256 hidden units each){Colors.RESET}")
    print(f"  {Colors.YELLOW}Cluster Embeddings: 5 clusters → 32-dim vectors{Colors.RESET}\n")

    print(f"{Colors.BOLD}{Colors.GREEN}Model saved to: runs/bilstm_cluster_optimized/best.pt{Colors.RESET}\n")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

def main():
    """Main simulation function"""
    # Enable ANSI color support on Windows
    import os
    os.system('color')

    print_header()
    time.sleep(1)

    # Phase 1: Window processing (5 seconds)
    simulate_window_processing()

    # Phase 2: Clustering (2 seconds)
    simulate_clustering()

    # Phase 3: Training (11 seconds)
    simulate_training()

    # Summary (2 seconds display)
    print_summary()
    time.sleep(2)

    print(f"{Colors.WHITE}Total time: ~20 seconds{Colors.RESET}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Training interrupted by user{Colors.RESET}\n")
        sys.exit(0)
