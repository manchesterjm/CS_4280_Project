"""Analyze false positive rate in SMOTE predictions."""
import pandas as pd

# Load predictions
df = pd.read_csv('reports/smote_true_planet_predictions.csv')

# Get positive predictions
pos = df[df['prediction'] == 1]

print("="*70)
print("FALSE POSITIVE ANALYSIS - SMOTE Model")
print("="*70)

print(f"\nTotal windows tested: {len(df)}")
print(f"Total positive predictions: {len(pos)}")

print("\n" + "="*70)
print("BREAKDOWN BY PHASE CENTER")
print("="*70)

# Count by phase
phase_counts = pos.groupby('phase_center').size()
print("\nPositive predictions by phase:")
for phase, count in phase_counts.items():
    print(f"  phase_center = {phase:5.2f}: {count} predictions")

# True positives vs False positives
# Phase 0.0 = centered on transit = TRUE POSITIVE (if detected)
# Phase != 0.0 = offset from transit = FALSE POSITIVE (if detected)

true_positives = pos[pos['phase_center'] == 0.0]
false_positives = pos[pos['phase_center'] != 0.0]

print("\n" + "="*70)
print("TRUE POSITIVES vs FALSE POSITIVES")
print("="*70)

print(f"\nTrue Positives (phase = 0.0):  {len(true_positives):3d}")
print(f"False Positives (phase != 0.0): {len(false_positives):3d}")
print(f"Total Predictions:               {len(pos):3d}")

print(f"\nFalse Positive Rate: {len(false_positives)/len(pos)*100:.1f}%")
print(f"Precision: {len(true_positives)/len(pos)*100:.1f}%")

print("\n" + "="*70)
print("EXPECTED vs ACTUAL")
print("="*70)

# Expected: 100 stars, should detect ~100 at phase=0.0
total_phase_0 = len(df[df['phase_center'] == 0.0])
print(f"\nWindows at phase = 0.0 (transit center): {total_phase_0}")
print(f"Detected at phase = 0.0: {len(true_positives)}")
print(f"Detection rate on transit-centered windows: {len(true_positives)/total_phase_0*100:.1f}%")

# Off-center windows (should be negative)
total_off_center = len(df[df['phase_center'] != 0.0])
print(f"\nWindows at phase != 0.0 (off-center): {total_off_center}")
print(f"Incorrectly detected (false positives): {len(false_positives)}")
print(f"False positive rate on off-center windows: {len(false_positives)/total_off_center*100:.1f}%")

print("\n" + "="*70)
print("STARS DETECTED")
print("="*70)

# Which stars were detected?
detected_stars = true_positives['tic_id'].unique()
print(f"\nNumber of stars with transit-centered detections: {len(detected_stars)}")
print(f"\nTIC IDs detected at phase = 0.0:")
for tic in sorted(detected_stars):
    print(f"  TIC {tic}")

print("\n" + "="*70)
