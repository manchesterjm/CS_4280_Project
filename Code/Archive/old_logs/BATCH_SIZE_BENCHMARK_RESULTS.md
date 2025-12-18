# Batch Size Benchmark Results

---

## RTX 5070 Ti (16 GB VRAM) - December 2, 2025

**Date:** December 2, 2025
**GPU:** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
**Dataset:** TESS Sector 1 Ground Truth (26,472 training samples)
**Test Duration:** 20 seconds per batch size

### Coarse Benchmark (Powers of 2)

| Batch Size | Batches/Epoch | Time/Batch (sec) | Est. Epoch Time (min) | Notes |
|------------|---------------|------------------|----------------------|-------|
| 8 | 3,309 | 0.389 | 21.48 | Too many iterations |
| 16 | 1,655 | 0.404 | 11.15 | |
| 32 | 828 | 0.418 | 5.77 | |
| 64 | 414 | 0.442 | 3.05 | |
| 128 | 207 | 0.514 | 1.77 | Near optimal |
| 256 | 104 | 8.666 | 15.02 | Memory pressure |
| 512 | - | OOM | ❌ | Crash |

### Fine-Grained Benchmark (112-144)

| Batch Size | Batches/Epoch | Time/Batch (sec) | Est. Epoch Time (min) | Notes |
|------------|---------------|------------------|----------------------|-------|
| 112 | 237 | 0.506 | 2.00 | |
| 120 | 221 | 0.506 | 1.86 | |
| 128 | 207 | 0.517 | 1.78 | |
| **136** | **195** | **0.540** | **1.75** | **OPTIMAL** |
| 144 | 184 | 0.580 | 1.78 | |

### Key Findings (RTX 5070 Ti)

1. **Optimal Batch Size: 136** (1.75 min/epoch)
   - ~1.8× faster than RTX 3060 Ti (was 3.15 min/epoch with batch 64)

2. **Sweet Spot: 128-144**
   - All within 2% of optimal (1.75-1.78 min/epoch)
   - Any batch size in this range is acceptable

3. **GPU Memory Cliff at 256**
   - Time/batch jumps from 0.58 sec (batch 144) to 8.67 sec (batch 256)
   - Batch 512 causes OOM crash

4. **Training Time Estimates (60 epochs):**
   | Batch Size | Training Time |
   |------------|---------------|
   | 136 (optimal) | **1.75 hours** |
   | 128-144 (sweet spot) | 1.78-1.80 hours |
   | 256 (avoid) | 15 hours |

---

## RTX 3060 Ti (8 GB VRAM) - November 29, 2025 (OLD SYSTEM)

**Date:** November 29, 2025
**GPU:** NVIDIA GeForce RTX 3060 Ti (8 GB VRAM)
**Dataset:** TESS Sector 1 Ground Truth (26,472 training samples)
**Test Duration:** 30 seconds per batch size

## Results (Full Granularity - Increments of 8)

| Batch Size | Batches/Epoch | Time/Batch (sec) | Est. Epoch Time (min) | Notes |
|------------|---------------|------------------|----------------------|-------|
| 8 | 3309 | 0.293 | 16.15 | Too many iterations |
| 16 | 1655 | 0.362 | 9.97 | |
| 24 | 1103 | 0.340 | 6.24 | |
| 32 | 828 | 0.363 | 5.00 | |
| 40 | 662 | 0.387 | 4.27 | |
| 48 | 552 | 0.411 | 3.78 | |
| 56 | 473 | 0.423 | 3.33 | |
| **64** | **414** | **0.457** | **3.15** | **OPTIMAL** |
| 72 | 368 | 0.530 | 3.25 | Sweet spot |
| 80 | 331 | 0.600 | 3.31 | Sweet spot |
| 88 | 301 | 0.655 | 3.29 | Sweet spot |
| 96 | 276 | 0.799 | 3.67 | Starting to slow |
| 104 | 255 | 1.796 | 7.63 | GPU memory cliff |
| 112 | 237 | 3.754 | 14.83 | Memory pressure |
| 120 | 221 | 4.293 | 15.81 | Memory pressure |
| 128 | 207 | 5.677 | 19.59 | Severe bottleneck |

## Key Findings

1. **Optimal Batch Size: 64** (3.15 min/epoch)
   - Best balance of GPU utilization and iteration count

2. **Sweet Spot: 56-88**
   - All within 5% of optimal (3.15-3.33 min/epoch)
   - Any batch size in this range is acceptable

3. **GPU Memory Cliff at 104**
   - Time/batch jumps from 0.8 sec (batch 96) to 1.8 sec (batch 104)
   - 2.25× slowdown indicates 8 GB VRAM limit hit
   - Batch 128 is 12× slower than batch 64

4. **Scaling Pattern**
   - Batch 8→64: Linear improvement (fewer iterations wins)
   - Batch 64→96: Slight degradation (GPU saturated)
   - Batch 104+: Severe degradation (memory pressure)

## Recommendations

### For RTX 3060 Ti (8 GB VRAM):
- **Use batch size 64** for fastest training
- Acceptable range: 56-88
- **AVOID batch sizes ≥104** (memory bottleneck)

### Training Time Estimates (60 epochs):
| Batch Size | Training Time |
|------------|---------------|
| 64 (optimal) | **3.15 hours** |
| 56-88 (sweet spot) | 3.3-3.5 hours |
| 128 (avoid) | 19.6 hours |

## For Future GPU (RTX 5070 Ti - 16 GB VRAM)

With double the VRAM:
- Memory cliff should shift from ~100 to ~200
- Batch size 128-192 likely optimal
- Expected epoch time: ~1-1.5 min (2-3× faster)

Re-run this benchmark on new hardware to find optimal batch size.

## Script Used

`benchmark_batch_sizes.py` - Tests batch sizes 8-128 in increments of 8, 30 seconds each.
