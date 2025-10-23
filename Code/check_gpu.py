"""
Check GPU availability and configuration
"""
import torch

print("="*70)
print(" GPU CONFIGURATION CHECK")
print("="*70)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Compute capability: {torch.cuda.get_device_capability(i)}")
    
    print(f"\nCurrent device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Test GPU
    print("\nTesting GPU...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print("✓ GPU test successful!")
    
else:
    print("\n⚠️  No GPU available - will use CPU")
    print(f"CPU threads: {torch.get_num_threads()}")

print("\n" + "="*70)
