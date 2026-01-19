"""
Script to check GPU availability and CUDA setup
"""
import sys
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import torch

print("="*60)
print("GPU and CUDA Check")
print("="*60)

# Check PyTorch version
print(f"\n1. PyTorch Version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"\n2. CUDA Available: {cuda_available}")

if cuda_available:
    # CUDA version
    print(f"   CUDA Version: {torch.version.cuda}")
    
    # Number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"   Number of GPUs: {num_gpus}")
    
    # GPU information
    for i in range(num_gpus):
        print(f"\n   GPU {i}:")
        print(f"     Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"     Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"     Compute Capability: {props.major}.{props.minor}")
        
        # Check current memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"     Memory Allocated: {allocated:.2f} GB")
            print(f"     Memory Reserved: {reserved:.2f} GB")
    
    # Test GPU computation
    print(f"\n3. Testing GPU computation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("   [OK] GPU computation test: SUCCESS")
        print(f"   Result device: {z.device}")
    except Exception as e:
        print(f"   [ERROR] GPU computation test: FAILED")
        print(f"   Error: {e}")
else:
    print("\n[ERROR] CUDA is not available!")
    print("\nPossible reasons:")
    print("  1. No NVIDIA GPU installed")
    print("  2. CUDA drivers not installed")
    print("  3. PyTorch installed without CUDA support")
    print("\nTo install PyTorch with CUDA support:")
    print("  Visit: https://pytorch.org/get-started/locally/")
    print("  Or run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

# Check if models were trained
print("\n" + "="*60)
print("Checking for trained models...")
print("="*60)

import os
from pathlib import Path

model_paths = [
    "./models/final",
    "./models/checkpoints",
]

for model_path in model_paths:
    path = Path(model_path)
    if path.exists():
        print(f"\n✅ Found: {model_path}")
        
        # Check for adapter (LoRA)
        adapter_config = path / "adapter_config.json"
        if adapter_config.exists():
            print(f"   Type: LoRA Adapter")
            print(f"   Adapter config: [OK]")
        
        # Check for full model
        config_file = path / "config.json"
        if config_file.exists():
            print(f"   Type: Full Model")
            print(f"   Config: [OK]")
        
        # Check for tokenizer
        tokenizer_files = [
            path / "tokenizer.json",
            path / "tokenizer_config.json",
            path / "spiece.model"
        ]
        tokenizer_found = any(f.exists() for f in tokenizer_files)
        if tokenizer_found:
            print(f"   Tokenizer: [OK]")
        else:
            print(f"   Tokenizer: [MISSING]")
        
        # List files
        files = list(path.glob("*"))
        print(f"   Files: {len(files)} files")
    else:
        print(f"\n[NOT FOUND] {model_path}")

print("\n" + "="*60)
print("Check complete!")
print("="*60)

