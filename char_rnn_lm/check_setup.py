"""
Environment & Hardware Setup Verification Script
Verifies Python version, PyTorch installation, CPU/GPU status, and key dependencies.
Windows friendly: safe UTF-8 encoding handling.
"""

import sys
import platform

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("=" * 60)
print("[*] AI Model From Scratch - Environment Check")
print("=" * 60)

# 1. Python & Operating System
print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
print(f"Python Version: {platform.python_version()} ({sys.executable})")

# 2. Check PyTorch
try:
    import torch
    print(f"[OK] PyTorch Version: {torch.__version__}")
    
    # Device check
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[OK] CUDA Available: Yes ({device_name})")
    else:
        print("[INFO] CUDA Available: No (Using CPU - Perfect for our Char-GRU model!)")
        
    num_threads = torch.get_num_threads()
    print(f"[INFO] PyTorch CPU Threads: {num_threads}")
    
    # Simple tensor operation test on CPU
    x = torch.randn(3, 3)
    y = torch.matmul(x, x)
    print("[OK] PyTorch Basic Math Check: Passed")
except ImportError:
    print("[ERROR] PyTorch is NOT installed yet.")
    print("👉 Run: pip install torch --index-url https://download.pytorch.org/whl/cpu")

# 3. Check helper libraries
for lib_name in ["numpy", "tqdm", "matplotlib"]:
    try:
        mod = __import__(lib_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK] {lib_name} Version: {version}")
    except ImportError:
        print(f"[ERROR] {lib_name} is NOT installed.")

print("=" * 60)
print("Setup check completed successfully!")
print("=" * 60)
