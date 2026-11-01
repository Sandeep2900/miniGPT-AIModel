"""
Phase 4: Sanity Checks and Debugging Script.
Runs the golden "Overfit a Single Batch" test to verify backpropagation and optimization.
Validates initial loss against theoretical ln(vocab_size) and checks for silent DL bugs.
"""

import sys
import math
import torch
from config import Config
from model import CharGRUModel
from data import load_dataset_splits, get_batch

# Ensure safe UTF-8 terminal output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_overfit_single_batch_test():
    print("=" * 65)
    print("🧪 Sanity Check 1: Overfit a Single Tiny Batch Test")
    print("=" * 65)
    print("Intuition:")
    print("  Agar model 1 single batch ko memorize karke loss ~0 nahi kar pa raha,")
    print("  to backpropagation, optimizer ya weight initialization me koi bug hai.")
    print("  Agar loss 0.1 se kam chala jata hai, to neural network 100% healthy hai!\n")

    # Lightweight configuration for rapid overfitting
    cfg = Config.get_preset("tiny")
    train_data, _, tokenizer = load_dataset_splits()
    cfg.vocab_size = tokenizer.vocab_size

    model = CharGRUModel(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Pick 1 fixed tiny batch
    torch.manual_seed(42)
    fixed_x, fixed_y = get_batch(train_data, batch_size=2, block_size=16, device="cpu")

    print(f"Fixed batch input sequence  : {tokenizer.decode(fixed_x[0].tolist())!r}")
    print(f"Fixed batch target sequence : {tokenizer.decode(fixed_y[0].tolist())!r}")
    print("-" * 65)

    model.train()
    initial_loss = None
    final_loss = None

    for step in range(120):
        logits, loss, _ = model(fixed_x, targets=fixed_y)
        
        if step == 0:
            initial_loss = loss.item()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 20 == 0 or step == 0:
            print(f"Step {step+1:3d} / 120 | Loss: {loss.item():.4f}")

    final_loss = loss.item()
    print("-" * 65)
    print(f"Initial Loss : {initial_loss:.4f}")
    print(f"Final Loss   : {final_loss:.4f}")

    if final_loss < 0.20:
        print("✅ [PASS] Overfit test passed! Neural network architecture & backprop are solid.")
    else:
        print("❌ [FAIL] Loss did not drop close to zero. Check optimizer and learning rate.")


def run_initial_loss_check():
    print("\n" + "=" * 65)
    print("🔍 Sanity Check 2: Initial Untrained Loss Theoretical Check")
    print("=" * 65)
    cfg = Config.get_preset("small")
    train_data, _, tokenizer = load_dataset_splits()
    cfg.vocab_size = tokenizer.vocab_size

    model = CharGRUModel(cfg)
    x, y = get_batch(train_data, batch_size=8, block_size=32, device="cpu")
    
    with torch.no_grad():
        _, loss, _ = model(x, targets=y)

    actual_loss = loss.item()
    expected_loss = math.log(cfg.vocab_size)
    diff = abs(actual_loss - expected_loss)

    print(f"Vocabulary Size (V)      : {cfg.vocab_size}")
    print(f"Theoretical ln(V)        : {expected_loss:.4f}")
    print(f"Actual Initial Loss      : {actual_loss:.4f}")
    print(f"Absolute Difference      : {diff:.4f}")

    if diff < 0.25:
        print("✅ [PASS] Weights initialized with uniform entropy (ln(V)). No initial bias.")
    else:
        print("⚠️ [WARNING] Loss deviates significantly from ln(V). Check weight initialization.")


def print_silent_bugs_checklist():
    print("\n" + "=" * 65)
    print("📋 Common Silent Bugs Checklist (How to Detect & Prevent):")
    print("=" * 65)
    checklist = [
        ("Wrong Shift of y", "y must be x[1:] and not x itself", "data.py visually aligns each step"),
        ("Forgot zero_grad()", "Gradients accumulate and explode to NaN", "optimizer.zero_grad(set_to_none=True)"),
        ("Forgot model.eval()", "Dropout remains active during validation", "estimate_loss switches to model.eval()"),
        ("Loss Reshape Bug", "CrossEntropy needs (N, C) logits and (N) targets", "logits.view(-1, V) and targets.view(-1)"),
        ("Device Mismatch", "Tensors on CPU while model on CUDA", "Everything explicitly routed to device")
    ]
    for bug, desc, fix in checklist:
        print(f"  • {bug:22s} : {desc} (Solved: {fix})")
    print("=" * 65)


if __name__ == "__main__":
    run_overfit_single_batch_test()
    run_initial_loss_check()
    print_silent_bugs_checklist()
