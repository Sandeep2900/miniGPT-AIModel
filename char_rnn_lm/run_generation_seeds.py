import os
import sys
import torch
from eval_experiments import generate_with_seed
from generate import load_model_and_tokenizer

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

best_ckpt = "checkpoints/exp_a/best_model.pt"
model, tokenizer, cfg, step, val_loss = load_model_and_tokenizer(best_ckpt)

prompts = ["आज सुबह ", "Aaj subah ", "The morning "]
seeds = [1, 2, 3]

print("=" * 75)
print(f"✨ GENERATION TESTS: Best Model (Exp A @ Step {step}, Val Loss: {val_loss:.4f})")
print("Settings: Temperature = 0.7, Max New Tokens = 150, Top-K = 40, Top-P = 0.9")
print("=" * 75)

for p in prompts:
    print(f"\n{'#' * 70}")
    print(f"PROMPT: {p!r}")
    print(f"{'#' * 70}")
    for s in seeds:
        text = generate_with_seed(
            model,
            tokenizer,
            prompt=p,
            seed=s,
            max_new_tokens=150,
            temperature=0.7,
            top_k=40,
            top_p=0.9
        )
        print(f"\n--- [Seed {s}] ---")
        print(text)
