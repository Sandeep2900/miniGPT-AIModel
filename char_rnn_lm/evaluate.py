"""
Phase 6: Model Evaluation & Multi-Temperature Sampling.
- Computes Train and Validation Cross-Entropy Loss and Perplexity
- Generates 5 sample texts at various temperatures (0.2 to 1.4)
- Explains perplexity interpretation and saves samples to samples.txt
"""

import os
import sys
import math
import torch
from generate import load_model_and_tokenizer, generate_text

# Ensure safe UTF-8 terminal output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def evaluate_model(checkpoint_path: str = os.path.join("checkpoints", "best_model.pt")):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, cfg, step, val_loss = load_model_and_tokenizer(checkpoint_path, device=device)

    # 1. Perplexity calculation
    # Perplexity = exp(loss)
    # Intuition: The effective number of equally likely character choices the model is torn between.
    perplexity = math.exp(val_loss) if val_loss is not None else float("nan")

    print("=" * 65)
    print("📊 Model Evaluation & Perplexity Analysis")
    print("=" * 65)
    print(f"Checkpoint File  : {checkpoint_path}")
    print(f"Trained Steps    : {step}")
    print(f"Validation Loss  : {val_loss:.4f}")
    print(f"Perplexity (PPL) : {perplexity:.2f} (exp({val_loss:.4f}))")
    print("\n🧠 Perplexity Interpretation:")
    print(f"  • Random Model PPL   : ~{cfg.vocab_size} (complete confusion among {cfg.vocab_size} chars)")
    print(f"  • Current Model PPL  : {perplexity:.1f} (on average choosing among {perplexity:.1f} characters)")
    print(f"  • Mature Model Target: 4.0 - 8.0 (on a 130-char multilingual dataset)")

    # 2. Temperature Sweep Comparison
    temperatures = [0.2, 0.5, 0.8, 1.0, 1.3]
    prompt = "आज सुबह "
    samples_output = []

    print("\n" + "=" * 65)
    print("🌡️ Multi-Temperature Sampling Comparison:")
    print("=" * 65)
    
    for temp in temperatures:
        sample = generate_text(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=100,
            temperature=temp,
            top_k=40,
            top_p=0.9,
            device=device
        )
        sample_entry = f"--- [Temperature = {temp:.1f}] ---\nPrompt: {prompt}\nOutput:\n{sample}\n"
        samples_output.append(sample_entry)
        print(f"\n[Temperature = {temp:.1f}]:")
        print(sample.replace("\n", "\n  "))

    # Save to samples.txt
    out_file = "samples.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(samples_output))
    print("\n" + "=" * 65)
    print(f"💾 All samples saved to: {out_file}")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_model()
