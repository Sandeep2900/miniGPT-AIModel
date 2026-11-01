"""
Ratta-check (Memorization Analysis)
Generates 20 samples (seeds 1 to 20, temp 0.7, 200 tokens) for each prompt.
Evaluates 25-character sliding window match % against data/train.txt and data/val.txt.
"""

import os
import sys
import torch
from generate import load_model_and_tokenizer
from eval_experiments import generate_with_seed

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 1. Load Data
with open(os.path.join("data", "train.txt"), "r", encoding="utf-8") as f:
    train_text = f.read()

with open(os.path.join("data", "val.txt"), "r", encoding="utf-8") as f:
    val_text = f.read()

# 2. Load Model (Exp A @ Step 4000)
ckpt_path = os.path.join("checkpoints", "best_model.pt")
model, tokenizer, cfg, step, val_loss = load_model_and_tokenizer(ckpt_path)

prompts = ["आज सुबह ", "Aaj subah ", "The morning "]
window_size = 25
num_seeds = 20
max_tokens = 200
temperature = 0.7

results = {}
all_samples = []

print("=" * 75)
print(f"🔍 Running Memorization (Ratta) Check on Exp A (Step {step}, Val Loss: {val_loss:.4f})")
print(f"Settings: Window Size = {window_size} chars, 20 Seeds (1-20), Temp = {temperature}, Tokens = {max_tokens}")
print("=" * 75)

for prompt in prompts:
    prompt_train_matches = []
    prompt_val_matches = []
    print(f"\nProcessing prompt: {prompt!r} (20 samples)...")

    for seed in range(1, num_seeds + 1):
        sample_text = generate_with_seed(
            model,
            tokenizer,
            prompt=prompt,
            seed=seed,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.9
        )
        
        # Sliding windows of 25 characters
        total_windows = max(0, len(sample_text) - window_size + 1)
        if total_windows == 0:
            continue

        train_matched = 0
        val_matched = 0
        matching_train_snippets = []

        for i in range(total_windows):
            window = sample_text[i:i + window_size]
            if window in train_text:
                train_matched += 1
                matching_train_snippets.append(window)
            if window in val_text:
                val_matched += 1

        train_pct = (train_matched / total_windows) * 100.0
        val_pct = (val_matched / total_windows) * 100.0

        prompt_train_matches.append(train_pct)
        prompt_val_matches.append(val_pct)

        all_samples.append({
            "prompt": prompt,
            "seed": seed,
            "text": sample_text,
            "train_pct": train_pct,
            "val_pct": val_pct,
            "matched_snippets": matching_train_snippets
        })

    avg_train = sum(prompt_train_matches) / len(prompt_train_matches)
    avg_val = sum(prompt_val_matches) / len(prompt_val_matches)
    results[prompt] = {"avg_train": avg_train, "avg_val": avg_val}
    print(f"  -> Avg 25-char Window Match: Train = {avg_train:.2f}% | Val = {avg_val:.2f}%")

print("\n" + "=" * 75)
print("📊 OVERALL MEMORIZATION (RATTA) RESULTS:")
print("=" * 75)
for prompt, res in results.items():
    print(f"Prompt: {prompt!r:16s} | Train Match: {res['avg_train']:5.2f}% | Val Match: {res['avg_val']:5.2f}%")

# Find the sample with the highest train match %
highest_sample = max(all_samples, key=lambda x: x["train_pct"])
print("\n" + "=" * 75)
print(f"🔥 HIGHEST MATCH SAMPLE across all 60 generations:")
print(f"Prompt: {highest_sample['prompt']!r} | Seed: {highest_sample['seed']}")
print(f"Train Match: {highest_sample['train_pct']:.2f}% | Val Match: {highest_sample['val_pct']:.2f}%")
print("=" * 75)
print("Generated Text:")
print("-" * 75)
print(highest_sample["text"])
print("-" * 75)

# Locate matching lines in train.txt
print("\n🔍 Matching Line(s) in data/train.txt for this sample:")
print("-" * 75)
train_lines = train_text.split("\n")
matched_lines_found = set()

for snippet in highest_sample["matched_snippets"]:
    for line_idx, line in enumerate(train_lines):
        if snippet in line:
            matched_lines_found.add((line_idx + 1, line.strip()))

if matched_lines_found:
    for line_num, line_content in sorted(matched_lines_found):
        print(f"Line {line_num}: {line_content}")
else:
    # If match crosses line boundaries:
    print("Match spans across newline characters in training text.")
print("-" * 75)
