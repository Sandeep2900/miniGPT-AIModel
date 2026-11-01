"""
Calculate Word Validity Baseline for Exp A Checkpoint.
Evaluates the percentage of generated words found in data/train.txt.
Saves metrics to baseline_metrics.json.
"""

import os
import sys
import json
import string
import re
import torch
from eval_experiments import generate_with_seed
from generate import load_model_and_tokenizer

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PUNCTUATION_TO_STRIP = string.punctuation + "‘’“”–—…।॥\n\r\t "


def extract_words(text: str):
    """
    Extracts words by whitespace splitting and stripping punctuation.
    Lowercases Latin words and preserves Devanagari.
    Categorizes into Devanagari vs Latin words.
    """
    raw_tokens = text.split()
    dev_words = []
    latin_words = []

    for t in raw_tokens:
        clean = t.strip(PUNCTUATION_TO_STRIP)
        if not clean:
            continue

        has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in clean)
        has_latin = any(('a' <= c <= 'z') or ('A' <= c <= 'Z') for c in clean)

        if has_devanagari:
            dev_words.append(clean)
        elif has_latin:
            latin_words.append(clean.lower())

    return dev_words, latin_words


# 1. Build word sets from data/train.txt
with open(os.path.join("data", "train.txt"), "r", encoding="utf-8") as f:
    train_text = f.read()

train_dev_words_list, train_latin_words_list = extract_words(train_text)
train_dev_set = set(train_dev_words_list)
train_latin_set = set(train_latin_words_list)

print("=" * 70)
print("📚 Training Set Lexicon Stats (data/train.txt):")
print(f"  • Total Devanagari word tokens: {len(train_dev_words_list):,d} | Unique vocabulary: {len(train_dev_set):,d}")
print(f"  • Total Latin word tokens     : {len(train_latin_words_list):,d} | Unique vocabulary: {len(train_latin_set):,d}")
print("=" * 70)

# 2. Load Exp A Checkpoint
ckpt_path = os.path.join("checkpoints", "best_model.pt")
model, tokenizer, cfg, step, val_loss = load_model_and_tokenizer(ckpt_path)

prompts = ["आज सुबह ", "Aaj subah ", "The morning "]
num_seeds = 20
max_tokens = 200
temperature = 0.7

all_gen_dev = []
all_gen_latin = []
dev_valid_count = 0
latin_valid_count = 0
invalid_dev_words = []
invalid_latin_words = []

prompt_stats = {}

print(f"\n🔬 Generating 20 samples per prompt (seeds 1-20, temp {temperature}, 200 tokens)...")

for p in prompts:
    p_dev_total = 0
    p_dev_valid = 0
    p_latin_total = 0
    p_latin_valid = 0

    for s in range(1, num_seeds + 1):
        sample = generate_with_seed(
            model,
            tokenizer,
            prompt=p,
            seed=s,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=40,
            top_p=0.9
        )
        # Extract generated words
        dev_w, latin_w = extract_words(sample)

        for w in dev_w:
            all_gen_dev.append(w)
            p_dev_total += 1
            if w in train_dev_set:
                dev_valid_count += 1
                p_dev_valid += 1
            else:
                invalid_dev_words.append(w)

        for w in latin_w:
            all_gen_latin.append(w)
            p_latin_total += 1
            if w in train_latin_set:
                latin_valid_count += 1
                p_latin_valid += 1
            else:
                invalid_latin_words.append(w)

    prompt_stats[p] = {
        "devanagari": {
            "total_words": p_dev_total,
            "valid_in_train": p_dev_valid,
            "validity_pct": round((p_dev_valid / p_dev_total * 100) if p_dev_total > 0 else 0.0, 2)
        },
        "latin": {
            "total_words": p_latin_total,
            "valid_in_train": p_latin_valid,
            "validity_pct": round((p_latin_valid / p_latin_total * 100) if p_latin_total > 0 else 0.0, 2)
        }
    }

total_dev = len(all_gen_dev)
total_latin = len(all_gen_latin)
dev_pct = (dev_valid_count / total_dev * 100) if total_dev > 0 else 0.0
latin_pct = (latin_valid_count / total_latin * 100) if total_latin > 0 else 0.0

print("\n" + "=" * 70)
print("📊 WORD VALIDITY BASELINE METRICS (Lower Bound vs train.txt):")
print("=" * 70)
print(f"Devanagari Words Validity: {dev_pct:.2f}% ({dev_valid_count:,d} / {total_dev:,d})")
print(f"Latin Words Validity     : {latin_pct:.2f}% ({latin_valid_count:,d} / {total_latin:,d})")

print("\nPer-Prompt Breakdown:")
for p, st in prompt_stats.items():
    print(f"  • Prompt {p!r:16s}:")
    print(f"      Devanagari: {st['devanagari']['validity_pct']:5.2f}% ({st['devanagari']['valid_in_train']}/{st['devanagari']['total_words']})")
    print(f"      Latin     : {st['latin']['validity_pct']:5.2f}% ({st['latin']['valid_in_train']}/{st['latin']['total_words']})")

# Sample 10 invalid words (5 Devanagari + 5 Latin, or first 10 unique)
unique_invalid_dev = list(dict.fromkeys(invalid_dev_words))
unique_invalid_latin = list(dict.fromkeys(invalid_latin_words))

sample_invalid = []
for w in unique_invalid_dev[:5]:
    sample_invalid.append(f"{w} (Devanagari)")
for w in unique_invalid_latin[:5]:
    sample_invalid.append(f"{w} (Latin)")
# If less than 10, fill from whichever has more
while len(sample_invalid) < 10 and len(unique_invalid_latin) > 5:
    sample_invalid.append(f"{unique_invalid_latin[len(sample_invalid)]} (Latin)")

print("\n🔍 10 Sample Invalid Words (Generated but NOT in train.txt):")
for idx, w in enumerate(sample_invalid[:10], 1):
    print(f"  {idx:2d}. {w}")

# Save to baseline_metrics.json
metrics_payload = {
    "checkpoint": ckpt_path,
    "model_step": step,
    "val_loss": val_loss,
    "total_samples_evaluated": len(prompts) * num_seeds,
    "overall_devanagari_validity_pct": round(dev_pct, 2),
    "devanagari_valid_words": dev_valid_count,
    "devanagari_total_words": total_dev,
    "overall_latin_validity_pct": round(latin_pct, 2),
    "latin_valid_words": latin_valid_count,
    "latin_total_words": total_latin,
    "per_prompt_breakdown": prompt_stats,
    "sample_invalid_words": sample_invalid[:10],
    "note": "Lower bound metric: legitimate dictionary words not present in train.txt vocabulary are also counted as invalid. Useful strictly for relative before/after comparison."
}

out_json = "baseline_metrics.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

print(f"\n💾 Baseline metrics successfully saved to: {out_json}")
print("=" * 70)
