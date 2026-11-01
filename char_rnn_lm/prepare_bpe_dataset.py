"""
prepare_bpe_dataset.py
Tokenizes data/train.txt and data/val.txt using data/bpe_tokenizer.json.
Saves token IDs directly into PyTorch binary tensor files for high-speed training.
"""

import os
import sys
import torch
from tokenizers import Tokenizer

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TOKENIZER_PATH = os.path.join("data", "bpe_tokenizer.json")
TRAIN_TXT = os.path.join("data", "train.txt")
VAL_TXT = os.path.join("data", "val.txt")
TRAIN_BIN = os.path.join("data", "train_bpe.pt")
VAL_BIN = os.path.join("data", "val_bpe.pt")


def main():
    print("=" * 70)
    print("⚡ Pre-tokenizing Multilingual Dataset with BPE Tokenizer")
    print("=" * 70)

    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: {TOKENIZER_PATH} not found. Run train_bpe_tokenizer.py first!")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.get_vocab_size():,d}")

    # Process Train Split
    with open(TRAIN_TXT, "r", encoding="utf-8", errors="ignore") as f:
        train_text = f.read()

    print(f"Encoding Train Split ({len(train_text):,d} characters)...")
    train_ids = tokenizer.encode(train_text).ids
    train_tensor = torch.tensor(train_ids, dtype=torch.long)
    torch.save(train_tensor, TRAIN_BIN)
    compression_ratio = len(train_text) / max(1, len(train_ids))
    print(f"✅ Saved {TRAIN_BIN}: {len(train_ids):,d} tokens ({compression_ratio:.2f} chars/token)")

    # Process Val Split
    with open(VAL_TXT, "r", encoding="utf-8", errors="ignore") as f:
        val_text = f.read()

    print(f"Encoding Val Split ({len(val_text):,d} characters)...")
    val_ids = tokenizer.encode(val_text).ids
    val_tensor = torch.tensor(val_ids, dtype=torch.long)
    torch.save(val_tensor, VAL_BIN)
    print(f"✅ Saved {VAL_BIN}: {len(val_ids):,d} tokens")
    print("=" * 70)


if __name__ == "__main__":
    main()
