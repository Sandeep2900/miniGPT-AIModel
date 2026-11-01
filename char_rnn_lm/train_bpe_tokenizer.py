"""
train_bpe_tokenizer.py
Trains a custom Byte-Level BPE Tokenizer from scratch on our multilingual dataset.
Saves tokenizer to data/bpe_tokenizer.json.
Guarantees 0% unknown tokens, compresses text into meaningful words/subwords.
"""

import os
import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUTPUT_TOKENIZER_FILE = os.path.join("data", "bpe_tokenizer.json")
TRAIN_FILE = os.path.join("data", "train.txt")
VAL_FILE = os.path.join("data", "val.txt")
VOCAB_SIZE = 2048

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<sep>"]


def train_custom_bpe(vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    print("=" * 70)
    print("🛠️ Training Custom Byte-Level BPE Tokenizer from Scratch")
    print(f"Target Vocabulary Size: {vocab_size}")
    print(f"Special Tokens        : {SPECIAL_TOKENS}")
    print("=" * 70)

    # Initialize BPE model with unknown token
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet()
    )

    training_files = [TRAIN_FILE]
    if os.path.exists(VAL_FILE):
        training_files.append(VAL_FILE)

    print(f"Training on files: {training_files}")
    tokenizer.train(training_files, trainer)

    os.makedirs(os.path.dirname(OUTPUT_TOKENIZER_FILE), exist_ok=True)
    tokenizer.save(OUTPUT_TOKENIZER_FILE)
    print(f"✅ Tokenizer saved successfully to: {OUTPUT_TOKENIZER_FILE}")

    # Verification Tests
    test_phrases = [
        "User: Hello! How are you doing today?",
        "Assistant: I am doing great, thank you! How can I help you?",
        "User: नमस्ते! आप कैसे हैं?",
        "Assistant: नमस्ते! मैं बिल्कुल ठीक हूँ।",
        "User: Aaj ka mausam kaisa hai?",
        "Assistant: Aaj mausam bahut accha aur suhana hai."
    ]

    print("\n🔍 Verification & Compression Tests:")
    print("-" * 70)
    for phrase in test_phrases:
        encoded = tokenizer.encode(phrase)
        decoded = tokenizer.decode(encoded.ids)
        tokens_preview = [tokenizer.id_to_token(i) for i in encoded.ids[:8]]
        print(f"Original : {phrase}")
        print(f"Tokens ({len(encoded.ids)} tokens, chars: {len(phrase)}): {tokens_preview}...")
        assert decoded == phrase, f"Decode mismatch: '{decoded}' != '{phrase}'"
        print("  -> Round-trip exact match: ✅ PASS\n")

    return tokenizer


if __name__ == "__main__":
    train_custom_bpe()
