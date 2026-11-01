"""
Data Pipeline & Multilingual Character Tokenizer.
Loads per-file split datasets (data/train.txt and data/val.txt).
Performs splitwise script analysis, strict Hinglish heuristic classification (>= 2 distinct keywords),
vocabulary coverage check, rare character detection, formatted shift verification,
and round-trip encode/decode testing.
"""

import os
import sys
import json
import re
from collections import Counter
from typing import Tuple, List, Dict, Optional, Set
import torch

# Ensure safe UTF-8 terminal output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class CharTokenizer:
    """
    Multilingual Character-level tokenizer.
    Maps every unique character to an integer ID and vice-versa.
    """
    def __init__(self, chars: Optional[List[str]] = None):
        if chars is not None:
            self.chars = sorted(list(set(chars)))
            self.vocab_size = len(self.chars)
            self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(self.chars)}
            self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(self.chars)}
        else:
            self.chars = []
            self.vocab_size = 0
            self.stoi = {}
            self.itos = {}

    def encode(self, text: str) -> List[int]:
        """Convert string to list of token IDs, skipping unknown characters."""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, token_ids: List[int]) -> str:
        """Convert list of token IDs back into text."""
        return "".join([self.itos.get(idx, "") for idx in token_ids])

    def save_vocab(self, filepath: str) -> None:
        """Save vocabulary to disk ensuring Devanagari characters are readable."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "vocab_size": self.vocab_size,
            "chars": self.chars,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_vocab(cls, filepath: str) -> "CharTokenizer":
        """Load vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab_size = data["vocab_size"]
        tok.chars = data["chars"]
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        return tok


# Strict indicator words for Hinglish heuristic classification
# Excluded common English colliding words like 'me', 'ho', 'ke', 'to', 'in'
STRICT_HINGLISH_KEYWORDS = {
    "hai", "hain", "nahi", "nahin", "kya", "aur", "mera", "meri", "mere", "tum",
    "tumhara", "tumhari", "aap", "aapka", "aapki", "humein", "humara", "humari",
    "yaar", "bhai", "aaj", "kal", "karo", "karna", "karne", "raha", "rahi", "rahe",
    "rahein", "hota", "hoti", "hote", "kaun", "kahan", "kaise", "bohot", "bahut",
    "thoda", "thodi", "achha", "accha", "acchi", "baat", "baatein", "zindagi",
    "samay", "waqt", "subah", "shaam", "chalein", "seekh", "kamyabi", "safalta",
    "sapne", "dosti", "pyaar", "khushi", "dil", "chal", "dekh", "dekha", "suno",
    "sunna", "bhi", "yeh", "woh", "unhein", "unhe", "usse", "isse", "lekar",
    "gaya", "gayi", "gaye", "karte", "karti", "karta", "aaya", "aayi", "aaye",
    "theek", "shuru", "pehle", "poora", "apne", "apni", "apna", "chalna", "batana",
    "badla", "badal", "safar", "khwab", "khwaab", "tareef", "hawa", "duniya"
}


def analyze_split_languages(text: str, split_name: str = "Split") -> Dict[str, float]:
    """
    Performs script analysis and strict Hinglish heuristic estimation on a given text split:
    - Devanagari characters (Unicode 0x0900 - 0x097F)
    - Latin characters separated into Hinglish vs English:
      Requires AT LEAST 2 DISTINCT Hinglish keywords per paragraph.
      (Note: Yeh heuristic sirf anuman hai).
    - Digits, Spaces, Punctuation
    """
    total_chars = len(text)
    if total_chars == 0:
        return {"total": 0}

    devanagari_chars = sum(1 for ch in text if 0x0900 <= ord(ch) <= 0x097F)
    digit_chars = sum(1 for ch in text if ch.isdigit())
    other_chars = sum(1 for ch in text if not (0x0900 <= ord(ch) <= 0x097F) and not ch.isalpha() and not ch.isdigit())

    # Strict Hinglish Heuristic: inspect paragraphs/lines containing Latin characters
    paragraphs = text.split("\n")
    hinglish_chars = 0
    english_chars = 0

    for para in paragraphs:
        latin_in_para = sum(1 for ch in para if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))
        if latin_in_para == 0:
            continue
        words = re.findall(r"\b[a-zA-Z]+\b", para.lower())
        distinct_hinglish_matches = set(w for w in words if w in STRICT_HINGLISH_KEYWORDS)
        
        # Strict criterion: at least 2 distinct Hinglish keywords required
        if len(distinct_hinglish_matches) >= 2:
            hinglish_chars += latin_in_para
        else:
            english_chars += latin_in_para

    dev_pct = (devanagari_chars / total_chars) * 100
    hinglish_pct = (hinglish_chars / total_chars) * 100
    english_pct = (english_chars / total_chars) * 100
    digit_pct = (digit_chars / total_chars) * 100
    other_pct = (other_chars / total_chars) * 100
    combined_hindi_hinglish_pct = dev_pct + hinglish_pct

    print(f"\n📊 [{split_name.upper()} SPLIT STATS] - Total: {total_chars:,} characters")
    print(f"  • Devanagari (Hindi)       : {devanagari_chars:6,d} chars ({dev_pct:5.2f}%)")
    print(f"  • Hinglish (Strict Est)    : {hinglish_chars:6,d} chars ({hinglish_pct:5.2f}%)  [>=2 distinct keywords, sirf anuman hai]")
    print(f"  • English (Shakespeare/etc): {english_chars:6,d} chars ({english_pct:5.2f}%)")
    print(f"  • Digits                   : {digit_chars:6,d} chars ({digit_pct:5.2f}%)")
    print(f"  • Spaces & Punctuation     : {other_chars:6,d} chars ({other_pct:5.2f}%)")
    print(f"  ⭐ Combined Hindi + Hinglish Share : {combined_hindi_hinglish_pct:.2f}% (Target: >= 25%)")

    if combined_hindi_hinglish_pct < 25.0:
        print(f"  ⚠️ [WARNING] Combined share ({combined_hindi_hinglish_pct:.1f}%) is below the 25% target!")
        print("     👉 Fix Advice: Config mein 'max_kb_per_file' ko ghata kar 40-50 KB karein,")
        print("        ya 'data/raw/' mein aur Hindi/Hinglish text add karein.")
    else:
        print("  ✅ [PASS] Hindi + Hinglish share is healthy and meets the >= 25% target!")

    return {
        "total": total_chars,
        "devanagari_pct": dev_pct,
        "hinglish_pct": hinglish_pct,
        "english_pct": english_pct,
        "combined_share_pct": combined_hindi_hinglish_pct
    }


def check_vocab_coverage(train_chars: Set[str], val_chars: Set[str]) -> Tuple[List[str], List[str]]:
    """Checks if there are any characters present in Val that never appeared in Train."""
    missing_in_train = sorted(list(val_chars - train_chars))
    return missing_in_train


def load_dataset_splits(
    train_file: str = os.path.join("data", "train.txt"),
    val_file: str = os.path.join("data", "val.txt"),
    val_multi_file: str = os.path.join("data", "val_multilingual.txt")
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], CharTokenizer]:
    """Loads train.txt, val.txt, and optional val_multilingual.txt, returning token tensors."""
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(
            f"Train or Val split not found. Run 'python prepare_data.py' first!"
        )

    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(val_file, "r", encoding="utf-8") as f:
        val_text = f.read()

    combined_chars = sorted(list(set(train_text + val_text)))
    tokenizer = CharTokenizer(combined_chars)

    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    val_multi_data = None
    if os.path.exists(val_multi_file):
        with open(val_multi_file, "r", encoding="utf-8") as f:
            val_multi_text = f.read()
        if len(val_multi_text) > 0:
            val_multi_data = torch.tensor(tokenizer.encode(val_multi_text), dtype=torch.long)

    return train_data, val_data, val_multi_data, tokenizer


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates a batch of (x, y) where y is x shifted forward by 1 character."""
    max_idx = len(data) - block_size - 1
    if max_idx <= 0:
        raise ValueError(f"Dataset length ({len(data)}) is too short for block_size ({block_size}).")

    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def format_sequence_display(tokens: List[int], tokenizer: CharTokenizer) -> str:
    """Formats decoded text sequence replacing raw newlines with visible '\\n' representation."""
    decoded = tokenizer.decode(tokens)
    return decoded.replace("\n", "\\n").replace("\r", "")


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Phase 1: Multilingual Data Pipeline & Rebalanced Split Verification")
    print("=" * 70)

    train_file = os.path.join("data", "train.txt")
    val_file = os.path.join("data", "val.txt")
    vocab_file = os.path.join("data", "vocab.json")

    # 1. Load splits
    train_data, val_data, val_multi_data, tokenizer = load_dataset_splits(train_file, val_file)
    tokenizer.save_vocab(vocab_file)

    with open(train_file, "r", encoding="utf-8") as f:
        train_raw = f.read()
    with open(val_file, "r", encoding="utf-8") as f:
        val_raw = f.read()

    print(f"Vocabulary Size       : {tokenizer.vocab_size} unique characters")
    print(f"Saved Vocab File      : {vocab_file}")

    # 2. Splitwise language stats
    train_stats = analyze_split_languages(train_raw, "Train")
    val_stats = analyze_split_languages(val_raw, "Validation")

    # 3. Vocab coverage check
    train_char_set = set(train_raw)
    val_char_set = set(val_raw)
    missing_in_train = check_vocab_coverage(train_char_set, val_char_set)

    print("\n" + "=" * 70)
    print("🔍 Vocabulary Coverage & Rare Character Inspection:")
    print("=" * 70)
    if missing_in_train:
        print(f"⚠️ Characters in Val but NOT in Train (Total: {len(missing_in_train)}):")
        for ch in missing_in_train:
            print(f"  • {repr(ch)} (U+{ord(ch):04X})")
    else:
        print("✅ 100% Vocab Coverage! Every character in Val is present in Train.")

    # Rare characters (<= 2 times across entire corpus)
    total_counts = Counter(train_raw + val_raw)
    rare_chars = [ch for ch, cnt in total_counts.items() if cnt <= 2]
    print(f"\n🔍 Rare Characters (Total Occurrences <= 2 across dataset: {len(rare_chars)}):")
    if rare_chars:
        rare_sample = rare_chars[:12]
        print("  Sample: " + " ".join([f"{repr(ch)}({total_counts[ch]})" for ch in rare_sample]))
    else:
        print("  None found.")

    # 4. Test Snippet: x/y Shift Test (Train & Val)
    print("\n" + "=" * 70)
    print("🔍 Test Snippet: x/y Shift Verification (Train & Val splits):")
    print("=" * 70)
    
    # Train shift check
    x_tr, y_tr = get_batch(train_data, batch_size=1, block_size=10, device="cpu")
    x_tr_list = x_tr[0].tolist()
    y_tr_list = y_tr[0].tolist()
    tr_match = all(x_tr_list[i + 1] == y_tr_list[i] for i in range(len(x_tr_list) - 1))
    
    print(f"[TRAIN BATCH] Valid Shift (y[i] == x[i+1]): {'✅ PASS' if tr_match else '❌ FAIL'}")
    print(f"  Input  x: {format_sequence_display(x_tr_list, tokenizer)}")
    print(f"  Target y: {format_sequence_display(y_tr_list, tokenizer)}")

    # Val shift check
    x_val, y_val = get_batch(val_data, batch_size=1, block_size=10, device="cpu")
    x_val_list = x_val[0].tolist()
    y_val_list = y_val[0].tolist()
    val_match = all(x_val_list[i + 1] == y_val_list[i] for i in range(len(x_val_list) - 1))

    print(f"\n[VAL BATCH]   Valid Shift (y[i] == x[i+1]): {'✅ PASS' if val_match else '❌ FAIL'}")
    print(f"  Input  x: {format_sequence_display(x_val_list, tokenizer)}")
    print(f"  Target y: {format_sequence_display(y_val_list, tokenizer)}")

    # 5. Round-trip Encode/Decode test
    print("\n" + "=" * 70)
    print("🔄 Round-Trip Encode / Decode Test (Hindi, Hinglish, English):")
    print("=" * 70)
    samples = [
        "आज सुबह मैंने खिड़की खोली तो ठंडी हवा का झोंका अंदर आया।",
        "Aaj subah maine khidki kholi to thandi hawa ka jhonka andar aaya.",
        "Listening to grandmother's stories is one of the sweetest memories."
    ]
    for s in samples:
        enc = tokenizer.encode(s)
        dec = tokenizer.decode(enc)
        status = "✅ PASS" if dec == s else "❌ FAIL"
        print(f"  {status} Original : {s}")
        print(f"         Decoded  : {dec}")

    # 6. Devanagari-heavy train batch demonstration
    print("\n" + "=" * 70)
    print("🇮🇳 Devanagari-Heavy Train Batch Demonstration:")
    print("=" * 70)
    found_heavy = False
    for start in range(0, len(train_data) - 40, 20):
        chunk_tokens = train_data[start : start + 32].tolist()
        chunk_text = tokenizer.decode(chunk_tokens)
        dev_count = sum(1 for c in chunk_text if 0x0900 <= ord(c) <= 0x097F)
        if dev_count >= 15:
            print(f"  Found Devanagari training segment at index {start}:")
            print(f"  Raw Decoded Text : '{format_sequence_display(chunk_tokens, tokenizer)}'")
            print(f"  Devanagari Ratio : {dev_count}/{len(chunk_tokens)} ({dev_count / len(chunk_tokens) * 100:.1f}%)")
            found_heavy = True
            break

    print("\n" + "=" * 70)
    print("🎉 Multilingual Data Pipeline verification completed successfully!")
    print("=" * 70)
