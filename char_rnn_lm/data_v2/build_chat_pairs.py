"""
data_v2/build_chat_pairs.py
Pipeline to ingest, clean, sanitize, and split chat pairs for MiniGPT v2 training.
Reads raw user pairs from data/v2/pairs_raw.jsonl.
Cleans empty lines, removes duplicate turns, scrubs phone numbers/emails,
and creates train/val splits strictly BY CONVERSATION (preventing turn leakage).
"""

import os
import sys
import json
import re
import random
from typing import Dict, List, Tuple, Set

# Ensure safe UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DATA_DIR = os.path.join("data", "v2")
RAW_INPUT_FILE = os.path.join(DATA_DIR, "pairs_raw.jsonl")
TRAIN_OUTPUT_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_OUTPUT_FILE = os.path.join(DATA_DIR, "val.jsonl")

# Regular expressions for PII removal
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Matches international formats and standard Indian 10-digit mobile numbers
PHONE_REGEX = re.compile(r"\b(?:\+?91[-.\s]?)?[6-9]\d{9}\b|\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


def sanitize_text(text: str) -> str:
    """Removes emails, phone numbers, and normalizes excess whitespace."""
    if not text:
        return ""
    # Strip PII
    cleaned = EMAIL_REGEX.sub("", text)
    cleaned = PHONE_REGEX.sub("", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"[ \t]+", " ", cleaned).strip()
    return cleaned


def clean_and_group_pairs(input_path: str) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
    """
    Reads pairs_raw.jsonl, validates fields, scrubs PII, deduplicates,
    and groups pairs by conversation_id.
    """
    stats = {
        "raw_lines": 0,
        "valid_pairs": 0,
        "dropped_empty": 0,
        "dropped_duplicates": 0,
        "pii_sanitized": 0
    }

    if not os.path.exists(input_path):
        return {}, stats

    conversations: Dict[str, List[Dict]] = {}
    seen_pairs: Set[Tuple[str, str]] = set()

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            stats["raw_lines"] += 1

            try:
                record = json.loads(line_str)
            except Exception:
                stats["dropped_empty"] += 1
                continue

            raw_user = record.get("user", "")
            raw_assistant = record.get("assistant", "")
            lang = record.get("lang", "en").lower().strip()
            # If no explicit conv_id provided, treat single pair as independent conversation
            conv_id = str(record.get("conv_id") or record.get("conversation_id") or f"conv_{line_num}")

            clean_user = sanitize_text(raw_user)
            clean_assistant = sanitize_text(raw_assistant)

            if len(clean_user) < len(raw_user) or len(clean_assistant) < len(raw_assistant):
                stats["pii_sanitized"] += 1

            # Check for empty content
            if not clean_user or not clean_assistant:
                stats["dropped_empty"] += 1
                continue

            # Deduplication key (case-insensitive)
            dedup_key = (clean_user.lower(), clean_assistant.lower())
            if dedup_key in seen_pairs:
                stats["dropped_duplicates"] += 1
                continue

            seen_pairs.add(dedup_key)

            clean_record = {
                "user": clean_user,
                "assistant": clean_assistant,
                "lang": lang,
                "conv_id": conv_id
            }

            conversations.setdefault(conv_id, []).append(clean_record)
            stats["valid_pairs"] += 1

    return conversations, stats


def split_by_conversation(
    conversations: Dict[str, List[Dict]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Splits data into train and val splits BY CONVERSATION.
    Ensures all turns of a conversation reside in either train or val, never split.
    """
    conv_ids = list(conversations.keys())
    random.seed(seed)
    random.shuffle(conv_ids)

    split_idx = int(len(conv_ids) * train_ratio)
    train_conv_ids = set(conv_ids[:split_idx])
    val_conv_ids = set(conv_ids[split_idx:])

    train_records = []
    val_records = []

    for cid in conv_ids:
        turns = conversations[cid]
        if cid in train_conv_ids:
            train_records.extend(turns)
        else:
            val_records.extend(turns)

    return train_records, val_records


def build_chat_dataset():
    print("=" * 75)
    print("🛠️ MiniGPT v2: Chat Pairs Dataset Ingestion & Cleaning")
    print("=" * 75)

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(RAW_INPUT_FILE):
        print(f"[STATUS] Input file not found: {RAW_INPUT_FILE}")
        print(f"[INFO] Current valid pairs count: 0")
        print(f"[ACTION] Please populate {RAW_INPUT_FILE} with chat pairs.")
        print("Example line format in data/v2/pairs_raw.jsonl:")
        print('  {"user": "Namaste, aap kaise hain?", "assistant": "Namaste! Main badhiya hoon.", "lang": "hi", "conv_id": "c1"}')
        return

    conversations, stats = clean_and_group_pairs(RAW_INPUT_FILE)

    print(f"Total raw lines processed     : {stats['raw_lines']}")
    print(f"Valid cleaned pairs retained  : {stats['valid_pairs']}")
    print(f"Dropped empty/invalid records : {stats['dropped_empty']}")
    print(f"Dropped duplicate pairs       : {stats['dropped_duplicates']}")
    print(f"PII scrubbed occurrences      : {stats['pii_sanitized']}")
    print(f"Total distinct conversations  : {len(conversations)}")
    print("-" * 75)

    if stats["valid_pairs"] == 0:
        print("[WARNING] Zero valid pairs found to split.")
        return

    train_pairs, val_pairs = split_by_conversation(conversations, train_ratio=0.8, seed=42)

    with open(TRAIN_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in train_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(VAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in val_pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Split completed BY CONVERSATION:")
    print(f"  - Train Split : {len(train_pairs)} pairs saved to {TRAIN_OUTPUT_FILE}")
    print(f"  - Val Split   : {len(val_pairs)} pairs saved to {VAL_OUTPUT_FILE}")
    print("=" * 75)


if __name__ == "__main__":
    build_chat_dataset()
