"""
process_dailydialog.py
Downloads (or loads) DailyDialog from KaggleHub and formats it into
natural conversation pairs (User: ... / Assistant: ...).
Saves output to data/raw/dailydialog.txt (UTF-8).
"""

import os
import sys
import csv
import re
import ast

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUTPUT_PATH = os.path.join("data", "raw", "dailydialog.txt")


TURN_PATTERN = re.compile(r"'([^']*)'|\"([^\"]*)\"")


def clean_turn_text(text: str) -> str:
    """Cleans tokenized punctuation spaces common in DailyDialog."""
    t = text.strip()
    t = re.sub(r"\s+([,.:;?!])", r"\1", t)
    t = re.sub(r"\s*'\s*([smdt]|ll|re|ve)\b", r"'\1", t)
    t = re.sub(r"\s+('\w+)", r"\1", t)
    t = re.sub(r'\s+"', r'"', t)
    t = re.sub(r'"\s+', r'"', t)
    return t.strip()


def parse_dialog_string(raw_str: str) -> list:
    """Parses single-quoted and double-quoted utterances from dialog column."""
    tokens = []
    for m in TURN_PATTERN.finditer(raw_str):
        val = m.group(1) if m.group(1) is not None else m.group(2)
        cleaned = clean_turn_text(val)
        if cleaned:
            tokens.append(cleaned)
    return tokens


def main():
    print("=" * 70)
    print("📥 DailyDialog Dataset Processor for Chat LM Training")
    print("=" * 70)

    # 1. Download or get path from kagglehub
    try:
        import kagglehub
        dataset_dir = kagglehub.dataset_download("thedevastator/dailydialog-unlock-the-conversation-potential-in")
        print(f"Dataset directory: {dataset_dir}")
    except Exception as e:
        print(f"Error accessing Kaggle dataset: {e}")
        sys.exit(1)

    train_csv = os.path.join(dataset_dir, "train.csv")
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found!")
        sys.exit(1)

    formatted_conversations = []
    total_turns = 0

    with open(train_csv, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialog_str = row.get("dialog", "")
            turns = parse_dialog_string(dialog_str)
            if len(turns) < 2:
                continue

            conv_lines = []
            for idx, turn in enumerate(turns):
                speaker = "User" if (idx % 2 == 0) else "Assistant"
                conv_lines.append(f"{speaker}: {turn}")

            formatted_conversations.append("\n".join(conv_lines))
            total_turns += len(turns)

    print(f"Extracted {len(formatted_conversations):,d} valid conversations ({total_turns:,d} turns).")

    # Save to data/raw/dailydialog.txt
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    full_text = "\n\n".join(formatted_conversations)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_text)

    file_size_kb = os.path.getsize(OUTPUT_PATH) / 1024.0
    print(f"Saved to: {OUTPUT_PATH} ({file_size_kb:.2f} KB, {len(full_text):,d} characters)")
    print("=" * 70)
    print("Sample Conversation:")
    print("-" * 50)
    print(formatted_conversations[0])
    print("-" * 50)


if __name__ == "__main__":
    main()
