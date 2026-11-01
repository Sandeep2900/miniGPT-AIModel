"""
prepare_data.py
1. Auto-detects all .txt files inside data/raw/ (no hard-coding).
2. Downloads Tiny Shakespeare if not present.
3. Rebalances files using max_kb_per_file (caps large files from the start, default 50 KB).
4. Cross-file deduplication BEFORE splitting:
   - Removes lines that already appeared in a previous file (cross-file overlap).
   - Keeps within-file repeated lines intact (e.g. repeated speaker names in Shakespeare).
   - Clearly reports cross-file overlap (removed) vs within-file repeats (kept).
5. Performs PER-FILE 90/10 continuous split so Hindi, Hinglish, and English exist in both Train and Val.
6. Saves data/train.txt, data/val.txt, data/val_multilingual.txt, and data/data.txt.
"""

import os
import sys
import re
from collections import Counter
import urllib.request
from typing import Dict, List, Tuple
from config import Config

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

RAW_DIR = os.path.join("data", "raw")
OUT_DATA = os.path.join("data", "data.txt")
OUT_TRAIN = os.path.join("data", "train.txt")
OUT_VAL = os.path.join("data", "val.txt")
OUT_VAL_MULTI = os.path.join("data", "val_multilingual.txt")
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def clean_whatsapp_text(raw_text: str) -> str:
    """
    Cleans exported WhatsApp chat:
    - Removes system messages: '<Media omitted>', 'Messages and calls are end-to-end encrypted', etc.
    - Strips timestamps and sender names.
    """
    system_patterns = [
        r"<Media omitted>",
        r"Messages and calls are end-to-end encrypted",
        r"This message was deleted",
        r"You deleted this message",
    ]
    cleaned_lines = []
    ts_sender_regex = re.compile(
        r"^(?:\[?\d{1,2}[/-]\d{1,2}[/-]\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?\]?\s*[-–]?\s*)([^:]+:\s*)?"
    )

    for line in raw_text.split("\n"):
        line_str = line.strip()
        if not line_str:
            continue
        if any(re.search(pat, line_str, re.IGNORECASE) for pat in system_patterns):
            continue
        cleaned_line = ts_sender_regex.sub("", line_str).strip()
        if cleaned_line:
            cleaned_lines.append(cleaned_line)

    return "\n".join(cleaned_lines)


def clean_rare_characters(train_text: str, val_text: str, val_multi_text: str, min_freq: int = 5):
    """
    Removes any character that appears fewer than min_freq times across the combined corpus.
    Reports the number of unique rare characters and total character instances removed.
    """
    combined = train_text + val_text
    counts = Counter(combined)
    rare_chars = {ch for ch, cnt in counts.items() if cnt < min_freq}

    if not rare_chars:
        return train_text, val_text, val_multi_text, {
            "min_freq": min_freq,
            "unique_rare_chars_removed": 0,
            "total_instances_removed": 0,
            "removed_chars": []
        }

    clean_train = "".join(ch for ch in train_text if ch not in rare_chars)
    clean_val = "".join(ch for ch in val_text if ch not in rare_chars)
    clean_val_multi = "".join(ch for ch in val_multi_text if ch not in rare_chars)

    total_instances = sum(counts[ch] for ch in rare_chars)
    report = {
        "min_freq": min_freq,
        "unique_rare_chars_removed": len(rare_chars),
        "total_instances_removed": total_instances,
        "removed_chars": sorted(list(rare_chars))
    }
    return clean_train, clean_val, clean_val_multi, report


def download_shakespeare_if_missing():
    """Downloads Tiny Shakespeare if not already present in data/raw/."""
    path = os.path.join(RAW_DIR, "shakespeare.txt")
    if os.path.exists(path):
        return
    print("[DOWNLOAD] Downloading Tiny Shakespeare dataset...")
    try:
        req = urllib.request.Request(
            SHAKESPEARE_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
        with open(path, "wb") as f:
            f.write(content)
        print(f"[OK] Saved {path} ({len(content) / 1024:.1f} KB)")
    except Exception as e:
        print(f"[WARN] Shakespeare download skipped/failed: {e}")


def inspect_raw_files(raw_files: List[str]) -> None:
    """Inspects and prints size, lines, unique chars, and Devanagari count for all raw files."""
    print("=" * 70)
    print("🔍 Auto-detected Raw Files inside data/raw/:")
    print("=" * 70)
    for fname in raw_files:
        fpath = os.path.join(RAW_DIR, fname)
        size_kb = os.path.getsize(fpath) / 1024.0
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        lines = len([l for l in content.split("\n") if l.strip()])
        chars = len(set(content))
        dev_count = sum(1 for c in content if 0x0900 <= ord(c) <= 0x097F)
        print(f"📄 {fname:25s} | {size_kb:8.2f} KB | {lines:5d} lines | {chars:3d} unique chars | Devanagari: {dev_count:5d}")
    print("=" * 70)


def process_and_split_files(
    cfg: Config
) -> Tuple[str, str, str, Dict[str, dict]]:
    """
    1. Reads each file up to cfg.max_kb_per_file (continuous from start).
    2. Deduplicates strictly ACROSS DIFFERENT files (removes cross-file overlap).
       Keeps within-file repeated lines intact.
    3. Splits each file continuously: first 90% train, last 10% val.
    4. Separately extracts multilingual validation text to evaluate Hindi/Hinglish loss.
    5. Saves combined splits to train.txt, val.txt, val_multilingual.txt, and data.txt.
    """
    raw_files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".txt"))
    if not raw_files:
        raise FileNotFoundError(f"No .txt files found in {RAW_DIR}")

    inspect_raw_files(raw_files)

    seen_in_previous_files = set()
    file_stats = {}
    train_chunks = []
    val_chunks = []
    val_multi_chunks = []

    print("\n📦 Processing & Cross-File Deduplication (Before Split):")
    for fname in raw_files:
        fpath = os.path.join(RAW_DIR, fname)
        max_bytes = cfg.max_kb_per_file * 1024
        
        # Read file with continuous cap from start
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            full_content = f.read()

        # WhatsApp cleaning if file is whatsapp.txt
        if fname.lower() == "whatsapp.txt":
            print(f"  💬 Cleaning WhatsApp chat export ({fname})...")
            full_content = clean_whatsapp_text(full_content)

        raw_bytes = full_content.encode("utf-8")
        # Custom capping: Real chat and assistant dialogues uncapped, daily dialog and wiki capped
        if fname.lower() in ("real_chat_corpus.txt", "clean_assistant_corpus.txt", "hindi_chat.txt", "action_alignment.txt"):
            is_capped = False
            capped_content = full_content
        elif fname.lower() == "clean_dailydialog.txt":
            daily_cap_bytes = 300 * 1024  # 300 KB of clean conversational banter
            is_capped = len(raw_bytes) > daily_cap_bytes
            capped_content = raw_bytes[:daily_cap_bytes].decode("utf-8", errors="ignore") if is_capped else full_content
        elif fname.lower() == "hindi_wiki.txt":
            wiki_cap_bytes = 250 * 1024  # 250 KB of Hindi wiki vocabulary
            is_capped = len(raw_bytes) > wiki_cap_bytes
            capped_content = raw_bytes[:wiki_cap_bytes].decode("utf-8", errors="ignore") if is_capped else full_content
        else:
            is_capped = len(raw_bytes) > max_bytes
            capped_content = raw_bytes[:max_bytes].decode("utf-8", errors="ignore") if is_capped else full_content

        # Block-level deduplication to preserve complete dialogue integrity
        cross_file_overlap_count = 0
        within_file_repeats_kept = 0
        local_seen_in_this_file = set()

        if "User:" in capped_content and "Assistant:" in capped_content:
            raw_blocks = [b.strip() for b in capped_content.split("\n\n") if b.strip()]
            deduped_blocks = []
            for block in raw_blocks:
                # Must contain both User and Assistant
                if "User:" not in block or "Assistant:" not in block:
                    continue
                if block in seen_in_previous_files:
                    cross_file_overlap_count += 1
                    continue
                if block in local_seen_in_this_file:
                    within_file_repeats_kept += 1
                else:
                    local_seen_in_this_file.add(block)
                deduped_blocks.append(block)
            seen_in_previous_files.update(local_seen_in_this_file)
            clean_text = "\n\n".join(deduped_blocks)
        else:
            raw_blocks = [b.strip() for b in capped_content.split("\n\n") if b.strip()]
            deduped_blocks = []
            for block in raw_blocks:
                if block in seen_in_previous_files:
                    cross_file_overlap_count += 1
                    continue
                if block in local_seen_in_this_file:
                    within_file_repeats_kept += 1
                else:
                    local_seen_in_this_file.add(block)
                deduped_blocks.append(block)
            seen_in_previous_files.update(local_seen_in_this_file)
            clean_text = "\n\n".join(deduped_blocks)

        # Per-file continuous 90/10 split on complete block boundaries
        blocks_to_split = clean_text.split("\n\n")
        split_b_idx = int(len(blocks_to_split) * cfg.train_val_split)
        file_train = "\n\n".join(blocks_to_split[:split_b_idx])
        file_val = "\n\n".join(blocks_to_split[split_b_idx:])

        # Check if file has multilingual content
        devanagari_count = sum(1 for c in clean_text if 0x0900 <= ord(c) <= 0x097F)
        is_multilingual = devanagari_count > 0 or ("hindi" in fname.lower() or "user" in fname.lower() or "my" in fname.lower())

        # Optional train upsampling
        if is_multilingual and cfg.upsample_multilingual_train > 1:
            train_chunks.append((file_train + "\n\n") * cfg.upsample_multilingual_train)
            print(f"  🔁 Upsampled {fname} train split by {cfg.upsample_multilingual_train}x")
        else:
            train_chunks.append(file_train)

        val_chunks.append(file_val)
        if is_multilingual:
            val_multi_chunks.append(file_val)

        cap_note = f" (Capped at {cfg.max_kb_per_file} KB)" if is_capped else ""
        file_stats[fname] = {
            "original_kb": len(raw_bytes) / 1024.0,
            "used_kb": len(clean_text.encode("utf-8")) / 1024.0,
            "deduped_chars": len(clean_text),
            "cross_file_overlap_removed": cross_file_overlap_count,
            "within_file_repeats_kept": within_file_repeats_kept,
            "train_chars": len(file_train),
            "val_chars": len(file_val),
            "is_capped": is_capped,
            "devanagari_chars": devanagari_count
        }

        print(f"  • {fname:25s}{cap_note}")
        print(f"    Used: {file_stats[fname]['used_kb']:.2f} KB | Train: {len(file_train):,d} chars | Val: {len(file_val):,d} chars")
        print(f"    Cross-file overlap (removed): {cross_file_overlap_count} | Within-file repeats (kept): {within_file_repeats_kept}")

    combined_train = "\n\n".join(train_chunks).strip()
    combined_val = "\n\n".join(val_chunks).strip()
    combined_val_multi = "\n\n".join(val_multi_chunks).strip()

    # Rare-character cleanup (corpus me jo character 5 baar se kam aaye use hata do)
    combined_train, combined_val, combined_val_multi, rare_report = clean_rare_characters(
        combined_train, combined_val, combined_val_multi, min_freq=5
    )
    combined_all = combined_train + "\n\n" + combined_val

    print("\n🧹 Rare Character Cleanup (min_freq = 5):")
    print(f"  • Unique rare characters removed   : {rare_report['unique_rare_chars_removed']}")
    print(f"  • Total character instances removed: {rare_report['total_instances_removed']}")
    if rare_report['removed_chars']:
        sample_chars = [repr(c) for c in rare_report['removed_chars'][:15]]
        print(f"  • Sample removed rare characters   : {', '.join(sample_chars)}")

    # Save to disk
    os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        f.write(combined_train)
    with open(OUT_VAL, "w", encoding="utf-8") as f:
        f.write(combined_val)
    with open(OUT_VAL_MULTI, "w", encoding="utf-8") as f:
        f.write(combined_val_multi)
    with open(OUT_DATA, "w", encoding="utf-8") as f:
        f.write(combined_all)

    return combined_train, combined_val, combined_val_multi, file_stats


if __name__ == "__main__":
    cfg = Config.get_preset("small")
    train_txt, val_txt, val_multi_txt, stats = process_and_split_files(cfg)

    train_kb = len(train_txt.encode("utf-8")) / 1024.0
    val_kb = len(val_txt.encode("utf-8")) / 1024.0
    val_multi_kb = len(val_multi_txt.encode("utf-8")) / 1024.0
    total_kb = len((train_txt + val_txt).encode("utf-8")) / 1024.0

    print("\n" + "=" * 70)
    print("📊 Prepared Dataset Summary (Per-File Split & Rebalanced):")
    print("=" * 70)
    print(f"Total Dataset Size   : {total_kb:.2f} KB ({total_kb / 1024.0:.3f} MB)")
    print(f"Train Split Size     : {train_kb:.2f} KB ({len(train_txt):,d} characters)")
    print(f"Val Split Size (All) : {val_kb:.2f} KB ({len(val_txt):,d} characters)")
    print(f"Val Multilingual Size: {val_multi_kb:.2f} KB ({len(val_multi_txt):,d} characters)")
    print(f"Saved Files          : {OUT_TRAIN}, {OUT_VAL}, {OUT_VAL_MULTI}, {OUT_DATA}")
    print("=" * 70)
