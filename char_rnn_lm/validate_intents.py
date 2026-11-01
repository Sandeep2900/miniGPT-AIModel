"""
validate_intents.py
Stage B: Strict validation script for data/intents.jsonl.
Validates:
1. Every line is valid JSON.
2. No empty fields (intent, phrase, replies).
3. No duplicate phrases (case-insensitive across the entire dataset).
4. Count per intent is within 15-20 range.
5. All replies have strictly under 15 words.
6. Prints a structured summary table and 5 random samples.
"""

import os
import sys
import json
import random

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

INTENTS_FILE = os.path.join("data", "intents.jsonl")

EXPECTED_INTENTS = [
    "greeting",
    "identity",
    "user_name_unknown",
    "location_question",
    "time_question",
    "date_question",
    "thanks",
    "goodbye"
]


def validate():
    print("=" * 78)
    print("📋 Stage B: Intent Dataset Validation (data/intents.jsonl)")
    print("=" * 78)

    if not os.path.exists(INTENTS_FILE):
        print(f"[ERROR] Intents file not found at: {INTENTS_FILE}")
        sys.exit(1)

    seen_phrases = {}
    intent_counts = {intent: 0 for intent in EXPECTED_INTENTS}
    intent_samples = {intent: [] for intent in EXPECTED_INTENTS}
    all_records = []
    errors = []

    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Total lines read: {len(lines)}")

    for line_num, line in enumerate(lines, 1):
        line_str = line.strip()
        if not line_str:
            errors.append(f"Line {line_num}: Empty line detected.")
            continue

        # 1. Valid JSON Check
        try:
            record = json.loads(line_str)
        except Exception as e:
            errors.append(f"Line {line_num}: Invalid JSON syntax -> {e}")
            continue

        # 2. Key Presence & Non-Empty Check
        intent = record.get("intent", "").strip()
        phrase = record.get("phrase", "").strip()
        replies = record.get("replies", [])

        if not intent:
            errors.append(f"Line {line_num}: 'intent' field is missing or empty.")
        elif intent not in EXPECTED_INTENTS:
            errors.append(f"Line {line_num}: Unknown intent '{intent}'. Expected one of {EXPECTED_INTENTS}.")

        if not phrase:
            errors.append(f"Line {line_num}: 'phrase' field is missing or empty.")

        if not isinstance(replies, list) or len(replies) == 0:
            errors.append(f"Line {line_num}: 'replies' must be a non-empty list.")
        else:
            # Check length of replies and word count under 15 words
            for r_idx, r in enumerate(replies, 1):
                if not isinstance(r, str) or not r.strip():
                    errors.append(f"Line {line_num}: Reply #{r_idx} is empty or not string.")
                else:
                    word_count = len(r.strip().split())
                    if word_count > 15:
                        errors.append(f"Line {line_num}: Reply #{r_idx} exceeds 15 words ({word_count} words): '{r}'")

        # 3. Duplicate Phrase Check (Case-Insensitive)
        norm_phrase = phrase.lower()
        if norm_phrase in seen_phrases:
            first_line = seen_phrases[norm_phrase]
            errors.append(f"Line {line_num}: Duplicate phrase '{phrase}' already seen on line {first_line}.")
        else:
            seen_phrases[norm_phrase] = line_num

        if intent in intent_counts:
            intent_counts[intent] += 1
            intent_samples[intent].append(record)
            all_records.append(record)

    # 4. Intent Count Range Check (15 to 20 phrases)
    for intent, count in intent_counts.items():
        if count < 15 or count > 20:
            errors.append(f"Intent '{intent}' has {count} phrases (must be between 15 and 20).")

    # If any error found, display and exit
    if errors:
        print("\n❌ VALIDATION FAILED! Found errors:")
        for err in errors[:20]:
            print(f"  • {err}")
        if len(errors) > 20:
            print(f"  ...and {len(errors) - 20} more errors.")
        sys.exit(1)

    print("\n✅ All validation checks PASSED:")
    print("  • Valid JSON syntax on every line")
    print("  • Zero empty fields")
    print("  • Zero duplicate user phrases")
    print("  • All replies strictly <= 15 words")
    print("  • Exact 8 expected intents verified\n")

    # 5. Print Distribution Table
    print(f"{'#':<3} | {'Intent Name':<22} | {'Count':<7} | {'Sample Phrase':<30} | {'Sample Reply'}")
    print("-" * 110)
    for i, intent in enumerate(EXPECTED_INTENTS, 1):
        cnt = intent_counts[intent]
        sample = intent_samples[intent][0]
        s_phrase = sample["phrase"]
        s_reply = sample["replies"][0]
        if len(s_phrase) > 28:
            s_phrase = s_phrase[:25] + "..."
        if len(s_reply) > 40:
            s_reply = s_reply[:37] + "..."
        print(f"{i:<3} | {intent:<22} | {cnt:<7} | {s_phrase:<30} | {s_reply}")
    print("-" * 110)
    print(f"Total Intents: {len(EXPECTED_INTENTS)} | Total Unique Phrases: {len(all_records)}\n")

    # 6. Print 5 Random Examples
    random.seed(42)
    sample_records = random.sample(all_records, 5)
    print("🎲 5 Random Human-Verifiable Examples from Dataset:")
    print("=" * 78)
    for idx, item in enumerate(sample_records, 1):
        print(f"[{idx}] Intent  : {item['intent']}")
        print(f"    Phrase  : \"{item['phrase']}\"")
        print(f"    Replies :")
        for r_num, rep in enumerate(item["replies"], 1):
            words = len(rep.split())
            print(f"       ({r_num}) \"{rep}\" [{words} words]")
        print("-" * 78)


if __name__ == "__main__":
    validate()
