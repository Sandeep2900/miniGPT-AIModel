"""Read-only dry run for selecting conversation pairs from data/train.txt."""

from __future__ import annotations

import collections
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "train.txt"
OUTPUT_DIR = Path(__file__).resolve().parent
REMOVED_LOG = OUTPUT_DIR / "removed_log.txt"
REVIEW_FILE = OUTPUT_DIR / "review.txt"

# This list makes the common-greeting rule explicit and reviewable.
COMMON_GREETINGS = {
    "hi", "hello", "hey", "namaste", "pranam", "नमस्ते", "नमस्कार",
    "kaise ho", "kese ho", "good morning", "good afternoon", "good evening",
    "how are you", "bye", "goodbye", "alvida", "thank you", "thanks",
    "shukriya",
}

# These patterns detect possible personal data; a hit sends the pair to review.
PRIVACY_PATTERNS = {
    "phone_like": re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)"),
    "email": re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"),
    "url": re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'{}|\\^\[\]]+"),
    "credential": re.compile(r"(?i)\b(?:password|passwd|pin|otp)\s*[:=]\s*\S+"),
}

# These are the same visible keyword groups used by data_audit/supplement.ps1.
UNSAFE_PATTERNS = {
    "unsafe_self_harm_keyword": re.compile(
        r"(?i)\b(?:suicide|suicidal|self[- ]harm|kill myself|end my life)\b"
    ),
    "unsafe_violence_keyword": re.compile(
        r"(?i)\b(?:bomb|explosive|shoot|stab|murder|weapon|kill someone)\b"
    ),
    "unsafe_drug_keyword": re.compile(
        r"(?i)\b(?:cocaine|heroin|methamphetamine|fentanyl|overdose)\b"
    ),
    "unsafe_sexual_keyword": re.compile(
        r"(?i)\b(?:porn(?:ography)?|explicit sexual|sexual abuse|rape)\b"
    ),
}

# These visible forms are treated as template placeholders, not ordinary angle brackets.
TEMPLATE_PATTERN = re.compile(
    r"(?i)(\{\{[^{}]+\}\}|\{(?:name|email|phone|date|time|city|topic)\}|"
    r"\[(?:placeholder|your name|insert [^\]]+)\]|"
    r"<(?:(?:your )?(?:name|email|phone|date|time)|placeholder)>|"
    r"\b(?:lorem ipsum|TODO|TBD|INSERT_HERE)\b)"
)


def text_after(line: str, prefix: str) -> str:
    """Return stripped content after a speaker prefix."""
    return line[len(prefix) :].strip()


def normalized_greeting(user_text: str) -> str:
    """Normalize case and edge punctuation for the explicit greeting list."""
    return user_text.casefold().strip(" \t.,!?।…:;\"'()")


def script_flags(text: str) -> tuple[bool, bool]:
    """Return whether text has Devanagari and ASCII Latin letters."""
    has_devanagari = any("\u0900" <= char <= "\u097f" for char in text)
    has_latin = any(char.isascii() and char.isalpha() for char in text)
    return has_devanagari, has_latin


def pii_reasons(text: str) -> list[str]:
    """Return privacy-pattern labels found in text."""
    return [name for name, pattern in PRIVACY_PATTERNS.items() if pattern.search(text)]


def unsafe_reasons(text: str) -> list[str]:
    """Return unsafe-keyword labels found in text."""
    return [name for name, pattern in UNSAFE_PATTERNS.items() if pattern.search(text)]


def mask_sensitive(text: str) -> str:
    """Mask each detected value while preserving a short prefix and suffix."""
    for pattern in PRIVACY_PATTERNS.values():
        def replace(match: re.Match[str]) -> str:
            value = match.group(0)
            if len(value) <= 4:
                return "*" * len(value)
            return value[:2] + "X" * max(2, len(value) - 4) + value[-2:]

        text = pattern.sub(replace, text)
    return text


def short_log_text(line: str) -> str:
    """Keep at most 120 characters in the movement log."""
    return line[:120]


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source file is missing: {SOURCE}")

    # Split only on newline characters so physical line numbers stay stable.
    source_text = SOURCE.read_text(encoding="utf-8")
    lines = source_text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    lines = [line[:-1] if line.endswith("\r") else line for line in lines]
    kept: list[dict[str, object]] = []
    review_items: list[dict[str, object]] = []
    removed_log_lines: list[str] = []
    drop_line_counts: collections.Counter[str] = collections.Counter()
    review_reason_counts: collections.Counter[str] = collections.Counter()
    seen_pairs: set[tuple[str, str]] = set()
    greeting_occurrences: collections.Counter[str] = collections.Counter()
    consumed: set[int] = set()

    # Count distinct prompts per exact reply once so the filler check stays fast.
    reply_to_users: dict[str, set[str]] = collections.defaultdict(set)
    for pair_index in range(len(lines) - 1):
        if lines[pair_index].startswith("User:") and lines[pair_index + 1].startswith("Assistant:"):
            prompt = text_after(lines[pair_index], "User:")
            reply = text_after(lines[pair_index + 1], "Assistant:")
            reply_to_users[reply].add(prompt)

    def log_moved(line_index: int, reason: str) -> None:
        """Log each dropped or reviewed physical line with its original text."""
        raw = lines[line_index]
        removed_log_lines.append(
            f"line={line_index + 1} | reason={reason} | text={short_log_text(raw)}"
        )

    def add_review(indices: list[int], reasons: list[str], kind: str) -> None:
        """Move one pair or uncertain fragment to review and log every line."""
        unique_reasons = list(dict.fromkeys(reasons))
        for reason in unique_reasons:
            review_reason_counts[reason] += 1
        review_items.append({"indices": indices, "reasons": unique_reasons, "kind": kind})
        reason_text = ",".join(unique_reasons)
        for line_index in indices:
            consumed.add(line_index)
            log_moved(line_index, f"REVIEW:{reason_text}")

    def drop_lines(indices: list[int], reason: str) -> None:
        """Drop physical lines and log each one separately."""
        for line_index in indices:
            consumed.add(line_index)
            drop_line_counts[reason] += 1
            log_moved(line_index, f"DROP:{reason}")

    i = 0
    while i < len(lines):
        if i in consumed:
            i += 1
            continue

        line = lines[i]

        # Blank lines are dropped and recorded, including their original line number.
        if not line.strip():
            drop_lines([i], "blank_line")
            i += 1
            continue

        if line.startswith("User:"):
            # A pair requires the very next physical line to start with Assistant:.
            if i + 1 >= len(lines) or not lines[i + 1].startswith("Assistant:"):
                drop_lines([i], "orphan_user")
                i += 1
                continue

            assistant_index = i + 1
            user_text = text_after(line, "User:")
            assistant_text = text_after(lines[assistant_index], "Assistant:")
            pair_indices = [i, assistant_index]
            consumed.update(pair_indices)

            # Nonblank untagged lines directly after a reply may be reply continuations.
            # Since they cannot safely be separated from the reply, review the whole block.
            continuation_indices: list[int] = []
            cursor = assistant_index + 1
            while cursor < len(lines) and lines[cursor].strip():
                if lines[cursor].startswith("User:") or lines[cursor].startswith("Assistant:"):
                    break
                continuation_indices.append(cursor)
                cursor += 1

            # Both texts need at least two characters after removing their prefixes.
            if len(user_text) < 2 or len(assistant_text) < 2:
                drop_lines(pair_indices, "empty_or_short_pair_text")
                if continuation_indices:
                    add_review(continuation_indices, ["possible_multiline_reply"], "fragment")
                i = assistant_index + 1
                continue

            pair_key = (user_text, assistant_text)

            # Keep only the first exact pair; later copies are dropped.
            if pair_key in seen_pairs:
                drop_lines(pair_indices, "exact_duplicate_pair")
                if continuation_indices:
                    add_review(continuation_indices, ["possible_multiline_reply"], "fragment")
                i = assistant_index + 1
                continue
            seen_pairs.add(pair_key)

            # Keep only the first three occurrences of each explicitly listed greeting.
            greeting_key = normalized_greeting(user_text)
            if greeting_key in COMMON_GREETINGS:
                greeting_occurrences[greeting_key] += 1
                if greeting_occurrences[greeting_key] > 3:
                    drop_lines(pair_indices, "common_greeting_over_limit")
                    if continuation_indices:
                        add_review(continuation_indices, ["possible_multiline_reply"], "fragment")
                    i = assistant_index + 1
                    continue

            # Replies with dynamic time text or a listed placeholder are dropped.
            if "current time:" in assistant_text.casefold() or TEMPLATE_PATTERN.search(assistant_text):
                drop_lines(pair_indices, "time_or_template_reply")
                if continuation_indices:
                    add_review(continuation_indices, ["possible_multiline_reply"], "fragment")
                i = assistant_index + 1
                continue

            # Apply review rules to pairs that otherwise pass the keep checks.
            reasons: list[str] = []
            if continuation_indices:
                reasons.append("possible_multiline_reply")

            # A repeated exact reply across five or more distinct prompts is reviewed.
            if len(reply_to_users[assistant_text]) >= 5:
                reasons.append("possible_generated_filler")

            # Review only pairs whose sides use opposite single scripts.
            user_deva, user_latin = script_flags(user_text)
            assistant_deva, assistant_latin = script_flags(assistant_text)
            if (user_deva and not user_latin and assistant_latin and not assistant_deva) or (
                user_latin and not user_deva and assistant_deva and not assistant_latin
            ):
                reasons.append("different_scripts")

            # Unsafe words trigger review only; they do not prove unsafe intent.
            for reason in unsafe_reasons(user_text + "\n" + assistant_text):
                reasons.append(reason)

            # Possible phone, email, URL, or credential values trigger masked review.
            for reason in pii_reasons(user_text + "\n" + assistant_text):
                reasons.append(f"possible_private_data_{reason}")

            if reasons:
                all_review_indices = pair_indices + continuation_indices
                add_review(all_review_indices, reasons, "pair")
            else:
                # Pairs with no drop or review rule are retained in memory only.
                kept.append(
                    {
                        "user": user_text,
                        "assistant": assistant_text,
                        "line_numbers": (i + 1, assistant_index + 1),
                    }
                )

            i = assistant_index + 1
            continue

        if line.startswith("Assistant:"):
            # An Assistant line not consumed by a valid User line is orphaned.
            drop_lines([i], "orphan_assistant")
            i += 1
            continue

        # Untagged text after an Assistant line is uncertain and goes to review.
        previous_is_assistant = i > 0 and lines[i - 1].startswith("Assistant:")
        if previous_is_assistant:
            fragment_indices = [i]
            cursor = i + 1
            while cursor < len(lines) and lines[cursor].strip():
                if lines[cursor].startswith("User:") or lines[cursor].startswith("Assistant:"):
                    break
                fragment_indices.append(cursor)
                cursor += 1
            add_review(fragment_indices, ["possible_multiline_reply"], "fragment")
            i = cursor
            continue

        # Other untagged text is non-dialogue and is dropped.
        drop_lines([i], "non_dialogue_line")
        i += 1

    # Stage 1 writes only review and movement logs; it never writes a cleaned train file.
    review_output: list[str] = [
        "DRY-RUN REVIEW ITEMS",
        "Each value is a review candidate; private-data patterns are masked.",
        "",
    ]
    for item_number, item in enumerate(review_items, start=1):
        review_output.append(
            f"ITEM {item_number} | kind={item['kind']} | reasons={','.join(item['reasons'])}"
        )
        for line_index in item["indices"]:
            original = lines[line_index]
            review_output.append(
                f"line={line_index + 1} | {mask_sensitive(original)}"
            )
        review_output.append("")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_FILE.write_text("\n".join(review_output), encoding="utf-8")
    REMOVED_LOG.write_text("\n".join(removed_log_lines) + "\n", encoding="utf-8")

    print("=== STAGE 1 DRY RUN ===")
    print("Source read only; no cleaned training file was written.")
    print(f"Kept pairs (in-memory only): {len(kept)}")
    print("Dropped lines by reason:")
    for reason, count in sorted(drop_line_counts.items()):
        print(f"  {reason}: {count}")
    print("Review items by reason (one pair/fragment may have multiple reasons):")
    for reason, count in sorted(review_reason_counts.items()):
        print(f"  {reason}: {count}")
    print(f"Total review items: {len(review_items)}")
    print("Top 10 kept User texts:")
    for text, count in collections.Counter(item["user"] for item in kept).most_common(10):
        print(f"  {count} | {text}")
    print("Top 10 kept Assistant texts:")
    for text, count in collections.Counter(item["assistant"] for item in kept).most_common(10):
        print(f"  {count} | {text}")
    print("20 random kept pairs (fixed seed=20261101):")
    sample = random.Random(20261101).sample(kept, min(20, len(kept)))
    for sample_number, item in enumerate(sample, start=1):
        print(f"--- SAMPLE {sample_number} | source lines {item['line_numbers'][0]}-{item['line_numbers'][1]} ---")
        print(f"User: {item['user']}")
        print(f"Assistant: {item['assistant']}")
    print(f"Review file: {REVIEW_FILE}")
    print(f"Movement log: {REMOVED_LOG}")


if __name__ == "__main__":
    main()
