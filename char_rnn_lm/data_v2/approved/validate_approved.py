# Validate approved JSONL records without changing the input file.

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


ALLOWED_LANGS = {"hi", "hinglish", "en"}
ALLOWED_SOURCES = {"human_written", "chat_export_anonymized"}
ALLOWED_TOPICS = {"greeting", "small_talk", "name", "clock", "unknown"}
REQUIRED_FIELDS = {"user", "assistant", "lang", "verified_by", "source", "topic"}

# These patterns flag possible personal data for human review.
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
URL_PATTERN = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'{}|\\^\[\]]+")


def has_phone_like_number(text: str) -> bool:
    """Match a phone-like sequence only when it contains at least ten digits."""
    for match in PHONE_PATTERN.finditer(text):
        if sum(char.isdigit() for char in match.group(0)) >= 10:
            return True
    return False


def script_counts(text: str) -> tuple[int, int]:
    """Count Devanagari and ASCII Latin letters in text."""
    devanagari = sum(1 for char in text if "\u0900" <= char <= "\u097f" and char.isalpha())
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    return devanagari, latin


def script_matches(record: dict[str, Any]) -> bool:
    """Check both user and assistant text against the declared language tag."""
    lang = record.get("lang")
    for field in ("user", "assistant"):
        text = record.get(field)
        if not isinstance(text, str):
            return False
        devanagari, latin = script_counts(text)
        if lang == "hi" and devanagari == 0:
            return False
        if lang == "en" and (latin == 0 or devanagari > 0):
            return False
        if lang == "hinglish" and latin == 0:
            return False
    return True


def validate_file(path: Path) -> tuple[collections.Counter[str], int, int]:
    """Return issue counts, nonblank line count, and valid record count."""
    issues: collections.Counter[str] = collections.Counter()
    seen_pairs: set[tuple[str, str]] = set()
    nonblank_lines = 0
    valid_records = 0

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            nonblank_lines += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                issues["invalid_json"] += 1
                continue

            if not isinstance(record, dict):
                issues["record_not_object"] += 1
                continue

            issue_count_before = sum(issues.values())
            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                issues["records_missing_fields"] += 1
                for field in sorted(missing):
                    issues[f"missing_field:{field}"] += 1

            if set(record) - REQUIRED_FIELDS:
                issues["records_with_extra_fields"] += 1

            for field in ("user", "assistant"):
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    issues[f"invalid_or_empty:{field}"] += 1

            lang = record.get("lang")
            if not isinstance(lang, str) or lang not in ALLOWED_LANGS:
                issues["invalid_lang"] += 1

            verified_by = record.get("verified_by")
            if not isinstance(verified_by, str) or not verified_by.strip():
                issues["verified_by_null_or_blank"] += 1

            source_value = record.get("source")
            if not isinstance(source_value, str) or source_value not in ALLOWED_SOURCES:
                issues["invalid_source"] += 1

            topic_value = record.get("topic")
            if not isinstance(topic_value, str) or topic_value not in ALLOWED_TOPICS:
                issues["invalid_topic"] += 1

            user_text = record.get("user") if isinstance(record.get("user"), str) else ""
            assistant_text = record.get("assistant") if isinstance(record.get("assistant"), str) else ""
            combined_text = user_text + "\n" + assistant_text

            if has_phone_like_number(combined_text):
                issues["possible_phone_pattern"] += 1
            if EMAIL_PATTERN.search(combined_text):
                issues["possible_email_pattern"] += 1
            if URL_PATTERN.search(combined_text):
                issues["possible_url_pattern"] += 1

            if lang in ALLOWED_LANGS and not script_matches(record):
                issues["script_mismatch"] += 1

            if isinstance(record.get("user"), str) and isinstance(record.get("assistant"), str):
                pair = (record["user"], record["assistant"])
                if pair in seen_pairs:
                    issues["duplicate_pair"] += 1
                else:
                    seen_pairs.add(pair)

            if sum(issues.values()) == issue_count_before:
                valid_records += 1

    return issues, nonblank_lines, valid_records


def main() -> int:
    default_path = Path(__file__).resolve().parents[1] / "candidates" / "conversation_candidates.jsonl"
    parser = argparse.ArgumentParser(description="Read-only approved-pair JSONL validator")
    parser.add_argument("file", nargs="?", type=Path, default=default_path)
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"ERROR: file not found: {args.file}")
        return 2

    issues, records, valid_records = validate_file(args.file)
    print(f"File: {args.file}")
    print(f"Nonblank records: {records}")
    print(f"Valid records: {valid_records}")
    print(f"Invalid records: {records - valid_records}")
    print("Issue counts:")
    if issues:
        for reason, count in sorted(issues.items()):
            print(f"  {reason}: {count}")
    else:
        print("  none")

    if issues:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
