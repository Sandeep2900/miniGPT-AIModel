# Validate AI-reviewed experiment records without changing the input file.

from __future__ import annotations

import collections
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = Path(__file__).resolve().parent / "ai_reviewed_pairs.jsonl"
EVAL_PROMPTS = Path(__file__).resolve().parent / "eval_prompts.txt"
ALLOWED_LANGS = {"hi", "hinglish", "en"}
ALLOWED_TOPICS = {"greeting", "small_talk", "unknown"}
REQUIRED = {
    "conversation_id", "user", "assistant", "lang", "topic", "source",
    "reviewed_by_ai", "review_status",
}
PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"'{}|\\^\[\]]+")


def normalize(text: str) -> str:
    """Normalize exact prompt and pair comparisons."""
    return text.strip().casefold()


def script_counts(text: str) -> tuple[int, int]:
    """Count Devanagari and ASCII Latin letters."""
    devanagari = sum(1 for char in text if "\u0900" <= char <= "\u097f" and char.isalpha())
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    return devanagari, latin


def load_reference_prompts() -> set[str]:
    """Read evaluation, candidate-user, and intent prompts for overlap checks."""
    references: set[str] = set()

    if EVAL_PROMPTS.is_file():
        for line in EVAL_PROMPTS.read_text(encoding="utf-8").splitlines():
            if "|" in line:
                references.add(normalize(line.split("|", 1)[1]))

    candidates = ROOT / "data_v2" / "candidates" / "conversation_candidates.jsonl"
    if candidates.is_file():
        with candidates.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    references.add(normalize(json.loads(line)["user"]))

    for path in sorted((ROOT / "data").glob("intents*.jsonl")):
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    references.add(normalize(json.loads(line)["phrase"]))
    return references


def has_phone(text: str) -> bool:
    """Require at least ten digits in a phone-like sequence."""
    return any(sum(char.isdigit() for char in match.group(0)) >= 10 for match in PHONE.finditer(text))


def script_matches(lang: str, user: str, assistant: str) -> bool:
    """Check script compatibility for both sides of a pair."""
    for text in (user, assistant):
        devanagari, latin = script_counts(text)
        if lang == "hi" and devanagari == 0:
            return False
        if lang == "en" and (latin == 0 or devanagari > 0):
            return False
        if lang == "hinglish" and latin == 0:
            return False
    return True


def validate() -> tuple[
    collections.Counter[str], int, int, collections.Counter[str], collections.Counter[str]
]:
    """Return validation issue counts and record distributions."""
    issues: collections.Counter[str] = collections.Counter()
    languages: collections.Counter[str] = collections.Counter()
    topics: collections.Counter[str] = collections.Counter()
    seen_pairs: set[tuple[str, str]] = set()
    seen_prompts: set[str] = set()
    seen_conversations: set[str] = set()
    reference_prompts = load_reference_prompts()
    total = 0
    valid = 0

    with DATA_FILE.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                issues["invalid_json"] += 1
                continue
            if not isinstance(record, dict):
                issues["record_not_object"] += 1
                continue

            row_issues = 0
            missing = REQUIRED - record.keys()
            if missing:
                issues["records_missing_fields"] += 1
                for field in sorted(missing):
                    issues[f"missing_field:{field}"] += 1
                row_issues += 1
            if set(record) - REQUIRED:
                issues["records_with_extra_fields"] += 1
                row_issues += 1

            for field in ("conversation_id", "user", "assistant"):
                value = record.get(field)
                if not isinstance(value, str) or not value.strip():
                    issues[f"invalid_or_empty:{field}"] += 1
                    row_issues += 1

            lang = record.get("lang")
            if not isinstance(lang, str) or lang not in ALLOWED_LANGS:
                issues["invalid_lang"] += 1
                row_issues += 1
            else:
                languages[lang] += 1

            topic = record.get("topic")
            if not isinstance(topic, str) or topic not in ALLOWED_TOPICS:
                issues["invalid_topic"] += 1
                row_issues += 1
            else:
                topics[topic] += 1

            if record.get("source") != "ai_generated":
                issues["invalid_source"] += 1
                row_issues += 1
            if not isinstance(record.get("reviewed_by_ai"), str) or not record["reviewed_by_ai"].strip():
                issues["missing_ai_reviewer"] += 1
                row_issues += 1
            if record.get("review_status") != "ai_reviewed_experiment":
                issues["invalid_review_status"] += 1
                row_issues += 1
            if "verified_by" in record:
                issues["misleading_human_verification_field"] += 1
                row_issues += 1

            user = record.get("user") if isinstance(record.get("user"), str) else ""
            assistant = record.get("assistant") if isinstance(record.get("assistant"), str) else ""
            combined = user + "\n" + assistant
            if has_phone(combined):
                issues["possible_phone_pattern"] += 1
                row_issues += 1
            if EMAIL.search(combined):
                issues["possible_email_pattern"] += 1
                row_issues += 1
            if URL.search(combined):
                issues["possible_url_pattern"] += 1
                row_issues += 1

            if isinstance(lang, str) and lang in ALLOWED_LANGS and not script_matches(lang, user, assistant):
                issues["script_mismatch"] += 1
                row_issues += 1

            if user:
                prompt_key = normalize(user)
                if prompt_key in seen_prompts:
                    issues["duplicate_user_prompt"] += 1
                    row_issues += 1
                seen_prompts.add(prompt_key)
                if prompt_key in reference_prompts:
                    issues["overlap_with_eval_candidate_or_intent_prompt"] += 1
                    row_issues += 1

            if user and assistant:
                pair_key = (normalize(user), normalize(assistant))
                if pair_key in seen_pairs:
                    issues["duplicate_pair"] += 1
                    row_issues += 1
                seen_pairs.add(pair_key)

            conversation_id = record.get("conversation_id")
            if isinstance(conversation_id, str):
                if conversation_id in seen_conversations:
                    issues["duplicate_conversation_id"] += 1
                    row_issues += 1
                seen_conversations.add(conversation_id)

            if row_issues == 0:
                valid += 1

    return issues, total, valid, languages, topics


def main() -> int:
    if not DATA_FILE.is_file():
        print(f"ERROR: AI-reviewed pair file not found: {DATA_FILE}")
        return 2

    issues, total, valid, languages, topics = validate()
    print(f"Records: {total}")
    print(f"Valid AI-reviewed records: {valid}")
    print(f"Invalid records: {total - valid}")
    print(f"By language: {json.dumps(dict(sorted(languages.items())), ensure_ascii=False)}")
    print(f"By topic: {json.dumps(dict(sorted(topics.items())), ensure_ascii=False)}")
    if issues:
        print("Issue counts:")
        for reason, count in sorted(issues.items()):
            print(f"  {reason}: {count}")
        print("RESULT: FAIL")
        return 1
    print("Issue counts: none")
    print("RESULT: PASS (AI-reviewed experiment data; not human-verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
