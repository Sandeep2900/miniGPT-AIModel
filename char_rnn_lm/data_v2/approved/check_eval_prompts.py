# Check evaluation prompt count, labels, scripts, and source duplicates.

from __future__ import annotations

import collections
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = Path(__file__).resolve().parent / "eval_prompts.txt"
CANDIDATES_PATH = ROOT / "data_v2" / "candidates" / "conversation_candidates.jsonl"
INTENT_FILES = sorted((ROOT / "data").glob("intents*.jsonl"))
GROUPS = {"hi", "hinglish", "en"}
EXPECTED_PER_GROUP = 10
EXPECTED_TOTAL = 30
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")


def normalize(text: str) -> str:
    """Compare prompt text after trimming and case folding."""
    return text.strip().casefold()


def load_reference_prompts() -> tuple[set[str], set[str]]:
    """Read candidate user prompts and all intent phrases."""
    candidate_prompts: set[str] = set()
    with CANDIDATES_PATH.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                candidate_prompts.add(normalize(json.loads(line)["user"]))

    intent_prompts: set[str] = set()
    for path in INTENT_FILES:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    intent_prompts.add(normalize(json.loads(line)["phrase"]))
    return candidate_prompts, intent_prompts


def script_matches(label: str, prompt: str) -> bool:
    """Check the requested script only; this does not judge language fluency."""
    has_devanagari = bool(DEVANAGARI_PATTERN.search(prompt))
    has_latin = bool(LATIN_PATTERN.search(prompt))
    if label == "hi":
        return has_devanagari and not has_latin
    if label in {"hinglish", "en"}:
        return has_latin
    return False


def main() -> int:
    if not PROMPTS_PATH.is_file():
        print(f"ERROR: prompt file not found: {PROMPTS_PATH}")
        return 2

    lines = PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
    group_counts: collections.Counter[str] = collections.Counter()
    malformed_lines = 0
    empty_prompts = 0
    script_mismatches: list[int] = []
    duplicate_within_file: list[tuple[int, int]] = []
    seen: dict[str, int] = {}
    parsed: list[tuple[int, str, str]] = []

    for line_number, line in enumerate(lines, 1):
        if "|" not in line:
            malformed_lines += 1
            continue
        label, prompt = line.split("|", 1)
        prompt = prompt.strip()
        if label not in GROUPS:
            malformed_lines += 1
            continue
        if not prompt:
            empty_prompts += 1
            continue

        group_counts[label] += 1
        parsed.append((line_number, label, prompt))
        if not script_matches(label, prompt):
            script_mismatches.append(line_number)

        key = normalize(prompt)
        if key in seen:
            duplicate_within_file.append((seen[key], line_number))
        else:
            seen[key] = line_number

    candidate_prompts, intent_prompts = load_reference_prompts()
    candidate_duplicates = [
        line_number
        for line_number, _, prompt in parsed
        if normalize(prompt) in candidate_prompts
    ]
    intent_duplicates = [
        line_number
        for line_number, _, prompt in parsed
        if normalize(prompt) in intent_prompts
    ]

    wrong_group_counts = [
        label for label in sorted(GROUPS) if group_counts[label] != EXPECTED_PER_GROUP
    ]
    failed = bool(
        len(lines) != EXPECTED_TOTAL
        or malformed_lines
        or empty_prompts
        or wrong_group_counts
        or script_mismatches
        or duplicate_within_file
        or candidate_duplicates
        or intent_duplicates
    )

    print(f"Prompt lines: {len(lines)} (expected {EXPECTED_TOTAL})")
    for label in ("hi", "hinglish", "en"):
        print(f"  {label}: {group_counts[label]} (expected {EXPECTED_PER_GROUP})")
    print(f"Malformed lines: {malformed_lines}")
    print(f"Empty prompts: {empty_prompts}")
    print(f"Script mismatches: {len(script_mismatches)}")
    print(f"Duplicates within prompt file: {len(duplicate_within_file)}")
    print(f"Duplicates in candidate user prompts: {len(candidate_duplicates)}")
    print(f"Duplicates in data/intents*.jsonl phrases: {len(intent_duplicates)}")
    print("RESULT: FAIL" if failed else "RESULT: PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
