# Build a manifest only from the designated human-approved JSONL file.

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from validate_approved import validate_file


APPROVED_DIR = Path(__file__).resolve().parent
APPROVED_FILE = APPROVED_DIR / "verified_pairs.jsonl"
MANIFEST_FILE = APPROVED_DIR / "manifest.json"


def main() -> int:
    # Refuse to create a manifest until a human-approved file is provided.
    if not APPROVED_FILE.is_file():
        print(f"ERROR: approved file not found: {APPROVED_FILE}")
        print("No manifest was written.")
        return 2

    # Refuse to replace any existing manifest.
    if MANIFEST_FILE.exists():
        print(f"ERROR: refusing to overwrite existing manifest: {MANIFEST_FILE}")
        return 2

    issues, record_count, valid_count = validate_file(APPROVED_FILE)
    if issues or valid_count != record_count:
        print("ERROR: approved file failed validation.")
        print(f"Nonblank records: {record_count}")
        print(f"Valid records: {valid_count}")
        for reason, count in sorted(issues.items()):
            print(f"  {reason}: {count}")
        print("No manifest was written.")
        return 1

    topic_counts: Counter[str] = Counter()
    with APPROVED_FILE.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                topic_counts[json.loads(line)["topic"]] += 1

    file_hash = hashlib.sha256(APPROVED_FILE.read_bytes()).hexdigest()
    manifest = {
        "approved_file": APPROVED_FILE.name,
        "sha256": file_hash,
        "record_count": record_count,
        "topic_counts": dict(sorted(topic_counts.items())),
    }

    # Exclusive creation protects an existing manifest from accidental overwrite.
    with MANIFEST_FILE.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(f"Manifest written: {MANIFEST_FILE}")
    print(f"SHA-256: {file_hash}")
    print(f"Records: {record_count}")
    print(f"Topic counts: {json.dumps(manifest['topic_counts'], ensure_ascii=False, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
