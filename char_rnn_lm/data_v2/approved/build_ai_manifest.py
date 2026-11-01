# Build a manifest only after the AI-reviewed dataset passes validation.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from validate_ai_reviewed import DATA_FILE, validate


MANIFEST_FILE = Path(__file__).resolve().parent / "ai_review_manifest.json"


def main() -> int:
    if MANIFEST_FILE.exists():
        print(f"ERROR: refusing to overwrite manifest: {MANIFEST_FILE}")
        return 2

    issues, record_count, valid_count, language_counts, topic_counts = validate()
    if issues or valid_count != record_count:
        print("ERROR: AI-reviewed data failed validation; no manifest written.")
        print(f"Records: {record_count}")
        print(f"Valid records: {valid_count}")
        for reason, count in sorted(issues.items()):
            print(f"  {reason}: {count}")
        return 1

    with DATA_FILE.open(encoding="utf-8") as source:
        conversation_count = len({json.loads(line)["conversation_id"] for line in source if line.strip()})

    manifest = {
        "data_file": DATA_FILE.name,
        "sha256": hashlib.sha256(DATA_FILE.read_bytes()).hexdigest(),
        "record_count": record_count,
        "conversation_count": conversation_count,
        "language_counts": dict(sorted(language_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "source": "ai_generated",
        "review_status": "ai_reviewed_experiment",
        "human_verified": False,
    }

    with MANIFEST_FILE.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(f"Manifest: {MANIFEST_FILE}")
    print(f"SHA-256: {manifest['sha256']}")
    print(f"Records: {record_count}")
    print(f"Conversations: {conversation_count}")
    print(f"Languages: {json.dumps(manifest['language_counts'], sort_keys=True)}")
    print(f"Topics: {json.dumps(manifest['topic_counts'], sort_keys=True)}")
    print("Human verified: false; AI-reviewed experiment only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
