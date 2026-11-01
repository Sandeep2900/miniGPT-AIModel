# Check whether prompt and answer tokenization composes exactly.

import json
from pathlib import Path

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = Path(__file__).resolve().parent / "ai_reviewed_pairs.jsonl"
TOKENIZER_FILE = ROOT / "data" / "bpe_tokenizer.json"

tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
records = [
    json.loads(line)
    for line in DATA_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
mismatches = 0
for record in records:
    prefix = f"User: {record['user']}\nAssistant:"
    answer = f" {record['assistant']}\n"
    separate_ids = tokenizer.encode(prefix).ids + tokenizer.encode(answer).ids
    full_ids = tokenizer.encode(prefix + answer).ids
    if separate_ids != full_ids:
        mismatches += 1

print(f"Records checked: {len(records)}")
print(f"Separate-vs-full tokenization mismatches: {mismatches}")
