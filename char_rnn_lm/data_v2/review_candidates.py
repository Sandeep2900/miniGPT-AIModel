# Read-only review of candidate conversation pairs.
# It does NOT change the input file. It only prints a report.

import json
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

PATH = "data_v2/candidates/conversation_candidates.jsonl"
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")


def script_of(text):
    # Simple script check: which alphabet appears more.
    dev = len(DEVANAGARI.findall(text))
    lat = len(LATIN.findall(text))
    if dev == 0 and lat == 0:
        return "none"
    return "devanagari" if dev > lat else "latin"


records = []
with open(PATH, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if line.strip():
            records.append((i, json.loads(line)))

print(f"Total records: {len(records)}")

# 1. Exact duplicate pairs
pair_counts = Counter((r["user"], r["assistant"]) for _, r in records)
dup_pairs = [p for p, c in pair_counts.items() if c > 1]
print(f"Exact duplicate pairs: {len(dup_pairs)}")

# 2. Same user text used more than once
user_counts = Counter(r["user"] for _, r in records)
same_user = {u: c for u, c in user_counts.items() if c > 1}
print(f"User texts used more than once: {len(same_user)}")

# 3. Language tag vs actual script
mismatch = []
for ln, r in records:
    s = script_of(r["user"])
    tag = r["lang"]
    if tag == "hi" and s != "devanagari":
        mismatch.append((ln, tag, s, r["user"]))
    if tag == "en" and s == "devanagari":
        mismatch.append((ln, tag, s, r["user"]))
print(f"Language tag vs script mismatches: {len(mismatch)}")
for ln, tag, s, u in mismatch[:10]:
    print(f"  line {ln} tag={tag} script={s} user={u}")

# 4. Very short replies (likely too thin to teach anything)
short = [(ln, r["assistant"]) for ln, r in records if len(r["assistant"].split()) < 4]
print(f"Replies under 4 words: {len(short)}")
for ln, a in short[:10]:
    print(f"  line {ln}: {a}")

# 5. Factual-question hints (these belong to search/clock, not the model)
fact_hint = re.compile(
    r"\b(full form|kaha|kahan|where is|what is the time|kya time|date|weather|news|price)\b",
    re.I,
)
facts = [(ln, r["user"]) for ln, r in records if fact_hint.search(r["user"])]
print(f"Possible factual or live questions: {len(facts)}")
for ln, u in facts[:10]:
    print(f"  line {ln}: {u}")

# 6. Replies that mirror one fixed opener (style repetition)
openers = Counter(r["assistant"].split()[0] for _, r in records if r["assistant"].split())
print("Most common reply openers:")
for word, count in openers.most_common(5):
    print(f"  {word}: {count}")
