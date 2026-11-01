# Run deterministic model checks and save raw replies without changing inputs.

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "best_gpt.pt"
TOKENIZER_PATH = ROOT / "data" / "bpe_tokenizer.json"
INTENTS_PATH = ROOT / "data" / "intents_test.jsonl"
CANDIDATES_PATH = ROOT / "data_v2" / "candidates" / "conversation_candidates.jsonl"
OUTPUT_PATH = Path(__file__).resolve().parent / "raw_replies.jsonl"
MAX_NEW_TOKENS = 60
TEMPERATURE = 1e-5
TOP_K = 1
TOP_P = 1.0
REPETITION_PENALTY = 1.0

# Match Unicode words while keeping apostrophes inside contractions.
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def load_inputs() -> list[dict[str, object]]:
    """Read held-out intent phrases and candidate user prompts only."""
    records: list[dict[str, object]] = []

    with INTENTS_PATH.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            records.append(
                {
                    "set": "intent",
                    "source_line": line_number,
                    "intent": item["intent"],
                    "prompt": item["phrase"],
                }
            )

    with CANDIDATES_PATH.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            # Deliberately read only the user field; candidate replies are not gold answers.
            records.append(
                {
                    "set": "candidate",
                    "source_line": line_number,
                    "intent": None,
                    "prompt": item["user"],
                }
            )

    return records


def extract_reply(decoded_text: str) -> str:
    """Keep only the generated assistant response and stop at a new turn."""
    reply = decoded_text.split("Assistant:", 1)[-1]
    for marker in ("\nUser:", "\nAssistant:", "\n\n"):
        reply = reply.split(marker, 1)[0]
    return reply.strip()


def repeated_three_word_grams(reply: str) -> list[dict[str, object]]:
    """Find overlapping three-word phrases repeated at least three times."""
    words = [word.casefold() for word in WORD_PATTERN.findall(reply)]
    counts: collections.Counter[tuple[str, str, str]] = collections.Counter(
        tuple(words[index : index + 3]) for index in range(max(0, len(words) - 2))
    )
    return [
        {"phrase": " ".join(phrase), "occurrences": count}
        for phrase, count in sorted(counts.items())
        if count >= 3
    ]


def script_counts(reply: str) -> dict[str, int | str]:
    """Count Devanagari and Latin letters, then label the reply script."""
    devanagari = sum(1 for char in reply if "\u0900" <= char <= "\u097f" and char.isalpha())
    latin = sum(1 for char in reply if char.isascii() and char.isalpha())
    if devanagari and latin:
        label = "Mixed"
    elif devanagari:
        label = "Devanagari"
    elif latin:
        label = "Latin"
    else:
        label = "Other"
    return {"label": label, "devanagari_chars": devanagari, "latin_chars": latin}


def generate_replies(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Generate each reply with the same CPU greedy-decoding settings."""
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    sys.path.insert(0, str(ROOT))
    from gpt_model import GPTConfig, MiniGPT

    config = GPTConfig(**checkpoint["config"])
    config.device = "cpu"
    model = MiniGPT(config).to("cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generated_records: list[dict[str, object]] = []
    with torch.inference_mode():
        for record in records:
            prompt = f"User: {record['prompt']}\nAssistant:"
            prompt_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
            output_ids = model.generate(
                prompt_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
            )[0].tolist()
            reply = extract_reply(tokenizer.decode(output_ids))
            repeated_grams = repeated_three_word_grams(reply)
            leakage = "User:" in reply or "Assistant:" in reply

            generated_records.append(
                {
                    **record,
                    "reply": reply,
                    "reply_token_count": len(tokenizer.encode(reply).ids),
                    "is_empty": not bool(reply),
                    "prompt_leakage": leakage,
                    "repeated_three_word_ngram": bool(repeated_grams),
                    "repeated_three_word_ngrams": repeated_grams,
                    "reply_script": script_counts(reply),
                }
            )

    # Count identical replies across all prompts, excluding the current prompt itself.
    reply_counts = collections.Counter(record["reply"] for record in generated_records)
    for record in generated_records:
        identical_count = reply_counts[record["reply"]] - 1
        record["copy_collapse_count"] = identical_count
        record["identical_to_another_prompt"] = identical_count > 0

    return generated_records


def print_summary(records: list[dict[str, object]], checkpoint_step: object) -> None:
    """Print automatic counts only; do not calculate accuracy or a score."""
    print(f"Checkpoint step: {checkpoint_step}")
    print(
        "Decoding: "
        f"temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}, "
        f"max_new_tokens={MAX_NEW_TOKENS}, repetition_penalty={REPETITION_PENALTY}"
    )

    for set_name in ("intent", "candidate"):
        subset = [record for record in records if record["set"] == set_name]
        print(f"[{set_name}] prompts={len(subset)}")
        print(f"  empty_replies={sum(bool(record['is_empty']) for record in subset)}")
        print(f"  prompt_leakage={sum(bool(record['prompt_leakage']) for record in subset)}")
        print(
            "  replies_with_repeated_3gram="
            f"{sum(bool(record['repeated_three_word_ngram']) for record in subset)}"
        )
        for label in ("Devanagari", "Latin", "Mixed", "Other"):
            count = sum(record["reply_script"]["label"] == label for record in subset)
            print(f"  script_{label.lower()}={count}")
        print(
            "  replies_identical_to_another_prompt="
            f"{sum(bool(record['identical_to_another_prompt']) for record in subset)}"
        )
        print(
            "  copy_collapse_count_total="
            f"{sum(int(record['copy_collapse_count']) for record in subset)}"
        )


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT_PATH}")

    input_records = load_inputs()
    checkpoint_header = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    generated = generate_replies(input_records)

    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as output:
        for record in generated:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print_summary(generated, checkpoint_header.get("step"))
    print(f"Wrote raw replies: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
