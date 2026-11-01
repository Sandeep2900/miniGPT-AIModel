# Compare the original and v2 checkpoints on held-out user prompts.

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
APPROVED_DIR = Path(__file__).resolve().parent
PROMPTS_FILE = APPROVED_DIR / "eval_prompts.txt"
TOKENIZER_FILE = ROOT / "data" / "bpe_tokenizer.json"
BASE_CHECKPOINT = ROOT / "checkpoints" / "best_gpt.pt"
V2_CHECKPOINT = ROOT / "checkpoints" / "v2" / "best.pt"
OUTPUT_FILE = APPROVED_DIR / "eval_outputs_v2.txt"
MAX_NEW_TOKENS = 60
TEMPERATURE = 1e-5
TOP_K = 1
TOP_P = 1.0
REPETITION_PENALTY = 1.0
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def load_prompts():
    """Read only the labelled user-prompt file."""
    prompts = []
    for line_number, line in enumerate(PROMPTS_FILE.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if "|" not in line:
            raise ValueError(f"Malformed prompt line: {line_number}")
        lang, prompt = line.split("|", 1)
        prompts.append((lang.strip(), prompt.strip()))
    return prompts


def script_of(text: str) -> str:
    """Classify reply script by its Devanagari and Latin letter counts."""
    devanagari = sum(1 for char in text if "\u0900" <= char <= "\u097f" and char.isalpha())
    latin = sum(1 for char in text if char.isascii() and char.isalpha())
    if devanagari and latin:
        return "mixed"
    if devanagari:
        return "hi"
    if latin:
        return "latin"
    return "other"


def expected_script_matches(lang: str, reply_script: str) -> bool:
    """Check only script compatibility, not semantic correctness."""
    if lang == "hi":
        return reply_script == "hi"
    if lang in {"hinglish", "en"}:
        return reply_script in {"latin", "mixed"}
    return False


def repeated_three_word_grams(text: str) -> bool:
    """Flag any overlapping three-word phrase repeated at least three times."""
    words = [word.casefold() for word in WORD_PATTERN.findall(text)]
    grams = Counter(tuple(words[index : index + 3]) for index in range(max(0, len(words) - 2)))
    return any(count >= 3 for count in grams.values())


def load_model(checkpoint_path: Path, tokenizer):
    """Load a checkpoint on CPU without changing its weights."""
    sys.path.insert(0, str(ROOT))
    from gpt_model import GPTConfig, MiniGPT

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    config.device = "cpu"
    model = MiniGPT(config).to("cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def generate_reply(model, tokenizer, user_prompt: str) -> str:
    """Generate with the fixed greedy decoding configuration."""
    prompt = f"User: {user_prompt}\nAssistant:"
    token_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
    with torch.inference_mode():
        output_ids = model.generate(
            token_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
        )[0].tolist()
    decoded = tokenizer.decode(output_ids)
    reply = decoded.split("Assistant:", 1)[-1]
    for stop in ("\nUser:", "\nAssistant:", "\n\n"):
        reply = reply.split(stop, 1)[0]
    return reply.strip()


def run_checkpoint(label, path, prompts, tokenizer):
    model, checkpoint = load_model(path, tokenizer)
    records = []
    for index, (lang, prompt) in enumerate(prompts, 1):
        reply = generate_reply(model, tokenizer, prompt)
        reply_script = script_of(reply)
        records.append(
            {
                "checkpoint": label,
                "checkpoint_step": checkpoint.get("step"),
                "lang": lang,
                "prompt": prompt,
                "reply": reply,
                "empty": not bool(reply),
                "prompt_leakage": "User:" in reply or "Assistant:" in reply,
                "repeated_three_word_ngram": repeated_three_word_grams(reply),
                "reply_script": reply_script,
                "script_matches_label": expected_script_matches(lang, reply_script),
                "rating": "",
                "prompt_index": index,
            }
        )
    return records


def print_summary(label, records):
    """Print descriptive counts without calculating accuracy or a score."""
    replies = [record["reply"] for record in records]
    reply_counts = Counter(replies)
    repeated_outputs = sum(reply_counts[reply] > 1 for reply in replies)
    print(f"[{label}] prompts={len(records)}")
    print(f"  replies_in_expected_script={sum(record['script_matches_label'] for record in records)}")
    print(f"  empty_replies={sum(record['empty'] for record in records)}")
    print(f"  prompt_leakage={sum(record['prompt_leakage'] for record in records)}")
    print(f"  replies_with_repeated_3gram={sum(record['repeated_three_word_ngram'] for record in records)}")
    print(f"  replies_repeated_across_prompts={repeated_outputs}")


def main():
    if OUTPUT_FILE.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation output: {OUTPUT_FILE}")
    if not BASE_CHECKPOINT.is_file() or not V2_CHECKPOINT.is_file():
        raise FileNotFoundError("Both baseline and v2 checkpoints must exist.")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    prompts = load_prompts()
    all_records = []
    for label, checkpoint_path in (("baseline", BASE_CHECKPOINT), ("v2", V2_CHECKPOINT)):
        records = run_checkpoint(label, checkpoint_path, prompts, tokenizer)
        all_records.extend(records)

    with OUTPUT_FILE.open("x", encoding="utf-8", newline="\n") as output:
        for record in all_records:
            output.write(
                f"[{record['checkpoint'].upper()} {record['prompt_index']:02d}] LANG: {record['lang']}\n"
                f"PROMPT : {record['prompt']}\n"
                f"REPLY  : {record['reply']}\n"
                f"RATING : {record['rating']}\n\n"
            )

    print(f"Decoding: temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}, max_new_tokens={MAX_NEW_TOKENS}, repetition_penalty={REPETITION_PENALTY}")
    print(f"Prompt file: {PROMPTS_FILE}")
    print_summary("baseline", [record for record in all_records if record["checkpoint"] == "baseline"])
    print_summary("v2", [record for record in all_records if record["checkpoint"] == "v2"])
    print(f"Full prompt/reply output: {OUTPUT_FILE}")
    print("Ratings are blank; no automatic OK/WRONG judgment was made.")


if __name__ == "__main__":
    main()
