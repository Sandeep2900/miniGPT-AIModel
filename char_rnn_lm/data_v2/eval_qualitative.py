# Read-only qualitative test.
# Read only the user field from candidate pairs and print model replies.
# Save an empty rating column for a human reviewer.

import json
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "best_gpt.pt"
TOKENIZER = ROOT / "data" / "bpe_tokenizer.json"
CANDIDATES = ROOT / "data_v2" / "candidates" / "conversation_candidates.jsonl"
OUTPUT = ROOT / "logs" / "qualitative_eval.txt"
MAX_NEW_TOKENS = 60

sys.path.insert(0, str(ROOT))
from gpt_model import MiniGPT, GPTConfig


def load_user_prompts(path):
    """Read only each JSONL record's user field."""
    prompts = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                prompts.append(json.loads(line)["user"])
    return prompts


def main():
    # Refuse to overwrite a prior review sheet.
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {OUTPUT}")

    device = "cpu"
    tokenizer = Tokenizer.from_file(str(TOKENIZER))

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint["config"])
    config.device = device
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompts = load_user_prompts(CANDIDATES)
    print(f"Loaded {len(prompts)} user prompts. Checkpoint step {checkpoint.get('step')}.")

    output_lines = []
    with torch.no_grad():
        for index, user_text in enumerate(prompts, 1):
            prompt = f"User: {user_text}\nAssistant:"
            token_ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
            start_time = time.perf_counter()
            generated = model.generate(
                token_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=1e-5,
                top_k=1,
                top_p=1.0,
                repetition_penalty=1.0,
            )[0].tolist()
            elapsed = time.perf_counter() - start_time

            full_text = tokenizer.decode(generated)
            reply = full_text.split("Assistant:", 1)[-1]
            for stop_marker in ["\nUser:", "\nAssistant:", "\n\n"]:
                reply = reply.split(stop_marker)[0]
            reply = reply.strip()

            output_lines.append(f"[{index:02d}] PROMPT : {user_text}")
            output_lines.append(f"     REPLY  : {reply}")
            output_lines.append(f"     TIME   : {elapsed:.2f}s")
            output_lines.append("     RATING : ")
            output_lines.append("")
            print(f"[{index:02d}] done in {elapsed:.2f}s")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
