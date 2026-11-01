"""
eval_intents.py
Stage C: Measures baseline intent accuracy, exact-format rate, and token length
on a strictly held-out test split (20% of phrases per intent).
Saves results to logs/baseline_eval.json.
"""

import os
import sys
import json
import random
import re
import torch
from tokenizers import Tokenizer
from gpt_model import MiniGPT, GPTConfig

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

INTENTS_FILE = os.path.join("data", "intents.jsonl")
TRAIN_SPLIT_FILE = os.path.join("data", "intents_train.jsonl")
TEST_SPLIT_FILE = os.path.join("data", "intents_test.jsonl")
BASELINE_JSON = os.path.join("logs", "baseline_eval.json")
TOKENIZER_PATH = os.path.join("data", "bpe_tokenizer.json")
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "best_gpt.pt")


def split_intents_dataset(train_ratio: float = 0.8, seed: int = 42):
    """
    Splits phrases stratified by intent into 80% train and 20% test.
    Ensures held-out phrases are completely unseen during training.
    """
    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    grouped = {}
    for r in records:
        grouped.setdefault(r["intent"], []).append(r)

    train_records = []
    test_records = []

    random.seed(seed)
    for intent, items in grouped.items():
        # Shuffle deterministically
        shuffled = list(items)
        random.shuffle(shuffled)
        split_point = int(len(shuffled) * train_ratio)
        train_records.extend(shuffled[:split_point])
        test_records.extend(shuffled[split_point:])

    with open(TRAIN_SPLIT_FILE, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(TEST_SPLIT_FILE, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Dataset split complete: {len(train_records)} Train phrases, {len(test_records)} Test phrases.")
    return train_records, test_records


def has_word(text: str, word: str) -> bool:
    r"""
    Checks if a target word/phrase exists in text with strict word boundaries.
    Uses regex lookaround (?<![\w])word(?![\w]) with re.escape and re.IGNORECASE.
    Note: Python 3 regex \w natively matches Unicode letters (including Devanagari).
    """
    if not text or not word:
        return False
    pattern = rf"(?<![\w]){re.escape(word)}(?![\w])"
    return bool(re.search(pattern, text, re.IGNORECASE))


def classify_response_intent(reply_text: str, expected_intent: str) -> bool:
    """
    Transparent, readable rule-based classifier scoring if output matches target intent.
    Now uses strict word-boundary matching via has_word() instead of loose substring checks.
    """
    clean = reply_text.strip().lower()
    if not clean:
        return False

    if expected_intent == "greeting":
        keywords = ["hello", "namaste", "badhiya", "help", "madad", "assist", "swagat", "good morning", "subah", "hey", "hi", "how"]
        return any(has_word(clean, kw) for kw in keywords) and ("[tool:" not in clean)

    elif expected_intent == "identity":
        keywords = ["mini-gpt", "assistant", "ai", "scratch", "train", "transformer", "model"]
        return any(has_word(clean, kw) for kw in keywords) and not has_word(clean, "aapka naam nahi")

    elif expected_intent == "user_name_unknown":
        # Must explicitly acknowledge it does NOT know the user's name
        unknown_markers = ["nahi pata", "nahi maloom", "do not know", "personal data", "introduce", "naam nahi"]
        # Must NOT confuse user's name with assistant's name
        is_confused = has_word(clean, "mera naam mini-gpt") or has_word(clean, "my name is mini-gpt")
        return any(has_word(clean, m) for m in unknown_markers) and not is_confused

    elif expected_intent == "location_question":
        # Must produce valid tool search call for autonomous web search
        pattern = r'^\[tool: search\("([^"\r\n]+)"\)\]$'
        return bool(re.match(pattern, clean, re.IGNORECASE))

    elif expected_intent == "time_question":
        keywords = ["clock", "samay", "time", "ghadi", "taskbar", "live"]
        return any(has_word(clean, kw) for kw in keywords)

    elif expected_intent == "date_question":
        keywords = ["date", "tarikh", "taarikh", "din", "clock", "calendar"]
        return any(has_word(clean, kw) for kw in keywords)

    elif expected_intent == "thanks":
        keywords = ["welcome", "swagat", "pleasure", "khushi", "madad", "anytime"]
        return any(has_word(clean, kw) for kw in keywords)

    elif expected_intent == "goodbye":
        keywords = ["alvida", "goodbye", "bye", "care", "phir milenge", "ahead", "soon", "talk", "night", "later", "time"]
        return any(has_word(clean, kw) for kw in keywords)

    return False


def is_exact_format(raw_output: str, reply_text: str) -> bool:
    """Checks if output contains clean dialogue turn without prompt leakage or corrupt tags."""
    if not reply_text:
        return False
    # No prompt leakages
    if "User:" in reply_text or "Assistant:" in reply_text:
        return False
    # If tool call, brackets must be balanced
    if "[TOOL:" in reply_text and not reply_text.endswith(")]"):
        return False
    return True


def evaluate_checkpoint(checkpoint_path: str = DEFAULT_CHECKPOINT):
    print("=" * 80)
    print("📊 Stage C: Baseline Intent Evaluation on Held-out Test Set")
    print(f"Target Checkpoint : {checkpoint_path}")
    print("=" * 80)

    # 1. Create Train/Test Splits
    train_records, test_records = split_intents_dataset(train_ratio=0.8, seed=42)

    # 2. Load Model & Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = GPTConfig(**ckpt["config"])
    cfg.device = device
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    step = ckpt.get("step", "N/A")
    val_loss = ckpt.get("val_loss", 0.0)

    # 3. Evaluate on All 32 Held-out Test Phrases
    intent_stats = {}
    total_evaluated = 0
    total_correct = 0
    total_exact_format = 0
    total_tokens = 0
    detailed_results = []

    print(f"\nEvaluating {len(test_records)} test phrases with checkpoint Step {step}...")

    for item in test_records:
        intent = item["intent"]
        phrase = item["phrase"]
        total_evaluated += 1

        prompt = f"User: {phrase}\nAssistant:"
        prompt_ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            gen_ids = model.generate(
                idx,
                max_new_tokens=30,
                temperature=1e-5,  # Greedy argmax for deterministic benchmark
                top_k=1,
                top_p=1.0,
                repetition_penalty=1.0
            )[0].tolist()

        full_gen = tokenizer.decode(gen_ids)
        if "Assistant:" in full_gen:
            reply = full_gen.split("Assistant:", 1)[1].strip()
        else:
            reply = full_gen[len(prompt):].strip()

        for stop in ["\nUser:", "\nAssistant:", "\n\n"]:
            if stop in reply:
                reply = reply.split(stop)[0]
        reply = reply.strip()

        reply_token_count = len(tokenizer.encode(reply).ids)
        total_tokens += reply_token_count

        correct = classify_response_intent(reply, intent)
        valid_format = is_exact_format(full_gen, reply)

        if correct:
            total_correct += 1
        if valid_format:
            total_exact_format += 1

        if intent not in intent_stats:
            intent_stats[intent] = {"total": 0, "correct": 0}
        intent_stats[intent]["total"] += 1
        if correct:
            intent_stats[intent]["correct"] += 1

        detailed_results.append({
            "intent": intent,
            "phrase": phrase,
            "reply": reply,
            "correct": correct,
            "valid_format": valid_format,
            "tokens": reply_token_count
        })

    # 4. Compute Metrics
    intent_accuracy = (total_correct / max(1, total_evaluated)) * 100
    format_rate = (total_exact_format / max(1, total_evaluated)) * 100
    avg_token_len = total_tokens / max(1, total_evaluated)

    metrics = {
        "checkpoint": checkpoint_path,
        "checkpoint_step": step,
        "base_val_loss": round(val_loss, 4),
        "total_test_phrases": total_evaluated,
        "intent_accuracy_pct": round(intent_accuracy, 2),
        "exact_format_rate_pct": round(format_rate, 2),
        "avg_token_length": round(avg_token_len, 2),
        "per_intent_accuracy": {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy_pct": round((v["correct"] / max(1, v["total"])) * 100, 1)
            }
            for k, v in intent_stats.items()
        },
        "test_results": detailed_results
    }

    # 5. Save to logs/baseline_eval.json
    os.makedirs(os.path.dirname(BASELINE_JSON), exist_ok=True)
    with open(BASELINE_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 6. Print Summary Table
    print("\n" + "=" * 80)
    print("📈 BASELINE EVALUATION RESULTS (checkpoints/best_gpt.pt)")
    print("=" * 80)
    print(f"{'#':<3} | {'Intent Category':<22} | {'Test Total':<10} | {'Correct':<8} | {'Accuracy'}")
    print("-" * 80)
    for i, (intent, data) in enumerate(metrics["per_intent_accuracy"].items(), 1):
        print(f"{i:<3} | {intent:<22} | {data['total']:<10} | {data['correct']:<8} | {data['accuracy_pct']:>5.1f}%")
    print("-" * 80)
    print(f"Overall Intent Accuracy : {total_correct}/{total_evaluated} ({intent_accuracy:.1f}%)")
    print(f"Exact Format Rate       : {total_exact_format}/{total_evaluated} ({format_rate:.1f}%)")
    print(f"Average Token Length    : {avg_token_len:.1f} tokens")
    print("=" * 80)
    print(f"Saved complete metrics to: {BASELINE_JSON}")

    return metrics


def run_selftest():
    """
    Self-test asserting that has_word correctly rejects substring false positives
    and accepts valid whole words (both Latin and Devanagari).
    """
    print("=" * 70)
    print("RUNNING eval_intents.py MATCHING SELF-TEST...")
    print("=" * 70)

    # Assert 1: "hi" must not match inside "think"
    res1 = has_word("I think i'll be there.", "hi")
    assert res1 is False, f"Expected False, but got {res1}"
    print("[PASS] has_word(\"I think i'll be there.\", \"hi\") is False")

    # Assert 2: "hello" matches standalone word
    res2 = has_word("Hello! How can I help you today?", "hello")
    assert res2 is True, f"Expected True, but got {res2}"
    print("[PASS] has_word(\"Hello! How can I help you today?\", \"hello\") is True")

    # Assert 3: "swagat" matches in Romanized Hindi
    res3 = has_word("Aapka bahut swagat hai", "swagat")
    assert res3 is True, f"Expected True, but got {res3}"
    print("[PASS] has_word(\"Aapka bahut swagat hai\", \"swagat\") is True")

    # Assert 4: Devanagari word-boundary check
    res4 = has_word("कल सुबह", "सुबह")
    assert res4 is True, f"Expected True, but got {res4}"
    print("[PASS] has_word(\"कल सुबह\", \"सुबह\") is True (Devanagari supported)")

    print("=" * 70)
    print("ALL 4 ASSERTIONS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        run_selftest()
    else:
        evaluate_checkpoint()
