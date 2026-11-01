"""
diagnose_model.py
Stage A: Comprehensive Diagnostic Analysis of custom MiniGPT checkpoint (best_gpt.pt).
Does NOT modify or retrain weights.
Evaluates:
- 30 diverse test prompts under both Greedy (argmax) and Temperature 0.7 sampling.
- Prompt token length, Output token length, Word overlap between prompt & response.
- Corpus frequency analysis of key words ('naam', 'kaha', 'kahan', 'hai') in data/train.txt and data/data.txt.
Saves detailed report to logs/diagnosis.txt.
"""

import os
import re
import sys
import string
import torch
from tokenizers import Tokenizer
from gpt_model import MiniGPT, GPTConfig

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

CHECKPOINT_PATH = os.path.join("checkpoints", "best_gpt.pt")
TOKENIZER_PATH = os.path.join("data", "bpe_tokenizer.json")
REPORT_PATH = os.path.join("logs", "diagnosis.txt")

# 30 Diagnostic Test Prompts across relevant intent categories
DIAGNOSTIC_PROMPTS = [
    # 1. Identity & Name (User's failing cases)
    "tumhara naam kya hai",
    "mera kya naam hai",
    "who are you",
    "what is your name",
    "tum kaun ho",
    "what are you called",
    "mera naam batao",
    "do you know my name",

    # 2. Location & Geography (User's failing cases)
    "indore kaha hai",
    "Indore kaha aata hai",
    "delhi kaha hai",
    "where is India located",
    "mumbai kahan hai",
    "taj mahal kaha sthit hai",

    # 3. Greetings
    "hello",
    "hi",
    "namaste",
    "good morning",

    # 4. Time & Date
    "what time is it",
    "aaj kya date hai",
    "kitna samay hua",
    "current time",

    # 5. Gratitude & Parting
    "thank you",
    "shukriya dost",
    "bye",
    "alvida",

    # 6. Capabilities & Knowledge
    "what can you do",
    "kya tum Hindi bol sakte ho",
    "what is artificial intelligence",
    "can you help me with python"
]


def clean_words(text: str) -> set:
    """Extracts alphanumeric words in lowercase without punctuation."""
    clean = text.lower().translate(str.maketrans("", "", string.punctuation))
    return set(clean.split())


def check_word_overlap(prompt: str, output: str) -> tuple[bool, list[str]]:
    """Checks if any meaningful word from prompt appears in the model output."""
    p_words = clean_words(prompt)
    o_words = clean_words(output)
    # Exclude common single-letter noise
    meaningful_p = {w for w in p_words if len(w) > 1}
    overlap = meaningful_p.intersection(o_words)
    return len(overlap) > 0, sorted(list(overlap))


def extract_assistant_reply(full_text: str, prompt_text: str) -> str:
    """Extracts only the assistant's immediate response turn."""
    if "Assistant:" in full_text:
        reply = full_text.split("Assistant:", 1)[1].strip()
    else:
        reply = full_text[len(prompt_text):].strip()

    for stop in ["\nUser:", "\nAssistant:", "\n\n"]:
        if stop in reply:
            reply = reply.split(stop)[0]

    return reply.strip()


def analyze_corpus_keywords(file_path: str, keywords: list[str]) -> dict:
    """Counts how many dialogue examples in file contain the target keywords."""
    if not os.path.exists(file_path):
        return {"exists": False, "counts": {kw: 0 for kw in keywords}, "total_blocks": 0}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = content.split("\n\n")
    total_blocks = len(blocks)
    counts = {kw: 0 for kw in keywords}

    for b in blocks:
        b_lower = b.lower()
        for kw in keywords:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, b_lower):
                counts[kw] += 1

    return {
        "exists": True,
        "file_name": os.path.basename(file_path),
        "total_blocks": total_blocks,
        "counts": counts
    }


def run_diagnostics():
    print("=" * 76)
    print("🔬 MiniGPT Model Diagnosis Tool (Stage A)")
    print(f"Target Checkpoint : {CHECKPOINT_PATH}")
    print(f"Target Tokenizer  : {TOKENIZER_PATH}")
    print("=" * 76)

    # 1. Load Tokenizer & Model
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    cfg = GPTConfig(**ckpt["config"])
    cfg.device = device
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    step = ckpt.get("step", "N/A")
    val_loss = ckpt.get("val_loss", 0.0)
    num_params = model.get_num_params()

    print(f"Model Specs : {cfg.n_layer} Layers | {cfg.n_head} Heads | {cfg.n_embd} Dim | {num_params:,d} Params")
    print(f"Checkpoint  : Step {step} | Val Loss: {val_loss:.4f} | Device: {device.upper()}\n")

    # 2. Analyze Corpus Keywords
    keywords = ["naam", "kaha", "kahan", "hai"]
    train_stats = analyze_corpus_keywords(os.path.join("data", "train.txt"), keywords)
    data_stats = analyze_corpus_keywords(os.path.join("data", "data.txt"), keywords)

    # 3. Run Generative Evaluations for 30 Prompts
    results = []
    print("Running generative diagnosis on 30 test prompts (Greedy & Temp 0.7)...")

    for i, p in enumerate(DIAGNOSTIC_PROMPTS, 1):
        prompt_formatted = f"User: {p}\nAssistant:"
        prompt_ids = tokenizer.encode(prompt_formatted).ids
        prompt_len = len(prompt_ids)
        idx_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Mode A: Greedy Decoding (temperature=1e-5, top_k=1, top_p=1.0)
        with torch.no_grad():
            greedy_ids = model.generate(
                idx_tensor,
                max_new_tokens=40,
                temperature=1e-5,
                top_k=1,
                top_p=1.0,
                repetition_penalty=1.0
            )[0].tolist()

        greedy_full = tokenizer.decode(greedy_ids)
        greedy_reply = extract_assistant_reply(greedy_full, prompt_formatted)
        greedy_reply_ids = tokenizer.encode(greedy_reply).ids
        greedy_reply_len = len(greedy_reply_ids)
        greedy_has_overlap, greedy_overlap_words = check_word_overlap(p, greedy_reply)

        # Mode B: Sampling (temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1)
        with torch.no_grad():
            sample_ids = model.generate(
                idx_tensor,
                max_new_tokens=40,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.1
            )[0].tolist()

        sample_full = tokenizer.decode(sample_ids)
        sample_reply = extract_assistant_reply(sample_full, prompt_formatted)
        sample_reply_ids = tokenizer.encode(sample_reply).ids
        sample_reply_len = len(sample_reply_ids)
        sample_has_overlap, sample_overlap_words = check_word_overlap(p, sample_reply)

        results.append({
            "id": i,
            "prompt": p,
            "prompt_tokens": prompt_len,
            "greedy": {
                "reply": greedy_reply,
                "tokens": greedy_reply_len,
                "has_overlap": greedy_has_overlap,
                "overlap_words": greedy_overlap_words
            },
            "sample": {
                "reply": sample_reply,
                "tokens": sample_reply_len,
                "has_overlap": sample_has_overlap,
                "overlap_words": sample_overlap_words
            }
        })

    # 4. Generate Report Text
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as rep:
        rep.write("=" * 80 + "\n")
        rep.write("MINI-GPT MODEL DIAGNOSIS REPORT (STAGE A)\n")
        rep.write("=" * 80 + "\n\n")

        rep.write("1. MODEL CONFIGURATION & CHECKPOINT DETAILS\n")
        rep.write("-" * 50 + "\n")
        rep.write(f"Checkpoint File      : {CHECKPOINT_PATH}\n")
        rep.write(f"Tokenizer File       : {TOKENIZER_PATH}\n")
        rep.write(f"Parameters           : {num_params:,d}\n")
        rep.write(f"Layers               : {cfg.n_layer}\n")
        rep.write(f"Attention Heads      : {cfg.n_head}\n")
        rep.write(f"Embedding Dim        : {cfg.n_embd}\n")
        rep.write(f"Context Window       : {cfg.block_size} tokens\n")
        rep.write(f"Checkpoint Step      : {step}\n")
        rep.write(f"Validation Loss      : {val_loss:.4f}\n\n")

        rep.write("2. TRAINING CORPUS KEYWORD FREQUENCY ANALYSIS\n")
        rep.write("-" * 50 + "\n")
        rep.write("Checking target keywords ['naam', 'kaha', 'kahan', 'hai'] across training datasets:\n\n")

        for s in [train_stats, data_stats]:
            if s.get("exists"):
                rep.write(f"File: {s['file_name']} (Total Dialogue Blocks: {s['total_blocks']:,d})\n")
                for kw, cnt in s["counts"].items():
                    pct = (cnt / max(1, s["total_blocks"])) * 100
                    rep.write(f"  • '{kw:<6}': {cnt:>6,d} blocks ({pct:.2f}%)\n")
                rep.write("\n")
            else:
                rep.write(f"File: {s.get('file_name', 'data file')} - Not Found\n\n")

        rep.write("3. GENERATIVE BEHAVIOR EVALUATION (30 PROMPTS)\n")
        rep.write("-" * 80 + "\n")
        rep.write("Greedy Decoding: temp=0.0 (argmax) | Sampling: temp=0.7, top_k=40, top_p=0.9\n")
        rep.write("=" * 80 + "\n\n")

        greedy_overlap_count = sum(1 for r in results if r["greedy"]["has_overlap"])
        sample_overlap_count = sum(1 for r in results if r["sample"]["has_overlap"])

        for r in results:
            rep.write(f"[{r['id']:02d}] PROMPT: \"{r['prompt']}\" (Tokens: {r['prompt_tokens']})\n")
            
            # Greedy details
            g = r["greedy"]
            g_ov_str = f"YES ({', '.join(g['overlap_words'])})" if g["has_overlap"] else "NO"
            rep.write(f"     [Greedy] (Tokens: {g['tokens']}, Word Overlap: {g_ov_str})\n")
            rep.write(f"     Reply: {repr(g['reply'])}\n")

            # Sample details
            s = r["sample"]
            s_ov_str = f"YES ({', '.join(s['overlap_words'])})" if s["has_overlap"] else "NO"
            rep.write(f"     [Temp0.7] (Tokens: {s['tokens']}, Word Overlap: {s_ov_str})\n")
            rep.write(f"     Reply: {repr(s['reply'])}\n")
            rep.write("-" * 80 + "\n")

        rep.write("\n4. SUMMARY STATISTICS & DIAGNOSTIC OBSERVATIONS\n")
        rep.write("-" * 50 + "\n")
        rep.write(f"Total Evaluated Prompts          : {len(results)}\n")
        rep.write(f"Greedy Overlap with Prompt Words : {greedy_overlap_count}/{len(results)} ({greedy_overlap_count/len(results)*100:.1f}%)\n")
        rep.write(f"Temp0.7 Overlap with Prompt Words: {sample_overlap_count}/{len(results)} ({sample_overlap_count/len(results)*100:.1f}%)\n")

    print(f"\n✅ Diagnostic Report successfully generated and saved to: {REPORT_PATH}")
    return REPORT_PATH, results, train_stats, data_stats


if __name__ == "__main__":
    run_diagnostics()
