"""
eval_gpt.py
Automated evaluation testbench for Mini-GPT model.
Runs benchmark prompts across English, Hinglish, and Hindi.
Scores repetition, coherence, and length.
"""

import os
import sys
from chat_gpt import load_model_and_tokenizer, chat_turn

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BENCHMARK_PROMPTS = [
    # English Greetings & Capabilities
    "hi",
    "how are you",
    "who are you",
    "what can you do",
    "can you dance for me",
    # English Temporal & Short Queries (User's specific test cases)
    "what is the day today",
    "date",
    "what time is it",
    "help",
    # Hinglish Conversations
    "kaise ho",
    "aaj khane me kya khayein",
    "kuch acchi baat batao",
    # Hindi Conversations
    "नमस्ते!",
    "आपका नाम क्या है?",
    "सफलता पाने के लिए क्या करना चाहिए?"
]


def evaluate_model():
    print("=" * 70)
    print("📊 Automated Evaluation Benchmark for Custom Mini-GPT")
    print("=" * 70)

    try:
        model, tokenizer, cfg, step, val_loss, device = load_model_and_tokenizer()
        print(f"Loaded: Step {step} | Val Loss: {val_loss:.4f} | Params: {model.get_num_params():,d}\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    results = []
    total_valid_responses = 0

    for idx, prompt in enumerate(BENCHMARK_PROMPTS):
        reply = chat_turn(model, tokenizer, prompt, device=device, max_new_tokens=80, temperature=0.35)
        words = reply.split()
        unique_words = len(set(words))
        rep_ratio = (len(words) - unique_words) / max(1, len(words))  # Lower is better

        is_valid = len(words) >= 2 and rep_ratio < 0.6
        if is_valid:
            total_valid_responses += 1

        print(f"[{idx+1:2d}/{len(BENCHMARK_PROMPTS)}] User: {prompt}")
        print(f"     Assistant: {reply}")
        print(f"     Stats    : {len(words)} words, Repetition Ratio: {rep_ratio:.2f} ({'✅ PASS' if is_valid else '⚠️ WEAK'})\n")
        results.append({
            "prompt": prompt,
            "reply": reply,
            "words": len(words),
            "rep_ratio": rep_ratio,
            "valid": is_valid
        })

    pass_rate = (total_valid_responses / len(BENCHMARK_PROMPTS)) * 100
    print("=" * 70)
    print(f"Benchmark Results: {total_valid_responses}/{len(BENCHMARK_PROMPTS)} passed ({pass_rate:.1f}%)")
    print("=" * 70)
    return pass_rate >= 70.0


if __name__ == "__main__":
    evaluate_model()
