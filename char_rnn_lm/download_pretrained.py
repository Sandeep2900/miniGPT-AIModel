"""
download_pretrained.py
Downloads and verifies Qwen2.5-0.5B-Instruct locally for offline conversational inference.
Tests the exact questions from the user's screenshot.
"""

import os
import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def download_and_verify():
    print("=" * 70)
    print(f"Loading / Downloading Pretrained Model: {MODEL_ID}")
    print("=" * 70)
    start_t = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print("✓ Tokenizer loaded successfully.")

    # Using float32 for CPU inference compatibility and numerical stability
    print("Loading model weights (approx ~980 MB)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model.eval()
    elapsed = time.time() - start_t
    print(f"✓ Model loaded in {elapsed:.1f} seconds.")

    test_queries = [
        "how many months are there in a week",
        "indore kaha h",
        "hindu kaun sa religion hai",
        "Mera naam Sandeep hai, yaad rakhna",
        "Mera kya naam hai?"
    ]

    print("\n" + "=" * 70)
    print("TESTING USER PROMPTS ON PRETRAINED QWEN2.5-0.5B-INSTRUCT")
    print("=" * 70)

    for q in test_queries:
        messages = [
            {"role": "system", "content": "You are a helpful, smart AI assistant. Reply naturally in Hinglish, Hindi, or English matching the user's language."},
            {"role": "user", "content": q}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
        gen_time = time.time() - t0
        
        prompt_len = inputs.input_ids.shape[1]
        response_tokens = outputs[0][prompt_len:]
        reply = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        print(f"User:     {q}")
        print(f"Response: {reply}")
        print(f"Latency:  {gen_time:.2f}s ({len(response_tokens)} tokens, {len(response_tokens)/gen_time:.1f} tok/s)")
        print("-" * 70)

if __name__ == "__main__":
    download_and_verify()
