"""
test_qwen_agent.py
Tests Qwen2.5-0.5B integrated with Clock, Search, and Memory.
"""

import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tools.clock import get_current_time
from tools.search import web_search
from session_memory import SessionMemory

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

class QwenAgent:
    def __init__(self):
        print("Loading Qwen2.5-0.5B-Instruct into memory...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        self.model.eval()
        self.memory = SessionMemory(max_history=8)
        print("✓ QwenAgent initialized successfully.")

    def respond(self, user_text: str) -> dict:
        clean_user = user_text.strip()
        lower = clean_user.lower()

        # 1. Memory: Name Questions / Declarations
        name_ans = self.memory.answer_name_question(clean_user)
        if name_ans:
            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", name_ans)
            return {"response": name_ans, "route": "memory"}

        extracted_name = self.memory.extract_name(clean_user)
        if extracted_name:
            reply = f"Namaste {extracted_name}! Main aapka naam yaad rakhunga. Aap bataiye, aaj main aapki kya madad kar sakta hoon?"
            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", reply)
            return {"response": reply, "route": "memory"}

        # 2. Clock: Exact time
        if any(w in lower for w in ["time kya", "kya time", "samay kya", "kitna baje", "what time", "current time"]):
            curr = get_current_time()
            reply = f"Abhi ka samay hai: {curr}."
            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", reply)
            return {"response": reply, "route": "clock"}

        # 3. Live search for dynamic queries (weather, live prices, news)
        search_context = ""
        sources = []
        is_live = any(w in lower for w in ["weather", "mausam", "gold price", "share market", "news", "aaj ka score"])
        if is_live:
            res, err = web_search(clean_user, max_results=2)
            if res:
                sources = [r.get("url") for r in res if r.get("url")]
                snippets = [f"{r.get('title')}: {r.get('snippet')}" for r in res]
                search_context = "\n".join(snippets[:2])

        # 4. Neural generation with Qwen
        system_prompt = (
            "You are a helpful, smart, and friendly AI assistant. "
            "Reply naturally in Hinglish, Hindi, or English matching the user's style. "
            "Give clear, accurate, and concise answers."
        )
        if search_context:
            system_prompt += f"\n\nReal-time web search context:\n{search_context}\nUse this live context to answer accurately."

        messages = [{"role": "system", "content": system_prompt}]
        # Add past turns from memory
        for turn in self.memory.history[-4:]:
            messages.append({"role": turn["role"], "content": turn["text"]})
        messages.append({"role": "user", "content": clean_user})

        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([chat_text], return_tensors="pt")

        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=90,
                temperature=0.3,
                top_p=0.9,
                do_sample=True
            )
        latency = time.time() - t0
        prompt_len = inputs.input_ids.shape[1]
        reply_tokens = outputs[0][prompt_len:]
        reply = self.tokenizer.decode(reply_tokens, skip_special_tokens=True).strip()

        self.memory.add_turn("user", clean_user)
        self.memory.add_turn("assistant", reply)
        return {"response": reply, "route": "qwen_neural", "latency": latency, "sources": sources}


def main():
    agent = QwenAgent()
    test_suite = [
        "name",
        "Mera naam Sandeep hai",
        "kaun si list bhai",
        "indore kaha h",
        "hindu kaun sa religion hai",
        "how many months are there in a week",
        "Mera kya naam hai?"
    ]

    print("\n" + "=" * 70)
    print("RUNNING SCREENSHOT QUESTIONS THROUGH QWEN AGENT")
    print("=" * 70)

    for q in test_suite:
        res = agent.respond(q)
        print(f"User:     {q}")
        print(f"Route:    {res['route']}")
        print(f"Response: {res['response']}")
        if "latency" in res:
            print(f"Latency:  {res['latency']:.2f}s")
        print("-" * 70)

if __name__ == "__main__":
    main()
