"""
tests/run_stage2_conversation_eval.py
Stage 2: Model Reply Evaluation on Conversational Prompts
Tests default agent checkpoint with mocked search (zero unnecessary web calls).
Prints actual checkpoint, device, step, val loss, route, and full replies.
"""

import os
import sys
from unittest.mock import patch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agent import MiniGPTAgent

MOCK_SEARCH_RETURN = ("Mocked Search Context: Latest factual details found.", ["https://example.com/mock-source"])

# Suite 1: Prompts from user app chat log (where [02] was 'Aaj mera din...' and [08] was 'Main thoda break...')
SUITE_APP_SESSION = [
    "hi",
    "Aaj mera din thoda mushkil tha.",
    "my name is sandeep",
    "i am tellung you my name",
    "what is the full form of IPS",
    "indian police service",
    "can you fetch google",
    "Main thoda break lena chahta hoon.",
    "indore kaha hai",
    "tumhara naam kya hai",
]

# Suite 2: Approved 10 Hinglish benchmark prompts from eval_prompts.txt
SUITE_APPROVED_HINGLISH = [
    "Aaj tumhara mood kaisa chal raha hai?",
    "Main aaj thoda sa tired feel kar raha hoon.",
    "Mujhe apni ek chhoti achievement share karni hai.",
    "Kya tum meri baat bina advice diye sun sakte ho?",
    "Aaj mujhe thoda break chahiye.",
    "Chalo kisi light topic par baat karte hain.",
    "Main ek nayi hobby try karne ka soch raha hoon.",
    "Apni baat shuru karne mein mujhe thoda awkward lagta hai.",
    "Koi chhoti si funny line suna do.",
    "Aaj maine kaafi kuch manage kar liya.",
]


def run_stage2():
    print("=" * 75)
    print("STAGE 2: MODEL REPLY & CONVERSATIONAL EVALUATION")
    print("=" * 75)

    agent = MiniGPTAgent()
    print(f"Default Checkpoint : {agent.checkpoint_path}")
    print(f"Model Step         : {agent.step}")
    print(f"Validation Loss    : {agent.val_loss}")
    print(f"Execution Device   : {agent.device}")
    print("=" * 75)

    with patch("agent.execute_search_and_scrape", return_value=MOCK_SEARCH_RETURN):
        print("\n--- [SUITE 1: 10 Prompts from User's Observed App Session] ---")
        for idx, prompt in enumerate(SUITE_APP_SESSION, start=1):
            res = agent.respond(prompt)
            print(f"[{idx:02d}] USER : {prompt}")
            print(f"     ROUTE: {res['route']}")
            if res.get("tool_call"):
                print(f"     TOOL : {res['tool_call']}")
            print(f"     REPLY: {res['response']}")
            print("-" * 75)

        print("\n--- [SUITE 2: 10 Approved Hinglish Evaluation Prompts] ---")
        for idx, prompt in enumerate(SUITE_APPROVED_HINGLISH, start=1):
            res = agent.respond(prompt)
            print(f"[{idx:02d}] USER : {prompt}")
            print(f"     ROUTE: {res['route']}")
            if res.get("tool_call"):
                print(f"     TOOL : {res['tool_call']}")
            print(f"     REPLY: {res['response']}")
            print("-" * 75)

    print("STAGE 2 COMPLETE.")
    print("=" * 75)


if __name__ == "__main__":
    run_stage2()
