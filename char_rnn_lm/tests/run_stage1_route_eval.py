"""
tests/run_stage1_route_eval.py
Stage 1 Route Evaluation Script:
Runs all 6 required prompts against MiniGPTAgent with mocked search (zero internet access).
Evaluates Input, Expected Route, Actual Route, and Status (PASS/FAIL).
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

# Mock search & scrape return value to prevent any network calls
MOCK_SEARCH_RETURN = ("Mocked Search Context: Latest details found.", ["https://example.com/mock-source"])

TEST_CASES = [
    {
        "input": "Aaj mera din thoda mushkil tha.",
        "expected_route": "model_direct"
    },
    {
        "input": "Main thoda break lena chahta hoon.",
        "expected_route": "model_direct"
    },
    {
        "input": "Aaj ka weather kya hai?",
        "expected_route": "search_rule"
    },
    {
        "input": "Latest Python release news kya hai?",
        "expected_route": "search_rule"
    },
    {
        "input": "Abhi time kya hua hai?",
        "expected_route": "clock"
    },
    {
        "input": "Hi, kaise ho?",
        "expected_route": "model_direct"
    }
]


def run_evaluation():
    print("=" * 70)
    print("Stage 1: Route Evaluation (Offline with Mocked Search)")
    print("=" * 70)

    agent = MiniGPTAgent()
    print(f"Loaded checkpoint : {agent.checkpoint_path}")
    print(f"Device            : {agent.device}")
    print("-" * 70)

    all_passed = True

    # Patch execute_search_and_scrape so no network request is sent
    with patch("agent.execute_search_and_scrape", return_value=MOCK_SEARCH_RETURN):
        for idx, item in enumerate(TEST_CASES, start=1):
            query = item["input"]
            expected = item["expected_route"]
            
            res = agent.respond(query)
            actual = res["route"]
            passed = (actual == expected)
            if not passed:
                all_passed = False

            status = "PASS" if passed else "FAIL"
            print(f"[{idx:02d}] INPUT   : {query}")
            print(f"     EXPECTED: {expected}")
            print(f"     ACTUAL  : {actual}")
            print(f"     STATUS  : {status}")
            print(f"     REPLY   : {res['response']}")
            print("-" * 70)

    print(f"Overall Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
