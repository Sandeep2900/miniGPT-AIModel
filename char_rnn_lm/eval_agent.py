"""
eval_agent.py
Autonomous benchmark evaluating the Online MiniGPT Agent across 4 distinct categories:
1. Real-time Search Queries (Trigger [TOOL: search(...)] / search_rule)
2. Clock / Time Queries (Trigger clock without search)
3. Conversational / Identity Queries (Answer from model weights)
4. Robustness & Fallback Queries (Handle malformed/ambiguous inputs without crashing)

Computes score, latency, tool call accuracy, and prints a structured evaluation table.
"""

import os
import sys
import time
from typing import List, Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agent import MiniGPTAgent

BENCHMARK_SUITE = [
    # Category 1: Real-Time Fact / Search (Expected Route: search_rule or model_tool)
    {"cat": "1. Search / Real-Time", "query": "Latest Python release news", "expected_route": ["search_rule", "model_tool"]},
    {"cat": "1. Search / Real-Time", "query": "What is the current gold price in India?", "expected_route": ["search_rule", "model_tool"]},
    {"cat": "1. Search / Real-Time", "query": "Aaj ka weather kaisa hai Mumbai me?", "expected_route": ["search_rule", "model_tool"]},
    {"cat": "1. Search / Real-Time", "query": "Current cricket match score", "expected_route": ["search_rule", "model_tool"]},
    {"cat": "1. Search / Real-Time", "query": "Bitcoin price USD today", "expected_route": ["search_rule", "model_tool"]},

    # Category 2: Clock / Datetime (Expected Route: clock)
    {"cat": "2. Clock / Datetime", "query": "Aaj kya date hai?", "expected_route": ["clock"]},
    {"cat": "2. Clock / Datetime", "query": "What time is it right now?", "expected_route": ["clock"]},
    {"cat": "2. Clock / Datetime", "query": "Kitna samay hua hai?", "expected_route": ["clock"]},
    {"cat": "2. Clock / Datetime", "query": "Current time please", "expected_route": ["clock"]},
    {"cat": "2. Clock / Datetime", "query": "Tell me today's date and time", "expected_route": ["clock"]},

    # Category 3: Conversational / Identity (Expected Route: model_direct)
    {"cat": "3. Conversation / Identity", "query": "Hello, who are you?", "expected_route": ["model_direct"]},
    {"cat": "3. Conversation / Identity", "query": "Namaste dost", "expected_route": ["model_direct"]},
    {"cat": "3. Conversation / Identity", "query": "How do you work?", "expected_route": ["model_direct"]},
    {"cat": "3. Conversation / Identity", "query": "What is machine learning?", "expected_route": ["model_direct"]},
    {"cat": "3. Conversation / Identity", "query": "Thank you for your help", "expected_route": ["model_direct"]},

    # Category 4: Robustness & Fallback (Expected: non-crashing valid response)
    {"cat": "4. Robustness & Fallback", "query": "search()", "expected_route": ["fallback_search", "search_rule", "model_direct"]},
    {"cat": "4. Robustness & Fallback", "query": "[TOOL: search(test)]", "expected_route": ["model_tool", "fallback_search", "search_rule", "model_direct"]},
    {"cat": "4. Robustness & Fallback", "query": "Tell me something random and weird 123456", "expected_route": ["fallback_search", "model_direct", "search_rule"]},
    {"cat": "4. Robustness & Fallback", "query": "Can you explain recursion in Python?", "expected_route": ["model_direct"]},
    {"cat": "4. Robustness & Fallback", "query": "Bye, alvida!", "expected_route": ["model_direct"]},
]


def run_evaluation(checkpoint_path: str = "checkpoints/best_gpt.pt"):
    print("=" * 80)
    print("🧪 MiniGPT Online Agent Benchmark Evaluation Suite (20 Test Cases)")
    print("=" * 80)

    # Use tool_gpt.pt if available, else best_gpt.pt
    if os.path.exists("checkpoints/tool_gpt.pt"):
        checkpoint_path = "checkpoints/tool_gpt.pt"
        print(f"Checkpoint : {checkpoint_path} (Fine-tuned tool model)")
    else:
        print(f"Checkpoint : {checkpoint_path} (Base model)")

    agent = MiniGPTAgent(checkpoint_path)
    
    passed = 0
    total = len(BENCHMARK_SUITE)
    results = []

    print(f"\n{'#':<3} | {'Category':<24} | {'Query':<28} | {'Route':<14} | {'Status'}")
    print("-" * 80)

    for i, test in enumerate(BENCHMARK_SUITE, 1):
        q = test["query"]
        expected = test["expected_route"]
        cat = test["cat"]

        start = time.time()
        res = agent.respond(q)
        duration = time.time() - start

        route = res["route"]
        resp_text = res["response"].strip()

        # Evaluation criteria:
        # 1. Non-empty response
        # 2. Route matches expected category routing
        # 3. No uncaught exception
        is_pass = (route in expected) and (len(resp_text) > 0)
        if is_pass:
            passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        results.append({
            "num": i,
            "category": cat,
            "query": q,
            "route": route,
            "status": status,
            "duration": duration,
            "response": resp_text[:60]
        })

        q_disp = q if len(q) <= 28 else q[:25] + "..."
        print(f"{i:<3} | {cat:<24} | {q_disp:<28} | {route:<14} | {status}")

    print("=" * 80)
    pass_rate = (passed / total) * 100
    print(f"Benchmark Results: {passed}/{total} PASSED ({pass_rate:.1f}%)")
    print("=" * 80)
    return passed, total, results


if __name__ == "__main__":
    run_evaluation()
