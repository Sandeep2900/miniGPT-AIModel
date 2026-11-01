"""
audit_evaluator.py
Stage 1: Audits the keyword-based evaluator on the 32 baseline test results.
Compares the current substring matching rule with a strict word-boundary (\b) rule.
"""

import os
import sys
import json
import re

# Set UTF-8 encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from eval_intents import classify_response_intent

BASELINE_JSON = os.path.join("logs", "baseline_eval.json")

# Keywords exactly as defined in eval_intents.py
KEYWORD_REGISTRY = {
    "greeting": ["hello", "namaste", "badhiya", "help", "madad", "assist", "swagat", "good morning", "subah", "hey", "hi", "how"],
    "identity": ["mini-gpt", "assistant", "ai", "scratch", "train", "transformer", "model"],
    "user_name_unknown": ["nahi pata", "nahi maloom", "do not know", "personal data", "introduce", "naam nahi"],
    "location_question": [], # Handled by tool call regex pattern
    "time_question": ["clock", "samay", "time", "ghadi", "taskbar", "live"],
    "date_question": ["date", "tarikh", "taarikh", "din", "clock", "calendar"],
    "thanks": ["welcome", "swagat", "pleasure", "khushi", "madad", "anytime"],
    "goodbye": ["alvida", "goodbye", "bye", "care", "phir milenge", "ahead", "soon", "talk", "night", "later", "time"]
}


def classify_strict_word_boundary(reply_text: str, expected_intent: str) -> tuple[bool, str]:
    """
    Evaluates reply using strict word-boundary matching (\\bkeyword\\b).
    Returns (is_match, matched_keyword).
    """
    clean = reply_text.strip().lower()
    if not clean:
        return False, ""

    if expected_intent == "greeting":
        if "[tool:" in clean:
            return False, ""
        for kw in KEYWORD_REGISTRY["greeting"]:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, clean):
                return True, kw
        return False, ""

    elif expected_intent == "identity":
        if "aapka naam nahi" in clean:
            return False, ""
        for kw in KEYWORD_REGISTRY["identity"]:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, clean):
                return True, kw
        return False, ""

    elif expected_intent == "user_name_unknown":
        is_confused = "mera naam mini-gpt" in clean or "my name is mini-gpt" in clean
        if is_confused:
            return False, ""
        for kw in KEYWORD_REGISTRY["user_name_unknown"]:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, clean):
                return True, kw
        return False, ""

    elif expected_intent == "location_question":
        pattern = r'^\[tool: search\("([^"\r\n]+)"\)\]$'
        is_match = bool(re.match(pattern, clean, re.IGNORECASE))
        return is_match, "[tool: search(...)]" if is_match else ""

    elif expected_intent == "time_question":
        for kw in KEYWORD_REGISTRY["time_question"]:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, clean):
                return True, kw
        return False, ""

    elif expected_intent == "date_question":
        for kw in KEYWORD_REGISTRY["date_question"]:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, clean):
                return True, kw
        return False, ""

    elif expected_intent == "thanks":
        for kw in KEYWORD_REGISTRY["thanks"]:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, clean):
                return True, kw
        return False, ""

    elif expected_intent == "goodbye":
        for kw in KEYWORD_REGISTRY["goodbye"]:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, clean):
                return True, kw
        return False, ""

    return False, ""


def find_substring_culprits(reply_text: str, expected_intent: str) -> list[str]:
    """Finds which keywords matched as substring but failed word-boundary."""
    clean = reply_text.strip().lower()
    culprits = []
    keywords = KEYWORD_REGISTRY.get(expected_intent, [])
    for kw in keywords:
        if kw in clean:
            boundary_pattern = rf"\b{re.escape(kw)}\b"
            if not re.search(boundary_pattern, clean):
                culprits.append(kw)
    return culprits


def run_audit():
    print("=" * 100)
    print("STAGE 1: AUDIT OF EVALUATOR ON BASELINE RESULTS")
    print(f"Loading data from: {BASELINE_JSON}")
    print("=" * 100)

    if not os.path.exists(BASELINE_JSON):
        print(f"ERROR: {BASELINE_JSON} not found!")
        sys.exit(1)

    with open(BASELINE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_results = data.get("test_results", [])
    print(f"Total entries loaded: {len(test_results)}\n")

    # a) Print each row with current rule
    print("--- PART A: ALL 32 ROWS UNDER CURRENT EVALUATOR RULE ---")
    print(f"{'#':<3} | {'Intent':<20} | {'Current Result':<14} | {'Reply Snippet'}")
    print("-" * 100)
    
    current_results = []
    for idx, item in enumerate(test_results, 1):
        intent = item["intent"]
        reply = item["reply"]
        curr_res = classify_response_intent(reply, intent)
        current_results.append(curr_res)
        # Truncate reply snippet for clean tabular view
        snippet = reply.replace("\n", " ")
        if len(snippet) > 60:
            snippet = snippet[:57] + "..."
        print(f"{idx:<3} | {intent:<20} | {str(curr_res):<14} | {snippet}")

    # b) Check strict word boundary for current_result == True
    print("\n" + "=" * 100)
    print("--- PART B: ROWS WHERE CURRENT RESULT IS TRUE vs STRICT WORD-BOUNDARY RULE ---")
    print("-" * 100)

    changed_rows = []
    for idx, item in enumerate(test_results, 1):
        intent = item["intent"]
        phrase = item["phrase"]
        reply = item["reply"]
        curr_res = current_results[idx - 1]

        if curr_res:
            strict_res, matched_kw = classify_strict_word_boundary(reply, intent)
            if strict_res != curr_res:
                culprits = find_substring_culprits(reply, intent)
                changed_rows.append({
                    "row": idx,
                    "intent": intent,
                    "phrase": phrase,
                    "reply": reply,
                    "curr_res": curr_res,
                    "strict_res": strict_res,
                    "culprits": culprits
                })

    if not changed_rows:
        print("No rows changed between current rule and strict word-boundary rule.")
    else:
        print(f"Found {len(changed_rows)} row(s) where result changed from True to False:\n")
        for r in changed_rows:
            print(f"Row #{r['row']} | Intent: '{r['intent']}' | Test Phrase: '{r['phrase']}'")
            print(f"  Reply         : \"{r['reply']}\"")
            print(f"  Current Rule  : {r['curr_res']} (Substring match)")
            print(f"  Strict Rule   : {r['strict_res']} (Word-boundary match)")
            print(f"  Culprit Substr: {r['culprits']} (e.g. matched inside other words without boundary)")
            print("-" * 100)

    # c) Counts and short keywords
    print("\n--- PART C: SUMMARY METRICS & SHORT KEYWORDS (< 4 CHARACTERS) ---")
    print(f"Count of rows where Old Rule = True and Strict Rule = False: {len(changed_rows)}")
    
    old_correct_count = sum(current_results)
    strict_correct_count = old_correct_count - len(changed_rows)
    print(f"Old Rule Total Correct   : {old_correct_count} / {len(test_results)} ({(old_correct_count/len(test_results))*100:.2f}%)")
    print(f"Strict Rule Total Correct: {strict_correct_count} / {len(test_results)} ({(strict_correct_count/len(test_results))*100:.2f}%)")
    
    print("\nIntents containing keywords shorter than 4 characters:")
    for intent, kws in KEYWORD_REGISTRY.items():
        short_kws = [kw for kw in kws if len(kw) < 4]
        if short_kws:
            print(f"  - Intent '{intent}': {short_kws} (lengths: {[len(k) for k in short_kws]})")

    print("=" * 100)


if __name__ == "__main__":
    run_audit()
