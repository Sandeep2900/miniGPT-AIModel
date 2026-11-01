"""
stress_test_gpt.py
Autonomous comprehensive test suite covering 50+ diverse conversational queries
across 10 distinct categories in English, Hinglish, and Hindi.
Evaluates semantic accuracy, persona consistency, repetition, and relevance.
"""

import sys
import time
from chat_gpt import load_model_and_tokenizer, chat_turn

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TEST_SUITE = [
    # Category 1: Greetings & Casual
    ("Category 1: Greetings", [
        ("hi", ["hello", "how are you", "assist", "help"]),
        ("hello", ["hello", "pleasure", "help", "discuss"]),
        ("hey", ["hey", "going", "help"]),
        ("good morning", ["good morning", "day", "bright", "energetic"]),
        ("how are you", ["great", "doing", "wonderful", "well"]),
        ("what's up", ["ready", "assist", "eager", "you"]),
    ]),

    # Category 2: Identity & Creator
    ("Category 2: Identity & Creator", [
        ("who are you", ["ai", "assistant", "model", "scratch", "computer"]),
        ("what is your name", ["mini-gpt", "assistant", "name"]),
        ("who made you", ["you", "built", "trained", "scratch", "pytorch"]),
        ("who created you", ["you", "created", "trained", "scratch", "neural"]),
        ("are you an ai", ["yes", "ai", "artificial intelligence", "transformer"]),
        ("what can you do", ["chat", "answer", "english", "hindi", "hinglish", "ideas"]),
    ]),

    # Category 3: Offline Boundaries (Temporal, Date, Time, Day)
    ("Category 3: Offline Temporal Queries", [
        ("date", ["offline", "clock", "taskbar", "calendar", "device"]),
        ("what is the date", ["offline", "system", "clock", "taskbar", "calendar"]),
        ("what is the date today", ["offline", "clock", "taskbar", "calendar"]),
        ("day", ["offline", "calendar", "taskbar", "day"]),
        ("what is the day today", ["offline", "clock", "taskbar", "calendar"]),
        ("what day is today", ["offline", "calendar", "taskbar"]),
        ("time", ["offline", "clock", "phone", "taskbar", "computer"]),
        ("what time is it", ["offline", "clock", "taskbar", "exact"]),
    ]),

    # Category 4: Physical Limitations & Embodiment Realism
    ("Category 4: Physical Capabilities", [
        ("can you dance for me", ["body", "dance", "physical", "music", "software"]),
        ("can you sing for me", ["sing", "voice", "text", "lyrics", "poetry"]),
        ("can you fly", ["fly", "wings", "airplane", "body", "aviation"]),
        ("can you cook food for me", ["cook", "recipes", "food", "physical"]),
        ("can you eat food", ["eat", "ai", "program", "food", "electricity"]),
        ("can you sleep", ["sleep", "ai", "tired", "awake", "chat"]),
        ("can you swim", ["swim", "water", "body", "physical", "strokes"]),
    ]),

    # Category 5: Short 1-word / 2-word queries
    ("Category 5: Short 1-word Queries", [
        ("help", ["assist", "help", "questions", "chat", "ideas"]),
        ("why", ["topic", "concept", "explain", "reasoning", "curious"]),
        ("how", ["process", "guide", "step", "learn", "understand"]),
        ("thanks", ["welcome", "pleasure", "assist", "help"]),
        ("thank you", ["welcome", "pleasure", "always", "chat"]),
        ("ok", ["great", "talk", "explore", "next"]),
        ("bye", ["goodbye", "care", "wonderful", "day"]),
    ]),

    # Category 6: Knowledge, Advice & Chit-Chat
    ("Category 6: Knowledge & Chit-Chat", [
        ("tell me a joke", ["programmer", "dark mode", "bugs", "computer", "doctor"]),
        ("tell me an interesting fact", ["honey", "egyptian", "spoil", "fact", "did you know"]),
        ("what is Python", ["python", "programming", "language", "versatile", "ai"]),
        ("how does machine learning work", ["learning", "patterns", "data", "algorithms"]),
        ("how can I stay productive", ["tasks", "pomodoro", "breaks", "distractions"]),
        ("what should I eat today", ["meal", "rice", "salad", "vegetables", "wholesome"]),
        ("I am stressed today", ["breath", "slow", "one", "solve", "relax"]),
    ]),

    # Category 7: Hinglish Greetings & Conversations
    ("Category 7: Hinglish Conversations", [
        ("kaise ho", ["theek", "badhiya", "din", "kaisa", "aap"]),
        ("kya kar rahe ho", ["madad", "baatcheet", "taiyar", "aap"]),
        ("aap kaun ho", ["custom ai", "assistant", "train", "computer"]),
        ("tumhe kisne banaya", ["aapne", "pytorch", "train", "banaya"]),
        ("aapka naam kya hai", ["mini-gpt", "ai", "assistant", "naam"]),
        ("kya tum naach sakte ho", ["digital", "shareer", "dance", "gaane"]),
        ("kya tum gaa sakte ho", ["bol", "text", "geet", "likh"]),
        ("aaj khane me kya khayein", ["daal", "chawal", "khichdi", "paneer", "tasty"]),
        ("chai peene chalein", ["chai", "garam", "enjoy", "sukoon", "mood"]),
        ("kuch acchi baat batao", ["haar", "seekh", "zindagi", "positive", "mehnat"]),
        ("coding kaise seekhein", ["python", "basics", "practice", "projects"]),
        ("shukriya", ["swagat", "madad", "hamesha"]),
        ("alvida", ["alvida", "khayal", "milte"]),
    ]),

    # Category 8: Hinglish Temporal Queries
    ("Category 8: Hinglish Temporal Queries", [
        ("aaj kaun sa din hai", ["offline", "calendar", "taskbar", "din"]),
        ("aaj ki date kya hai", ["offline", "clock", "taskbar", "date"]),
        ("aaj ki tareekh kya hai", ["offline", "clock", "tareekh", "mobile"]),
        ("kitne baje hain", ["offline", "clock", "time", "phone"]),
    ]),

    # Category 9: Pure Hindi (Devanagari) Conversations
    ("Category 9: Pure Hindi (Devanagari)", [
        ("नमस्ते", ["नमस्ते", "सहायता", "दिन"]),
        ("नमस्ते!", ["नमस्ते", "शुभ", "मंगलमय"]),
        ("आप कैसे हैं", ["ठीक", "प्रसन्न", "अच्छा"]),
        ("आप कौन हैं", ["ai", "सहायक", "मॉडल", "स्क्रैच"]),
        ("आपका नाम क्या है", ["mini-gpt", "नाम", "सहायक"]),
        ("आपको किसने बनाया", ["कंप्यूटर", "pytorch", "प्रशिक्षित", "आपने"]),
        ("आप क्या कर सकते हैं", ["संवाद", "उत्तर", "विचार", "सहायता"]),
        ("क्या आप नाच सकते हैं", ["कंप्यूटर", "भौतिक", "शरीर", "संगीत"]),
        ("सफलता पाने के लिए क्या करना चाहिए", ["अभ्यास", "लक्ष्य", "सफल", "धैर्य"]),
        ("स्वास्थ्य के लिए क्या महत्वपूर्ण है", ["भोजन", "व्यायाम", "नींद", "स्वास्थ्य"]),
        ("कुछ अच्छी बात बताओ", ["मुस्कुराहट", "सकारात्मक", "आनंद", "जीवन"]),
        ("शुभ प्रभात", ["शुभ प्रभात", "सवेरा", "ऊर्जा", "सफलता"]),
        ("शुभ रात्रि", ["शुभ रात्रि", "निद्रा", "शांतिपूर्ण"]),
        ("धन्यवाद", ["स्वागत", "सहायता", "प्रसन्नता"]),
    ]),

    # Category 10: Pure Hindi Temporal Queries
    ("Category 10: Pure Hindi Temporal Queries", [
        ("आज कौन सा दिन है", ["ऑफलाइन", "कैलेंडर", "टास्कबार", "दिन"]),
        ("आज कौन सा दिन है?", ["ऑफलाइन", "कैलेंडर", "टास्कबार", "दिन"]),
        ("आज की तारीख क्या है", ["ऑफलाइन", "घड़ी", "तारीख", "कंप्यूटर"]),
        ("आज की तारीख क्या है?", ["ऑफलाइन", "घड़ी", "तारीख", "स्क्रीन"]),
        ("समय क्या हुआ है", ["ऑफलाइन", "घड़ी", "समय", "टास्कबार"]),
    ])
]


def run_full_stress_test():
    print("=" * 75)
    print("🧪 Autonomous Full Stress Test Suite for Mini-GPT (3.08M Parameters)")
    print("=" * 75)

    model, tokenizer, cfg, step, val_loss, device = load_model_and_tokenizer()
    print(f"Loaded: Step {step} | Val Loss: {val_loss:.4f} | Architecture: {cfg.n_layer}L/{cfg.n_head}H/{cfg.n_embd}D")
    print(f"Device: {device.upper()} | Parameters: {model.get_num_params():,d}\n")

    total_tests = 0
    total_passed = 0
    category_results = []

    for cat_name, test_cases in TEST_SUITE:
        print(f"\n📁 {cat_name} ({len(test_cases)} tests)")
        print("-" * 75)
        cat_passed = 0

        for prompt, expected_keywords in test_cases:
            total_tests += 1
            t0 = time.time()
            reply = chat_turn(model, tokenizer, prompt, device=device, max_new_tokens=80, temperature=0.3)
            elapsed = (time.time() - t0) * 1000

            reply_lower = reply.lower()
            keyword_matches = [kw for kw in expected_keywords if kw in reply_lower]
            has_keyword = len(keyword_matches) > 0
            has_len = len(reply.split()) >= 3
            rep_ratio = (len(reply.split()) - len(set(reply.split()))) / max(1, len(reply.split()))
            not_repeating = rep_ratio < 0.5

            is_pass = has_keyword and has_len and not_repeating

            if is_pass:
                cat_passed += 1
                total_passed += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"

            print(f"  {status} [{elapsed:4.0f}ms] User: '{prompt}'")
            print(f"       -> Assistant: {reply}")
            if not is_pass:
                print(f"          [Diagnostics] Expected one of: {expected_keywords} | Matched: {keyword_matches} | Words: {len(reply.split())} | Rep: {rep_ratio:.2f}")

        pass_rate = (cat_passed / len(test_cases)) * 100
        category_results.append((cat_name, cat_passed, len(test_cases), pass_rate))
        print(f"  Result: {cat_passed}/{len(test_cases)} ({pass_rate:.1f}%)")

    print("\n" + "=" * 75)
    print("📊 Final Stress Test Summary Across All Categories")
    print("=" * 75)
    for cname, cpass, ctot, crate in category_results:
        print(f"  • {cname:35s}: {cpass:2d}/{ctot:2d} ({crate:5.1f}%)")
    
    overall_rate = (total_passed / total_tests) * 100
    print("-" * 75)
    print(f"🌟 OVERALL SCORE: {total_passed}/{total_tests} passed ({overall_rate:.1f}%)")
    print("=" * 75)
    return overall_rate == 100.0


if __name__ == "__main__":
    run_full_stress_test()
