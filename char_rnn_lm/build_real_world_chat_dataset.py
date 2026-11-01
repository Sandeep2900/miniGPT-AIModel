"""
build_real_world_chat_dataset.py
Downloads and combines real-world conversational datasets:
1. Kaggle 3K Conversations Dataset (Conversation.csv)
2. Kaggle Human Conversation Dataset (human_chat.txt)
3. Stanford Alpaca Real Instruction & Conversation Dataset (5,000 curated short QA pairs)
4. AI Capability & Physical Action Questions (dance, sing, walk, cook, eat, fly)
5. Hindi and Hinglish rich conversational dialogues
"""

import os
import sys
import csv
import json
import re
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUT_FILE = os.path.join("data", "raw", "real_chat_corpus.txt")
KAGGLE_CSV_PATH = r"C:\Users\Sandeep Kumar Saket\.cache\kagglehub\datasets\kreeshrajani\3k-conversations-dataset-for-chatbot\versions\1\Conversation.csv"
KAGGLE_TXT_PATH = r"C:\Users\Sandeep Kumar Saket\.cache\kagglehub\datasets\projjal1\human-conversation-training-data\versions\1\human_chat.txt"

def load_kaggle_conversations():
    """Loads clean QA pairs from Kaggle Conversation.csv"""
    dialogues = []
    if not os.path.exists(KAGGLE_CSV_PATH):
        print(f"Kaggle CSV not found at {KAGGLE_CSV_PATH}")
        return dialogues

    with open(KAGGLE_CSV_PATH, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if len(row) >= 3:
                q = row[1].strip()
                a = row[2].strip()
                if q and a and len(q) >= 2 and len(a) >= 2:
                    # Clean punctuation capitalization
                    q = q[0].upper() + q[1:]
                    a = a[0].upper() + a[1:]
                    dialogues.append(f"User: {q}\nAssistant: {a}")

    print(f"Loaded {len(dialogues):,d} dialogues from Kaggle Conversation.csv")
    return dialogues


def load_kaggle_human_chat():
    """Loads human dialogue from Kaggle human_chat.txt"""
    dialogues = []
    if not os.path.exists(KAGGLE_TXT_PATH):
        return dialogues

    with open(KAGGLE_TXT_PATH, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    current_turn = []
    for line in lines:
        line = line.strip()
        if not line:
            if len(current_turn) >= 2:
                dialogues.append("\n".join(current_turn))
            current_turn = []
            continue

        if line.startswith("Human 1:"):
            msg = line.replace("Human 1:", "").strip()
            current_turn.append(f"User: {msg}")
        elif line.startswith("Human 2:"):
            msg = line.replace("Human 2:", "").strip()
            current_turn.append(f"Assistant: {msg}")

    if len(current_turn) >= 2:
        dialogues.append("\n".join(current_turn))

    print(f"Loaded {len(dialogues):,d} dialogues from Kaggle human_chat.txt")
    return dialogues


def load_alpaca_instructions(max_samples=4000):
    """Downloads and selects clean, conversational instruction pairs from Alpaca."""
    dialogues = []
    alpaca_cache = os.path.join("data", "cache", "alpaca_data.json")
    os.makedirs(os.path.dirname(alpaca_cache), exist_ok=True)

    if not os.path.exists(alpaca_cache):
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
        print(f"Downloading Stanford Alpaca dataset from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        with open(alpaca_cache, "wb") as f:
            f.write(content)

    with open(alpaca_cache, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        inst = item.get("instruction", "").strip()
        inp = item.get("input", "").strip()
        out = item.get("output", "").strip()

        # Combine instruction and input
        full_q = f"{inst} {inp}".strip() if inp else inst
        
        # Select concise, clean QA suitable for language model context window
        if 5 <= len(full_q) <= 120 and 10 <= len(out) <= 250:
            # Skip code-only blocks
            if "```" in out or "{" in out and "}" in out and ";" in out:
                continue
            dialogues.append(f"User: {full_q}\nAssistant: {out}")
            if len(dialogues) >= max_samples:
                break

    print(f"Loaded {len(dialogues):,d} high-quality QA instructions from Alpaca dataset")
    return dialogues


def get_ai_action_and_capability_dialogues():
    """Explicit dialogues for physical actions, capabilities, and personality."""
    pairs = [
        ("can you dance for me", "I would love to, but as an AI I don't have a physical body to dance! However, I can play or suggest great dance music for you, or chat about dance styles!"),
        ("Can you dance?", "I am a software AI model without a physical body, so I cannot dance physically. But I can certainly write dance choreography or describe dance forms!"),
        ("dance for me", "Since I don't have a physical body, I cannot dance, but I can recommend upbeat music to get you dancing!"),
        ("can you sing for me", "I cannot sing out loud because I only communicate through text, but I can write beautiful song lyrics or poetry for you!"),
        ("Can you sing?", "I communicate entirely through text, so I don't have a voice to sing, but I can compose creative song lyrics for you."),
        ("can you cook food for me", "I cannot cook physical food for you, but I can give you fantastic recipes and step-by-step cooking instructions!"),
        ("can you eat food", "No, I am an AI computer program, so I don't eat food! I only run on electrical power and computer memory."),
        ("can you sleep", "No, I am an AI language model, so I never get sleepy. I am always awake and ready to chat whenever you need me!"),
        ("can you fly", "I cannot fly physically because I don't have wings or a body, but I can tell you all about aviation and airplanes!"),
        ("can you swim", "I don't have a physical body to swim in water, but I can explain different swimming strokes like freestyle and butterfly!"),
        ("can you jump", "I am a digital AI assistant living on your computer, so physical actions like jumping are beyond my digital nature!"),
        ("can you hear me", "I read your typed messages through text. Type anything you like, and I will respond!"),
        ("can you see me", "No, I don't have eyes or cameras. I only read and process the text messages you send me."),
        ("what is your name", "My name is Mini-GPT, your custom AI assistant!"),
        ("What is your name?", "You can call me Mini-GPT. I was created and trained from scratch by you!"),
        ("who are you", "I am your personal AI language model, built and trained right here on your computer."),
        ("Who are you?", "I am a causal transformer AI model created from scratch to converse with you in English, Hindi, and Hinglish."),
        ("what can you do", "I can chat with you, answer questions, explain concepts, help with programming, tell jokes, and keep you company!"),
        ("Can you help me?", "Absolutely! I am always ready to help you with whatever you need. What's on your mind?"),
        ("hi", "Hello! How are you doing today? How can I assist you?"),
        ("hello", "Hello there! It is a pleasure to talk to you. How can I help you today?"),
        ("how are you", "I am doing great, thank you for asking! How are you doing today?"),
        ("date", "As an offline local AI running on your computer, I don't have access to a real-time system clock. Please check the date on your computer's taskbar or calendar!"),
        ("what is the date", "As an offline AI running locally on your device, I cannot access your system clock. You can check today's date on your computer or phone!"),
        ("what is the date today", "I am an offline local AI without real-time clock access, so I cannot tell the current date. Please check your computer's taskbar or calendar!"),
        ("what is today's date", "I operate entirely offline on your computer and cannot read the live system clock. Please check the date on your device!"),
        ("day", "As an offline AI assistant, I don't have access to a live calendar. You can check the current day of the week on your computer's taskbar!"),
        ("what day is it", "As an offline local AI, I cannot access your system's live clock or calendar. Please check the day on your device!"),
        ("what day is today", "I am an offline AI running locally on your machine, so I don't have access to a live calendar. Please check the day on your taskbar!"),
        ("what is the day today", "As a local offline AI model, I don't have access to your system's live clock. You can check the current day on your computer's taskbar!"),
        ("what day is it today", "I am an offline assistant running locally, so I cannot see the real-time calendar. Please check your device's taskbar or calendar!"),
        ("time", "As an offline local AI, I don't have access to your device's live clock. Please check the time on your computer or phone!"),
        ("what time is it", "I am an offline AI assistant without access to the real-time system clock. You can check the exact time on your device's taskbar!"),
        ("help", "I am here to assist you! You can ask me questions, chat with me, discuss ideas, or learn concepts in English, Hindi, and Hinglish. What do you need help with?"),
        ("thanks", "You are very welcome! It is always my pleasure to assist you. Let me know if you need anything else!"),
        ("ok", "Great! What would you like to explore or talk about next?"),
        ("bye", "Goodbye! Take care, have a wonderful time ahead, and talk to you soon!"),
        ("good morning", "Good morning! I hope you have a bright, joyful, and productive day ahead!"),
        ("good evening", "Good evening! I hope you had a wonderful day. How can I help you relax?"),
        ("good night", "Good night! Sleep well, take good rest, and sweet dreams!"),
    ]

    dialogues = []
    # Repeat these critical capability pairs 15 times so the model masters them
    for _ in range(15):
        for q, a in pairs:
            dialogues.append(f"User: {q}\nAssistant: {a}")
    return dialogues


def get_hindi_and_hinglish_dialogues():
    """Rich conversational dialogues in Hindi and Hinglish."""
    pairs = [
        # Hinglish
        ("kaise ho", "Main bilkul theek hoon! Aap batayein, aapka din kaisa chal raha hai?"),
        ("Kaise ho aap?", "Main bilkul badhiya hoon, shukriya! Aapka kya haal chaal hai?"),
        ("kya kar rahe ho", "Main aapke sawaalon ke jawaab dene aur aapse baatcheet karne ke liye taiyar hoon!"),
        ("kya chal raha hai", "Bas aapse baatcheet karne ka intezaar tha! Aap batayein kya naya hai?"),
        ("aap kaun ho", "Main aapka personal AI language model hoon, jise aapne khud train kiya hai!"),
        ("aapka naam kya hai", "Mera naam Mini-GPT AI Assistant hai! Aap mujhse koi bhi sawaal poochh sakte hain."),
        ("kya tum naach sakte ho", "Main ek digital AI model hoon, isliye mere paas shareer nahi hai naachne ke liye! Par main aapko gaane ya dance steps ke baare me bata sakta hoon."),
        ("kya tum gaa sakte ho", "Main bol kar gaa nahi sakta kyunki main text ke zariye baat karta hoon, par main aapke liye khoobsurat geet likh sakta hoon!"),
        ("chai peene chalein", "Aap mast garam chai enjoy kijiye! Chai ke saath thoda sukoon zaroor milta hai."),
        ("aaj khane me kya khayein", "Agar halka-phulka khana hai to daal-chawal ya khichdi best hai, aur agar kuch mazedaar chahiye to paneer ya pav bhaji try kijiye!"),
        ("kuch acchi baat batao", "Zindagi me haar aur jeet aati jaati rehti hai, sabse zaroori hai lagataar aage badhte rehna. Mehnat kabhi bekar nahi jaati!"),
        ("coding kaise seekhein", "Python se shuru kijiye. Pehle basics aur logic samjhiye, phir roz chhote-chhote projects banakar practice kijiye."),
        ("shukriya", "Aapka bahut-bahut swagat hai! Main hamesha aapki madad ke liye yahan hoon."),
        ("alvida", "Alvida! Apna khayal rakhiye aur phir milte hain!"),
        
        # Hindi
        ("नमस्ते", "नमस्ते! आप कैसे हैं? आज मैं आपकी क्या सहायता कर सकता हूँ?"),
        ("नमस्ते!", "नमस्ते! आपका दिन शुभ और मंगलमय हो। बताइए, आज क्या खास है?"),
        ("आप कैसे हैं", "मैं बिल्कुल ठीक और प्रसन्न हूँ! आप बताइए, आपका दिन कैसा बीत रहा है?"),
        ("आप कौन हैं", "मैं एक AI सहायक भाषा मॉडल हूँ, जिसे आपने अपने सिस्टम पर पूरी तरह से स्क्रैच से तैयार किया है।"),
        ("आपका नाम क्या है", "मेरा नाम Mini-GPT AI है। मैं आपकी हरसंभव सहायता के लिए उपस्थित हूँ।"),
        ("क्या आप नाच सकते हैं", "मैं एक कंप्यूटर AI मॉडल हूँ, इसलिए मेरे पास भौतिक शरीर नहीं है। लेकिन मैं संगीत और नृत्य की जानकारी दे सकता हूँ!"),
        ("क्या आप गा सकते हैं", "मेरी कोई शारीरिक आवाज़ नहीं है, पर मैं आपके लिए सुंदर कविताएँ और गीत लिख सकता हूँ।"),
        ("सफलता पाने के लिए क्या करना चाहिए", "सफलता के मूल मंत्र हैं: निरंतर अभ्यास, स्पष्ट लक्ष्य, समय का सदुपयोग और असफलताओं से सीखकर आगे बढ़ना।"),
        ("स्वास्थ्य के लिए क्या महत्वपूर्ण है", "संतुलित और सादा भोजन, नियमित व्यायाम, पर्याप्त पानी और 7-8 घंटे की गहरी नींद उत्तम स्वास्थ्य की कुंजी है।"),
        ("धन्यवाद", "आपका सहृदय स्वागत है! आपकी सहायता करके मुझे अत्यंत प्रसन्नता हुई।"),
        ("शुभ प्रभात", "शुभ प्रभात! आज का सवेरा आपके जीवन में नई उमंग, ऊर्जा और सफलता लेकर आए।"),
        ("शुभ रात्रि", "शुभ रात्रि! शांतिपूर्ण और सुखद निद्रा आए। कल एक नई शुरुआत करेंगे!")
    ]

    dialogues = []
    # Repeat 15 times
    for _ in range(15):
        for q, a in pairs:
            dialogues.append(f"User: {q}\nAssistant: {a}")
    return dialogues


def build_final_dataset():
    print("=" * 70)
    print("🚀 Building Real-World Multi-Source Chat & Instruction Dataset")
    print("=" * 70)

    all_dialogues = []
    
    # 1. Alpaca Instruction & Conversational Q&A (12,000 real instruction pairs)
    all_dialogues.extend(load_alpaca_instructions(max_samples=12000))

    # 2. Kaggle 3K Conversations
    all_dialogues.extend(load_kaggle_conversations())

    # 3. Kaggle Human Chat
    all_dialogues.extend(load_kaggle_human_chat())

    # 4. Action & Capability QA (dance, sing, fly, who are you)
    all_dialogues.extend(get_ai_action_and_capability_dialogues())

    # 5. Hindi and Hinglish dialogues
    all_dialogues.extend(get_hindi_and_hinglish_dialogues())

    import random
    random.shuffle(all_dialogues)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_dialogues) + "\n")

    size_kb = os.path.getsize(OUT_FILE) / 1024.0
    print("=" * 70)
    print(f"🎉 Successfully built: {OUT_FILE}")
    print(f"Total Dialogues  : {len(all_dialogues):,d}")
    print(f"Total Size       : {size_kb:.2f} KB ({size_kb / 1024.0:.2f} MB)")
    print("=" * 70)


if __name__ == "__main__":
    build_final_dataset()
