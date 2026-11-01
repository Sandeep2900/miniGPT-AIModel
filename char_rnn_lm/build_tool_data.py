"""
build_tool_data.py
Generates a balanced, aligned training dataset for tool calling:
- Mixes 400 high-quality conversational turns from data/train.txt (to prevent catastrophic forgetting)
- Mixes 150 diverse [TOOL: search("<query>")] tool calls (for real-time facts/news/prices)
- Injects 'Current time: YYYY-MM-DD HH:MM.' in every sample to teach the model how to condition on system clock
Saves to data/tool_train.txt and data/tool_val.txt.
"""

import os
import sys
import random

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TOOL_CALL_EXAMPLES = [
    # Hindi & Hinglish Search Queries
    ("Aaj ka mausam kaisa hai Delhi me?", '[TOOL: search("Delhi weather today")]'),
    ("Latest cricket match ka score kya hai?", '[TOOL: search("latest cricket match score")]'),
    ("Aaj ki taaza khabar batao.", '[TOOL: search("aaj ki taaza khabar hindi news")]'),
    ("Gold price kitna chal raha hai aaj?", '[TOOL: search("gold price today in India")]'),
    ("Petrol ka current price kya hai?", '[TOOL: search("petrol price today")]'),
    ("Nifty 50 ka aaj ka status kya hai?", '[TOOL: search("Nifty 50 stock status today")]'),
    ("Bitcoin ka rate abhi kya chal raha hai?", '[TOOL: search("Bitcoin price USD today")]'),
    ("Kal ka match kaun jeeta?", '[TOOL: search("yesterday match winner result")]'),
    ("Mumbai me aaj baarish hogi kya?", '[TOOL: search("Mumbai weather forecast today")]'),
    ("Silver rate aaj ka kitna hai?", '[TOOL: search("silver price today in India")]'),
    ("India vs Pakistan match ka score kya hai?", '[TOOL: search("India vs Pakistan live cricket score")]'),
    ("Dollar ka rate rupees me kitna hai abhi?", '[TOOL: search("USD to INR exchange rate today")]'),
    ("Aaj share market kyu gira?", '[TOOL: search("share market crash reason today")]'),
    ("New iPhone ka price kya hai?", '[TOOL: search("latest iPhone price in India")]'),
    ("IPL points table dikhao.", '[TOOL: search("IPL points table current")]'),
    ("ISRO ka agla mission konsa hai?", '[TOOL: search("ISRO upcoming space mission")]'),
    ("Aaj bank open hai ya holiday?", '[TOOL: search("bank holiday today India")]'),
    ("Aaj ka gold rate Mumbai me kya hai?", '[TOOL: search("Mumbai gold rate today")]'),
    ("Aaj ki top breaking news kya hai?", '[TOOL: search("top breaking news India today")]'),
    ("Chandigarh ka temperature kitna hai?", '[TOOL: search("Chandigarh temperature today")]'),
    ("LPG cylinder ka naya rate kya hai?", '[TOOL: search("LPG cylinder price today")]'),
    ("Bangalore me traffic status kya hai?", '[TOOL: search("Bangalore traffic update live")]'),
    ("Upcoming mobile launches in 2026?", '[TOOL: search("upcoming smartphone launches 2026")]'),

    # English Search Queries
    ("What is the latest news in artificial intelligence?", '[TOOL: search("latest artificial intelligence news")]'),
    ("What is the current stock price of Apple?", '[TOOL: search("Apple AAPL stock price today")]'),
    ("Who won the English Premier League match yesterday?", '[TOOL: search("English Premier League results yesterday")]'),
    ("What is the weather in New York right now?", '[TOOL: search("New York weather forecast")]'),
    ("Tell me the latest updates on Python 3.15.", '[TOOL: search("Python 3.15 release updates news")]'),
    ("What are the current mortgage interest rates?", '[TOOL: search("current mortgage interest rates")]'),
    ("What is the price of Tesla stock right now?", '[TOOL: search("Tesla TSLA stock price today")]'),
    ("Who is the current President of France?", '[TOOL: search("current President of France")]'),
    ("What were yesterday's NBA game scores?", '[TOOL: search("NBA scores yesterday")]'),
    ("What is the inflation rate in the US today?", '[TOOL: search("US inflation rate current")]'),
    ("Who won the most recent F1 Grand Prix?", '[TOOL: search("recent F1 Grand Prix race winner")]'),
    ("What is the exchange rate of Euro to USD today?", '[TOOL: search("EUR to USD exchange rate today")]'),
    ("What happened in the stock market today?", '[TOOL: search("stock market summary today")]'),
    ("What is the latest version of PyTorch?", '[TOOL: search("latest PyTorch release version")]'),
    ("Current price of Ethereum today?", '[TOOL: search("Ethereum price USD today")]'),
    ("Who won the men's tennis championship?", '[TOOL: search("latest men tennis championship winner")]'),
    ("What is the current crude oil price per barrel?", '[TOOL: search("crude oil Brent price today")]'),
    ("Tell me today's top world headlines.", '[TOOL: search("top world headlines news today")]'),
    ("What is the air quality index in New Delhi right now?", '[TOOL: search("New Delhi AQI air quality index live")]'),
    ("Is there a new update for Windows 11?", '[TOOL: search("Windows 11 latest update news")]'),
]


def load_conversational_replay_samples(max_samples: int = 400):
    """Loads clean conversation blocks from data/train.txt."""
    train_path = os.path.join("data", "train.txt")
    if not os.path.exists(train_path):
        return []

    with open(train_path, "r", encoding="utf-8") as f:
        text = f.read()

    blocks = text.split("\n\n")
    clean_pairs = []
    for b in blocks:
        lines = [line.strip() for line in b.strip().split("\n") if line.strip()]
        if len(lines) == 2 and lines[0].startswith("User:") and lines[1].startswith("Assistant:"):
            # Exclude old offline time answers to avoid confusing the new agent
            if "offline AI" in lines[1] or "calendar" in lines[1]:
                continue
            u = lines[0][len("User:"):].strip()
            a = lines[1][len("Assistant:"):].strip()
            clean_pairs.append((u, a))

    random.seed(42)
    random.shuffle(clean_pairs)
    return clean_pairs[:max_samples]


def main():
    print("=" * 68)
    print("🛠️ Generating Replay-Balanced Tool-Calling & Conversation Dataset")
    print("=" * 68)

    # 1. Load conversation replay samples
    conv_samples = load_conversational_replay_samples(max_samples=400)
    print(f"Loaded {len(conv_samples)} conversational replay samples from data/train.txt")

    # 2. Expand tool calling samples
    tool_samples = []
    prefixes = ["", "Please check: ", "Batao zara: ", "Search online: ", "Tell me: "]
    for q, a in TOOL_CALL_EXAMPLES:
        tool_samples.append((q, a))
        for p in random.sample(prefixes, 2):
            if p:
                variant_q = p + q[0].lower() + q[1:]
                tool_samples.append((variant_q, a))

    print(f"Generated {len(tool_samples)} tool call samples")

    # 3. Combine with time header injection (aligning with agent.py prompt format)
    fixed_time = "2026-11-01 20:45"
    time_header = f"Current time: {fixed_time}."

    all_dataset = []
    for q, a in conv_samples:
        all_dataset.append((f"{time_header}\nUser: {q}", a))

    for q, a in tool_samples:
        all_dataset.append((f"{time_header}\nUser: {q}", a))

    random.seed(1337)
    random.shuffle(all_dataset)

    split_idx = int(len(all_dataset) * 0.85)
    train_data = all_dataset[:split_idx]
    val_data = all_dataset[split_idx:]

    os.makedirs("data", exist_ok=True)
    train_path = os.path.join("data", "tool_train.txt")
    val_path = os.path.join("data", "tool_val.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        for u, a in train_data:
            f.write(f"{u}\nAssistant: {a}\n\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for u, a in val_data:
            f.write(f"{u}\nAssistant: {a}\n\n")

    print(f"Total dataset: {len(all_dataset)} samples")
    print(f"  • Train: {len(train_data)} samples saved to {train_path}")
    print(f"  • Val  : {len(val_data)} samples saved to {val_path}")
    print("=" * 68)


if __name__ == "__main__":
    main()
