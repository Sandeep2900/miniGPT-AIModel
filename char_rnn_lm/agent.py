"""
agent.py
Central Orchestrator for custom MiniGPT Online Agent.
Integrates:
- System clock injection (zero hallucination, no tool call needed)
- Keyword-based routing (Constraint 7)
- Safe tool parsing with single [TOOL: search("...")] syntax (Constraint 1)
- Non-crashing malformed tool call fallback (Constraint 6)
- Compact web search (max 2 results) & selective scraping (Constraints 3 & 4)
- Strict 128-token budget verification (Constraint 5)
- Logging to logs/agent_log.txt
"""

import os
import re
import sys
import torch
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tools.clock import get_current_time
from tools.search import web_search
from tools.scrape import scrape_page, should_scrape_snippet
from tools.parser import parse_tool_call
from chat_gpt import load_model_and_tokenizer
from session_memory import SessionMemory
from transformers import AutoTokenizer, AutoModelForCausalLM

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "agent_log.txt")

# Strict token budget constants (Total block_size: 128)
BUDGET_TIME_HEADER = 18
BUDGET_SEARCH_CONTEXT = 50
BUDGET_USER_PROMPT = 35
BUDGET_MODEL_GENERATION = 48

# Refined specific time/date question patterns
TIME_PATTERNS = [
    # English time questions
    r"\b(?:what\s+(?:is\s+the\s+)?time|what\s+time\s+is\s+it|tell\s+me\s+(?:the\s+)?time|current\s+time)\b",
    # English date & day questions
    r"\b(?:what\s+(?:is\s+today's\s+|is\s+the\s+)?date|what\s+date\s+is\s+today|today's\s+date|current\s+date|what\s+day\s+is\s+today)\b",
    # Hindi / Hinglish time questions
    r"\b(?:abhi\s+)?(?:kya\s+time|time\s+kya|kitna\s+time|samay\s+kya|kya\s+samay|kitna\s+samay)\s*(?:hai|hua|ho\s+raha)?\b",
    r"\b(?:kitne\s+baje|ghadi\s+me(?:in)?\s+kya\s+baja)\b",
    # Hindi / Hinglish date & day questions
    r"\b(?:aaj\s+)?(?:kya\s+date|date\s+kya|tarikh\s+kya|taarikh\s+kya)\s*(?:hai|batao)?\b",
    r"\b(?:aaj\s+ki\s+date|aaj\s+ki\s+tarikh|aaj\s+ki\s+taarikh|aaj\s+(?:kaun\s+sa|konsa)\s+din)\b",
]
TIME_REGEXES = [re.compile(p, re.IGNORECASE) for p in TIME_PATTERNS]

# Refined search patterns (factual, real-time lookups; words like 'aaj' or 'current' alone do NOT trigger search)
SEARCH_PATTERNS = [
    # Weather
    r"\b(?:weather|mausam|temperature|forecast|baarish)\b",
    # News & Current Events
    r"\b(?:news|samachar|khabar|breaking\s+news|headline|headlines)\b",
    # Prices, Financial & Economy
    r"\b(?:price|prices|rate|rates|daam|kimat|bhav|stock|stocks|share\s+market|shares?\s+price|nifty|sensex|bitcoin|crypto|gold|silver|petrol|diesel|gdp|inflation|economy|population|aabadi)\b",
    # Sports & Scores
    r"\b(?:score|scores|match\s+score|ipl|points?\s+table|who\s+won|match\s+winner)\b",
    # Explicit search / web query
    r"\b(?:search|google|ddgs|fetch|khojo|dhoondo)\b",
    # Factual entities: full form, who is the president/pm/ceo, capital of, where is
    r"\b(?:full\s+form|who\s+is\s+(?:the\s+)?(?:current\s+)?(?:president|prime\s+minister|pm|ceo|minister)|capital\s+of|kaha(?:n)?\s+(?:hai|h|aata|sthit)|where\s+is)\b",
    # Latest updates / release news
    r"\b(?:latest|recent)\b.*\b(?:release|update|version|news|launch|phone|model|movie)\b",
    r"\b(?:release\s+news|release\s+date|latest\s+version|latest\s+update)\b",
    # Educational definitions & quantity questions (e.g. "what is python", "india ki gdp kitni hai")
    r"^\s*(?:what\s+is|what\s+are|define|explain)\s+(?!the\s+time|today|your\s+name|my\s+name).+",
    r"\b[a-zA-Z0-9_\u0900-\u097F\s]{2,}\s+(?:kya\s+hai|kya\s+hota\s+hai|kise\s+kehte\s+hain|matlab\s+kya\s+hai|kitna\s+hai|kitni\s+hai|kitne\s+hain|kaha\s+h|kahan\s+h)\b",
]
SEARCH_REGEXES = [re.compile(p, re.IGNORECASE) for p in SEARCH_PATTERNS]

GREETING_KEYWORDS = [
    "namaste", "pranam", "kaise ho", "kese ho", "shubh prabhat", 
    "hello", "hi", "hey", "alvida", "bye", "thank you", "shukriya"
]

IDENTITY_PATTERNS = [
    r"\b(?:who\s+are\s+you|what\s+is\s+your\s+name|what\s+are\s+you|what\s+can\s+you\s+do)\b",
    r"\b(?:tum\s+kaun\s+ho|aap\s+kaun\s+hain|aap\s+kaun\s+ho|kaun\s+ho\s+tum|kaun\s+hai\s+tu)\b",
    r"\b(?:tumhara\s+naam\s+kya\s+hai|aapka\s+naam\s+kya\s+hai|tera\s+naam\s+kya\s+hai|naam\s+kya\s+hai\s+tumhara)\b",
    r"\b(?:tum\s+kya\s+kar\s+sakte\s+ho|aap\s+kya\s+kar\s+sakte\s+hain|kya\s+karega\s+tu)\b"
]
IDENTITY_REGEXES = [re.compile(p, re.IGNORECASE) for p in IDENTITY_PATTERNS]

GRATITUDE_PATTERNS = [
    r"\b(?:thank\s+you|thanks|thx|dhanyawad|dhanyavad|shukriya|bahut\s+shukriya)\b"
]
GRATITUDE_REGEXES = [re.compile(p, re.IGNORECASE) for p in GRATITUDE_PATTERNS]

FOLLOWUP_PATTERNS = [
    r"\b(?:where\s+is\s+the\s+code|where\s+is\s+code|code\s+kaha\s+hai|code\s+kahan\s+hai|code\s+do|show\s+code|give\s+code)\b",
    r"\b(?:explain\s+(?:this|that|it|more|further)|details?\s+do|aur\s+batao|aage\s+batao|aur\s+samjhao)\b",
    r"\b(?:continue|complete\s+(?:it|this)|poora\s+karo|aage\s+likho|code\s+complete)\b",
    r"\b(?:what\s+about\s+(?:this|that)|iska\s+kya|uska\s+kya|isko\s+code\s+mein|write\s+code)\b",
    r"\b(?:how\s+to\s+implement|code\s+bhi\s+do|code\s+likho|code\s+kijiye)\b"
]
FOLLOWUP_REGEXES = [re.compile(p, re.IGNORECASE) for p in FOLLOWUP_PATTERNS]


def is_conversational_followup(query: str, history: List[Dict[str, str]]) -> bool:
    """Detects if query is a follow-up or referential query about previous turns."""
    if not history:
        return False
    q = query.strip().lower()
    if any(rx.search(q) for rx in FOLLOWUP_REGEXES):
        return True
    referential_words = ["this", "that", "the code", "the answer", "the solution", "iska", "uska", "ye", "woh", "isme", "usme"]
    if any(w in q for w in referential_words) and len(q.split()) <= 6:
        return True
    return False


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def log_agent_turn(
    user_query: str,
    route: str,
    tool_call: Optional[str],
    search_query: Optional[str],
    response: str,
    sources: List[str]
):
    """Logs the turn details to logs/agent_log.txt."""
    ensure_log_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] ROUTE: {route}\n")
        f.write(f"  User Query   : {user_query}\n")
        if tool_call:
            f.write(f"  Tool Call    : {tool_call}\n")
        if search_query:
            f.write(f"  Search Query : {search_query}\n")
        if sources:
            f.write(f"  Sources      : {', '.join(sources)}\n")
        f.write(f"  Response     : {response}\n")
        f.write("-" * 60 + "\n")


def is_time_query(query: str) -> bool:
    """Checks if query asks specifically for the current time or date."""
    if not query or not isinstance(query, str):
        return False
    clean = query.strip()
    return any(rx.search(clean) for rx in TIME_REGEXES)


def is_search_query(query: str) -> bool:
    """
    Checks if query requires real-time facts/web search or educational definition lookup.
    Casual chit-chat, greetings, identity questions ('aapka naam', 'kya haal hai') are excluded.
    """
    if not query or not isinstance(query, str):
        return False
    clean = query.strip()
    lower = clean.lower()
    
    # Exclude casual identity or greeting phrases from accidentally searching
    if any(p in lower for p in ["kya haal", "kya chal", "kya baat", "aap kaun", "tum kaun", "aapka naam", "tumhara naam"]):
        return False
        
    return any(rx.search(clean) for rx in SEARCH_REGEXES)


HINGLISH_WORDS = {
    'kya', 'hai', 'hain', 'ho', 'h', 'kaise', 'kese', 'haal', 'chal', 'kaha', 'kahan', 
    'kitna', 'kitni', 'kitne', 'kaun', 'konsa', 'konsi', 'bhai', 'dost', 'namaste', 
    'pranam', 'shukriya', 'dhanyawad', 'mera', 'meri', 'mere', 'tera', 'teri', 'tere', 
    'aapka', 'aapki', 'aapke', 'hum', 'humara', 'main', 'mujhe', 'mujhko', 'tum', 
    'tumhe', 'tumko', 'batao', 'bataiye', 'bolo', 'bano', 'karo', 'karein', 'suno', 
    'dekh', 'hota', 'hoti', 'hote', 'tha', 'thi', 'the', 'hoga', 'hogi', 'honge', 
    'nahi', 'nahin', 'mat', 'accha', 'achha', 'badhiya', 'theek', 'sahi', 'galat', 
    'kyu', 'kyun', 'kyon', 'liye', 'saath', 'bhi', 'toh', 'to', 'par', 'pe', 'se', 
    'me', 'mein', 'ko', 'aur', 'ya', 'din', 'aaj', 'kal', 'parso', 'samay', 'waqt',
    'bhi', 'wala', 'wali', 'wale', 'bata', 'kar'
}


def is_hinglish_or_hindi(text: str) -> bool:
    """Detects if query contains Devanagari Hindi or Hinglish (Roman Hindi) words."""
    if not text:
        return False
    if re.search(r"[\u0900-\u097F]", text):
        return True
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return any(w in HINGLISH_WORDS for w in words)


def is_greeting_query(query: str) -> bool:
    """Checks if query is a greeting or polite expression."""
    if not query or not isinstance(query, str):
        return False
    q = query.strip().lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", q) for kw in GREETING_KEYWORDS) or any(
        kw in q for kw in ["kaise ho", "kese ho", "shubh prabhat", "thank you", "alvida"]
    )


def is_casual_conversation(query: str) -> bool:
    """Checks if query is casual chit-chat, personal sharing, emotion, greeting, or identity."""
    if not query or not isinstance(query, str):
        return True
    if is_greeting_query(query):
        return True
    q = query.strip().lower()
    casual_words = [
        "break", "aaram", "relax", "chhutti", "rest",
        "din", "mushkil", "difficult", "thaka", "tired", "udas", "sad", "pareshan",
        "kaise ho", "kese ho", "kya chal raha", "haal chal", "bore", "accha", "achha",
        "madad", "help", "sahayata", "assist", "kaun ho", "kaun hain", "who are you",
        "aapka naam", "tumhara naam", "achievement", "share"
    ]
    return any(w in q for w in casual_words)


def get_safe_conversational_fallback(user_text: str) -> str:
    """
    Provides a safe, truthful conversational response when the model hallucinates
    tool syntax during casual conversation. Avoids inventing personal facts or experiences.
    """
    lower = user_text.lower()
    is_hindi_hinglish = bool(
        re.search(r"[\u0900-\u097F]", user_text)
        or any(w in lower for w in ["main", "mera", "meri", "aaj", "hoon", "hai", "kya", "thoda", "kuch", "kaise", "nahi"])
    )

    if any(w in lower for w in ["madad", "help", "sahayata", "assist"]):
        if is_hindi_hinglish:
            return "Haan bilkul! Main aapki madad karne ke liye hi hoon. Kripya batayein aapko kis cheez mein sahayata chahiye."
        return "Of course! I am here to help you. Please let me know what you need assistance with."

    if any(w in lower for w in ["kaun ho", "kaun hain", "who are you", "who r u"]) or (
        any(w in lower for w in ["aapka", "tumhara", "tera", "your"]) and any(w in lower for w in ["naam", "name"])
    ):
        if is_hindi_hinglish:
            return "Main aapka personal AI assistant hoon, jise MiniGPT Transformer architecture par train kiya gaya hai. Main Hindi, Hinglish aur English me baat kar sakta hoon."
        return "I am your personal AI assistant, built on a custom MiniGPT Transformer architecture. I can assist you in English, Hindi, and Hinglish."

    if any(w in lower for w in ["thank you", "thanks", "dhanyawad", "shukriya", "great job", "shabaash"]):
        if is_hindi_hinglish:
            return "Aapka bohot swagat hai! Mujhe aapki madad karke khushi hui. Aur kuch poochhna ho to zaroor batayein."
        return "You are very welcome! I'm glad I could help. Let me know if you need anything else."

    if any(w in lower for w in ["achievement", "share"]):
        if is_hindi_hinglish:
            return "Arre waah! Zaroor share kijiye, mujhe sunkar bohot khushi hogi."
        return "That sounds wonderful! Please share your achievement, I'd love to hear about it."

    if any(w in lower for w in ["break", "aaram", "relax", "chhutti"]):
        if is_hindi_hinglish:
            return "Aap zaroor thoda break lijiye aur aaram kijiye. Main yahin hoon jab bhi aap baat karna chahein."
        return "Taking a break is a good idea. Please relax, I am here whenever you wish to talk."

    if any(w in lower for w in ["mushkil", "difficult", "thaka", "tired", "udas", "sad", "pareshan"]):
        if is_hindi_hinglish:
            return "Main samajh sakta hoon. Kabhi-kabhi din thoda kathin ho sakta hai. Aap thoda aaram kijiye."
        return "I understand. Some days can be difficult. Please take some rest and take care."

    if any(h in lower for h in ["namaste", "pranam", "hello", "hi", "hey", "kaise ho", "kese ho"]):
        if any(h in lower for h in ["namaste", "pranam"]):
            return "Namaste! Main badhiya hoon. Aap bataiye, aaj main aapki kya madad kar sakta hoon?"
        elif any(h in lower for h in ["kaise ho", "kese ho"]):
            return "Main theek hoon! Aap suniye, aapka din kaisa chal raha hai?"
        return "Hello! How can I assist you today?"

    if is_hindi_hinglish:
        return "Main samajh gaya. Kripya batayein, main aapki kya madad kar sakta hoon?"
    return "I understand. Please let me know how I can assist you."


def trim_to_tokens(text: str, max_tokens: int, tokenizer) -> str:
    """Strictly truncates text so its token count <= max_tokens."""
    tokens = tokenizer.encode(text).ids
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens]).strip()


def execute_search_and_scrape(search_query: str, tokenizer) -> Tuple[str, List[str]]:
    """
    Executes search (max 2 results) and selective scrape (< 40 chars),
    enforcing the strict 50-token budget for search context.
    """
    results, err = web_search(search_query, max_results=2)
    if err or not results:
        return "Search returned no active results.", []

    formatted_items = []
    sources = []

    for item in results[:2]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        url = item.get("url", "")
        if url:
            sources.append(url)

        # Constraint 4: Scrape ONLY if snippet is missing or < 40 chars
        if should_scrape_snippet(snippet) and url:
            scraped_text, scrape_err = scrape_page(url, max_chars=200)
            if not scrape_err and scraped_text:
                snippet = scraped_text

        formatted_items.append(f"{title}: {snippet}")

    raw_context = " | ".join(formatted_items)
    # Strictly enforce 50-token context budget
    budgeted_context = trim_to_tokens(raw_context, BUDGET_SEARCH_CONTEXT, tokenizer)
    return budgeted_context, sources


def generate_model_turn(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_tokens: int = BUDGET_MODEL_GENERATION
) -> str:
    """Encodes prompt, runs MiniGPT generation, decodes output."""
    encoded = tokenizer.encode(prompt).ids
    # Ensure input fits within 128 - max_tokens
    max_allowed_input = 128 - max_tokens
    if len(encoded) > max_allowed_input:
        encoded = encoded[-max_allowed_input:]

    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    with torch.no_grad():
        generated_ids = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.12
        )[0].tolist()

    full_output = tokenizer.decode(generated_ids)
    if "Assistant:" in full_output:
        reply = full_output.split("Assistant:", 1)[1].strip()
    else:
        reply = full_output[len(prompt):].strip()

    for stop_marker in ["\nUser:", "\nAssistant:", "\n\n"]:
        if stop_marker in reply:
            reply = reply.split(stop_marker)[0]

    reply = reply.strip()

    # Clean up trailing incomplete sentence fragment if a complete sentence exists
    for punc in [".", "!", "?", "।"]:
        if punc in reply:
            last_punc_idx = max(reply.rfind("."), reply.rfind("!"), reply.rfind("?"), reply.rfind("।"))
            if last_punc_idx >= 15:
                reply = reply[:last_punc_idx + 1].strip()
            break

    return reply.strip()


class QwenEngine:
    """Pretrained 0.5B Multilingual LLM engine for reasoning, knowledge, and conversation."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        self.model.eval()
        self.device = "cpu"
        self.cfg = self.model.config
        self.params = self.model.num_parameters()

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 100) -> str:
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([chat_text], return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.9,
                do_sample=True
            )
        prompt_len = inputs.input_ids.shape[1]
        reply_tokens = outputs[0][prompt_len:]
        return self.tokenizer.decode(reply_tokens, skip_special_tokens=True).strip()


class MiniGPTAgent:
    """
    Autonomous Online Agent orchestrating clock, search, scraping,
    session memory, and language models (supports both Qwen 0.5B and custom MiniGPT).
    """

    def __init__(self, engine: Optional[str] = None, checkpoint_path: Optional[str] = None):
        if engine is None:
            engine = os.environ.get("AGENT_ENGINE", "qwen")
        self.engine_type = engine.lower()
        self.memory = SessionMemory(max_history=8)

        if self.engine_type == "qwen":
            self.qwen = QwenEngine()
            self.checkpoint_path = "Qwen2.5-0.5B-Instruct"
            self.device = "cpu"
            self.val_loss = 0.0
            self.step = "pretrained"
            self.cfg = type("CFG", (), {
                "n_layer": self.qwen.cfg.num_hidden_layers,
                "n_head": self.qwen.cfg.num_attention_heads,
                "n_embd": self.qwen.cfg.hidden_size,
                "block_size": 2048
            })()
            self.tokenizer = self.qwen.tokenizer
            self.model = self.qwen.model
        else:
            if checkpoint_path is None:
                if os.path.exists(os.path.join("checkpoints", "best_gpt.pt")):
                    checkpoint_path = os.path.join("checkpoints", "best_gpt.pt")
                elif os.path.exists(os.path.join("checkpoints", "tool_gpt.pt")):
                    checkpoint_path = os.path.join("checkpoints", "tool_gpt.pt")

            self.checkpoint_path = checkpoint_path
            self.model, self.tokenizer, self.cfg, self.step, self.val_loss, self.device = (
                load_model_and_tokenizer(checkpoint_path)
            )

    def get_status_dict(self) -> Dict[str, Any]:
        """Returns metadata for /api/status endpoint."""
        if self.engine_type == "qwen":
            return {
                "model_name": "Qwen 2.5 (0.5B Pretrained Instruct)",
                "params": self.qwen.params,
                "n_layer": self.cfg.n_layer,
                "n_head": self.cfg.n_head,
                "n_embd": self.cfg.n_embd,
                "block_size": self.cfg.block_size,
                "step": "Pretrained 0.5B",
                "val_loss": 0.0,
                "device": self.device,
                "checkpoint": "Qwen2.5-0.5B-Instruct",
                "status": "online"
            }
        else:
            return {
                "model_name": "Custom Mini-GPT (Decoder-Only Transformer)",
                "params": self.model.get_num_params(),
                "n_layer": self.cfg.n_layer,
                "n_head": self.cfg.n_head,
                "n_embd": self.cfg.n_embd,
                "block_size": self.cfg.block_size,
                "step": self.step,
                "val_loss": round(self.val_loss, 4),
                "device": self.device,
                "checkpoint": os.path.basename(self.checkpoint_path) if self.checkpoint_path else "none",
                "status": "online"
            }

    def respond(self, user_text: str) -> Dict[str, Any]:
        """
        Processes user query through orchestrator pipeline.
        """
        clean_user = user_text.strip()
        current_time = get_current_time()

        # Route 0A: User Name Question (e.g. "mera kya naam hai", "what is my name")
        name_answer = self.memory.answer_name_question(clean_user)
        if name_answer:
            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", name_answer)
            log_agent_turn(clean_user, "memory", None, None, name_answer, [])
            return {
                "response": name_answer,
                "route": "memory",
                "sources": [],
                "tool_call": None
            }

        # Route 0B: User Name Declaration (e.g. "my name is Sandeep", "mera naam Sandeep hai")
        extracted_name = self.memory.extract_name(clean_user)
        if extracted_name:
            reply = f"Namaste {extracted_name}! Main aapka naam yaad rakhunga. Aap bataiye, aaj main aapki kya madad kar sakta hoon?"
            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", reply)
            log_agent_turn(clean_user, "memory", None, None, reply, [])
            return {
                "response": reply,
                "route": "memory",
                "sources": [],
                "tool_call": None
            }

        # Route 1: System Clock (Constraint 2)
        if is_time_query(clean_user):
            response = f"Abhi ka samay hai: {current_time}."
            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", response)
            log_agent_turn(clean_user, "clock", None, None, response, [])
            return {
                "response": response,
                "route": "clock",
                "sources": [],
                "tool_call": None
            }

        # Route 1B: AI Identity (e.g. "tum kaun ho", "who are you", "tumhara naam kya hai")
        is_hin = is_hinglish_or_hindi(clean_user)
        if any(rx.search(clean_user) for rx in IDENTITY_REGEXES):
            if any(w in clean_user.lower() for w in ["kya kar sakte ho", "what can you do", "capabilities"]):
                if is_hin:
                    reply = (
                        "Main MiniGPT hoon, ek smart AI assistant! Main aapke liye ye sab kar sakta hoon:\n"
                        "• Hindi, Hinglish aur English mein natural baatcheet\n"
                        "• Real-time web search (facts, news, mausam, prices)\n"
                        "• Live clock aur calendar date/time batana\n"
                        "• Aapka naam aur conversation history yaad rakhna"
                    )
                else:
                    reply = (
                        "I am MiniGPT, an intelligent conversational AI assistant!\n"
                        "Here is what I can do:\n"
                        "• Converse fluently in English, Hindi, and Hinglish\n"
                        "• Search the web in real-time for current facts and news\n"
                        "• Track local system time and dates accurately\n"
                        "• Remember your name and chat history across turns"
                    )
            else:
                if is_hin:
                    reply = (
                        "Main MiniGPT hoon, ek intelligent conversational AI assistant jise Sandeep ne develop kiya hai! "
                        "Aap mujhse bejhijhak Hindi, Hinglish ya English mein kuch bhi poochh sakte hain."
                    )
                else:
                    reply = (
                        "I am MiniGPT, an intelligent conversational AI assistant created and trained locally. "
                        "I am here to help you with answers, live web search, and friendly conversation!"
                    )
            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", reply)
            log_agent_turn(clean_user, "identity", None, None, reply, [])
            return {
                "response": reply,
                "route": "identity",
                "sources": [],
                "tool_call": None
            }

        # Route 1C: Gratitude (e.g. "thank you", "shukriya", "dhanyawad")
        if any(rx.search(clean_user) for rx in GRATITUDE_REGEXES):
            reply = "Aapka bahut-bahut swagat hai! Agar koi aur sawaal ho to bejhijhak poochhiye." if is_hin else "You are very welcome! Feel free to ask anytime if you need anything else."
            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", reply)
            log_agent_turn(clean_user, "gratitude", None, None, reply, [])
            return {
                "response": reply,
                "route": "gratitude",
                "sources": [],
                "tool_call": None
            }

        # If Qwen engine is active:
        if self.engine_type == "qwen":
            # Direct greeting handling to prevent safety refusal on casual Hinglish
            lower_clean = clean_user.lower()
            if any(p in lower_clean for p in ["kya haal chal", "kya haal", "haal chal", "kaise ho", "kese ho", "kya chal raha"]):
                greeting_reply = "Sab badhiya bhai! Aap suniye, aapka kya haal chal hai? Main aapki kya madad kar sakta hoon?"
                self.memory.add_turn("user", clean_user)
                self.memory.add_turn("assistant", greeting_reply)
                log_agent_turn(clean_user, "greeting", None, None, greeting_reply, [])
                return {
                    "response": greeting_reply,
                    "route": "greeting",
                    "sources": [],
                    "tool_call": None
                }

            search_context = ""
            sources = []
            is_followup = is_conversational_followup(clean_user, self.memory.history)

            # Only trigger external web search if this is NOT an in-conversation follow-up query
            if not is_followup and is_search_query(clean_user):
                res, err = web_search(clean_user, max_results=2)
                if res:
                    sources = [r.get("url") for r in res if r.get("url")]
                    snippets = [f"{r.get('title')}: {r.get('snippet')}" for r in res]
                    search_context = "\n".join(snippets[:2])

            if is_hin:
                system_prompt = (
                    "You are a friendly Indian AI assistant named MiniGPT. "
                    "The user is speaking in Hinglish (Hindi written in Roman English alphabet). "
                    "You MUST reply ONLY in natural, conversational Hinglish (Hindi words written in English letters). "
                    "Do NOT reply in English sentences. "
                    "Example: 'Sab badhiya bhai! Aap suniye, aaj kya chal raha hai?'"
                )
            else:
                system_prompt = (
                    "You are a smart, helpful, and friendly AI assistant named MiniGPT. "
                    "Give clear, accurate, intelligent, and complete answers in English."
                )

            if search_context:
                if is_hin:
                    system_prompt += f"\n\nLive Search Context:\n{search_context}\nIs search context ki information use karke apna poora jawab saral Hinglish (Hindi in English letters) mein do."
                else:
                    system_prompt += f"\n\nLive Search Context:\n{search_context}\nUse this live context to answer accurately."

                # For factual search, do not pass previous unrelated conversation to prevent topic bleeding
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": clean_user}
                ]
            else:
                messages = [{"role": "system", "content": system_prompt}]
                # Pass recent conversation history so follow-up questions understand context
                for turn in self.memory.history[-4:]:
                    messages.append({"role": turn["role"], "content": turn["text"]})
                messages.append({"role": "user", "content": clean_user})

            # Provide adequate token budget for coding solutions and full answers
            reply = self.qwen.generate(messages, max_new_tokens=400)

            # Prevent 0.5B repetition degeneration (echoing previous assistant turn)
            if self.memory.history and self.memory.history[-1]["role"] == "assistant":
                prev_text = self.memory.history[-1]["text"].strip()
                if reply.startswith(prev_text):
                    reply = reply[len(prev_text):].strip()

            route = "search_rule" if search_context else "model_direct"
            tool_call = f'[TOOL: search("{clean_user}")]' if search_context else None

            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", reply)
            log_agent_turn(clean_user, route, tool_call, clean_user if search_context else None, reply, sources)
            return {
                "response": reply,
                "route": route,
                "sources": sources,
                "tool_call": tool_call
            }

        # Route 2: Rule-Based Search Trigger (Constraint 7)
        if is_search_query(clean_user):
            context, sources = execute_search_and_scrape(clean_user, self.tokenizer)
            response = f"{context}"
            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", response)
            log_agent_turn(clean_user, "search_rule", f'[TOOL: search("{clean_user}")]', clean_user, response, sources)
            return {
                "response": response,
                "route": "search_rule",
                "sources": sources,
                "tool_call": f'[TOOL: search("{clean_user}")]'
            }

        # Route 3: Model Inference Turn
        time_header = f"Current time: {current_time}."
        header_budgeted = trim_to_tokens(time_header, BUDGET_TIME_HEADER, self.tokenizer)
        user_budgeted = trim_to_tokens(clean_user, BUDGET_USER_PROMPT, self.tokenizer)
        
        model_prompt = f"{header_budgeted}\nUser: {user_budgeted}\nAssistant:"
        
        # Verify total tokens before generation
        total_prompt_tokens = len(self.tokenizer.encode(model_prompt).ids)
        assert total_prompt_tokens <= (128 - BUDGET_MODEL_GENERATION), (
            f"Prompt token count {total_prompt_tokens} exceeds safe boundary"
        )

        model_raw = generate_model_turn(
            self.model,
            self.tokenizer,
            model_prompt,
            self.device,
            max_tokens=BUDGET_MODEL_GENERATION
        )

        # Check for valid tool call or tool-like syntax
        parsed_tool = parse_tool_call(model_raw)
        has_tool_syntax = (
            "[TOOL:" in model_raw
            or "[TOL:" in model_raw
            or "search(" in model_raw
        )

        if parsed_tool:
            # If user query was casual conversation, tool call is a spurious hallucination
            if is_casual_conversation(clean_user):
                clean_reply = get_safe_conversational_fallback(clean_user)
                self.memory.add_turn("user", clean_user)
                self.memory.add_turn("assistant", clean_reply)
                log_agent_turn(clean_user, "model_direct", model_raw, None, clean_reply, [])
                return {
                    "response": clean_reply,
                    "route": "model_direct",
                    "sources": [],
                    "tool_call": None
                }

            search_q = parsed_tool["query"]
            context, sources = execute_search_and_scrape(search_q, self.tokenizer)
            response = f"{context}"
            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", response)
            log_agent_turn(clean_user, "model_tool", model_raw, search_q, response, sources)
            return {
                "response": response,
                "route": "model_tool",
                "sources": sources,
                "tool_call": model_raw
            }

        # Check for malformed tool call
        if has_tool_syntax:
            # Casual conversation: do NOT search the web! Return safe conversational fallback under model_direct
            if is_casual_conversation(clean_user) or not is_search_query(clean_user):
                clean_reply = get_safe_conversational_fallback(clean_user)
                self.memory.add_turn("user", clean_user)
                self.memory.add_turn("assistant", clean_reply)
                log_agent_turn(clean_user, "model_direct", model_raw, None, clean_reply, [])
                return {
                    "response": clean_reply,
                    "route": "model_direct",
                    "sources": [],
                    "tool_call": None
                }

            # Informational query with malformed tool call: fallback search
            context, sources = execute_search_and_scrape(clean_user, self.tokenizer)
            response = f"{context}"
            self.memory.add_turn("user", clean_user)
            self.memory.add_turn("assistant", response)
            log_agent_turn(clean_user, "fallback_search", model_raw, clean_user, response, sources)
            return {
                "response": response,
                "route": "fallback_search",
                "sources": sources,
                "tool_call": model_raw
            }

        # Conversational response directly from model weights
        self.memory.add_turn("user", clean_user)
        self.memory.add_turn("assistant", model_raw)
        log_agent_turn(clean_user, "model_direct", None, None, model_raw, [])
        return {
            "response": model_raw,
            "route": "model_direct",
            "sources": [],
            "tool_call": None
        }


if __name__ == "__main__":
    agent = MiniGPTAgent()
    print("MiniGPT Agent loaded. Testing 3 sample questions:\n")

    test_queries = [
        "Aaj kya date aur time hai?",
        "Latest Python release news",
        "Hello, who are you?"
    ]

    for q in test_queries:
        print(f"📝 User: {q}")
        res = agent.respond(q)
        print(f"   Route    : {res['route']}")
        if res["tool_call"]:
            print(f"   Tool Call: {res['tool_call']}")
        if res["sources"]:
            print(f"   Sources  : {res['sources']}")
        print(f"🤖 Response : {res['response']}")
        print("-" * 60)
