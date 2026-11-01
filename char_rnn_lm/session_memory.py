"""
session_memory.py
Lightweight session memory module for MiniGPT.
Maintains in-memory user name tracking and short multi-turn conversation history.
Does not depend on PyTorch or neural model weights.
"""

import re
from typing import Optional, List, Dict, Any

# Stop words and common non-name terms that must never be treated as names
NAME_STOP_WORDS = {
    # English adjectives & common words
    "good", "fine", "here", "sure", "well", "great", "nice", "cool", "awesome",
    "bad", "happy", "sad", "sorry", "okay", "ok", "ready", "alone", "busy",
    "better", "best", "right", "wrong", "trying", "learning", "listening",
    # English verbs & common conversational words
    "telling", "tellung", "saying", "asking", "speaking", "talking", "writing",
    "just", "only", "here", "there", "now", "again", "already",
    # English nouns / roles / pronouns
    "student", "teacher", "doctor", "engineer", "developer", "coder",
    "boy", "girl", "man", "woman", "person", "human", "user", "someone", "anyone",
    "nobody", "nothing", "bot", "assistant", "ai", "model", "robot", "computer",
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "from",
    # Hindi / Hinglish adjectives & roles
    "ek", "theek", "thik", "badhiya", "kalyan", "khush", "udas", "chhatra",
    "ladka", "ladki", "aadmi", "aurat", "insan", "bachha", "dost",
    "kaun", "kya", "kaisa", "kaise", "nahi", "nahin", "nhi", "mat",
    "haan", "han", "kuch", "sab", "aap", "tum", "hum", "main", "mai", "me",
    # Assistant / system identity
    "minigpt", "mini-gpt"
}

# Regex patterns for name declaration statements
NAME_EXTRACTION_PATTERNS = [
    # "my name is Sandeep" / "my name Sandeep"
    re.compile(r"\bmy\s+name\s+(?:is\s+)?([A-Za-z\u0900-\u097F]+)\b", re.IGNORECASE),
    # "mera naam Sandeep hai" / "mera naam Sandeep" / "mera naam hai Sandeep"
    re.compile(r"\bmera\s+naam\s+(?:hai\s+)?([A-Za-z\u0900-\u097F]+)(?:\s+hai)?\b", re.IGNORECASE),
    # "I am Priya" / "I'm Priya" (must end name at sentence boundary)
    re.compile(r"\b(?:i\s+am|i'm|im)\s+([A-Za-z\u0900-\u097F]+)(?:[.,!?\s]*$)", re.IGNORECASE),
    # "main Sandeep hoon" / "mai Sandeep hu" / "me Sandeep hu"
    re.compile(r"\b(?:main|mai|me)\s+([A-Za-z\u0900-\u097F]+)\s+(?:hoon|hu|hn)\b", re.IGNORECASE),
]

# Regex patterns for user asking for their own name
NAME_QUERY_PATTERNS = [
    re.compile(r"\bwhat(?:'s|\s+is)\s+my\s+name\b", re.IGNORECASE),
    re.compile(r"\b(?:do\s+you\s+know|remember)\s+my\s+name\b", re.IGNORECASE),
    re.compile(r"\btell\s+me\s+my\s+name\b", re.IGNORECASE),
    re.compile(r"\bwho\s+am\s+i\b", re.IGNORECASE),
    re.compile(r"\bmera\s+(?:kya\s+)?naam\s+(?:kya\s+)?(?:hai|batao|yaad\s+hai)\b", re.IGNORECASE),
    re.compile(r"\bmera\s+naam\s+batao\b", re.IGNORECASE),
    re.compile(r"\bmera\s+naam\s+kya\b", re.IGNORECASE),
    re.compile(r"\bmujhe\s+pehchano\b", re.IGNORECASE),
    re.compile(r"\bkya\s+aapko\s+mera\s+naam\s+(?:pata|yaad)\s+hai\b", re.IGNORECASE),
]


class SessionMemory:
    """
    Session memory managing user name and conversation history.
    Limits conversation history to a fixed number of recent turns.
    """

    def __init__(self, max_history: int = 6):
        self.user_name: Optional[str] = None
        self.history: List[Dict[str, str]] = []
        self.max_history: int = max_history

    def extract_name(self, text: str) -> Optional[str]:
        """
        Detects if text contains a user name declaration.
        Validates against stop words and filters non-name descriptors.
        If valid, stores capitalized name in self.user_name and returns it.
        Otherwise returns None.
        """
        if not text or not isinstance(text, str):
            return None

        clean = text.strip()
        if not clean:
            return None

        for pattern in NAME_EXTRACTION_PATTERNS:
            match = pattern.search(clean)
            if match:
                candidate = match.group(1).strip()
                # Must be at least 2 characters and strictly alphabetical
                if len(candidate) < 2 or not candidate.isalpha():
                    continue

                # Reject known stop words and descriptors
                if candidate.lower() in NAME_STOP_WORDS:
                    continue

                # Format properly (e.g., 'sandeep' -> 'Sandeep')
                capitalized_name = candidate.capitalize()
                self.user_name = capitalized_name
                return capitalized_name

        return None

    def answer_name_question(self, text: str) -> Optional[str]:
        """
        Checks if text asks for the user's name.
        If it is a name question:
            - If user_name is set, returns 'Aapka naam {name} hai.'
            - If user_name is not set, returns 'Aapne abhi tak apna naam nahi bataya.'
        If it is not a name question, returns None.
        """
        if not text or not isinstance(text, str):
            return None

        clean = text.strip()
        if not clean:
            return None

        is_question = any(pattern.search(clean) for pattern in NAME_QUERY_PATTERNS)
        if not is_question:
            return None

        if self.user_name:
            return f"Aapka naam {self.user_name} hai."
        else:
            return "Aapne abhi tak apna naam nahi bataya."

    def add_turn(self, role: str, text: str):
        """
        Adds a dialogue turn to conversation history.
        Keeps at most max_history recent turns.
        """
        if not text:
            return
        self.history.append({"role": role, "text": text.strip()})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def reset(self):
        """Resets stored user name and clears conversation history."""
        self.user_name = None
        self.history = []
