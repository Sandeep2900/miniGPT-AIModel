"""
tools/parser.py
Strict, safe parser for model tool calls.
Matches exactly: [TOOL: search("<query>")]
Rejects any malformed call, multiple calls, or empty query.
"""

import re
from typing import Optional, Dict

# Strict regex matching exactly: [TOOL: search("...")]
# Must start with [TOOL: search(" and end with ")]
TOOL_CALL_PATTERN = re.compile(r'^\[TOOL: search\("([^"\r\n]+)"\)\]$')


def parse_tool_call(raw_text: str) -> Optional[Dict[str, str]]:
    """
    Parses a generated text string to check if it contains a valid, single tool call.
    
    Rules:
    - Must match exactly: [TOOL: search("query")]
    - Query must be non-empty and non-whitespace.
    - Multiple tool calls in the same text are strictly rejected (returns None).
    - Malformed brackets, missing quotes, or extra spaces are rejected (returns None).
    
    Returns:
        dict: {"name": "search", "query": "<query>"} if valid.
        None: If text is plain conversational response or invalid format.
    """
    if not raw_text or not isinstance(raw_text, str):
        return None

    cleaned = raw_text.strip()

    # Reject if multiple tool call markers exist in the same turn
    if cleaned.count("[TOOL:") != 1 or cleaned.count("]") != 1:
        return None

    match = TOOL_CALL_PATTERN.match(cleaned)
    if not match:
        return None

    query = match.group(1).strip()
    if not query:
        return None

    return {
        "name": "search",
        "query": query
    }
