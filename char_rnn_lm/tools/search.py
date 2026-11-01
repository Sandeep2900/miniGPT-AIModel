"""
tools/search.py
Online web search tool using DuckDuckGo (ddgs).
Returns at most 2 results: {title, snippet, url}.
Handles exceptions safely without crashing, returning (results, error_msg).
"""

from typing import List, Dict, Tuple, Optional

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


def web_search(query: str, max_results: int = 2) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    Performs a live web search for the given query.
    
    Args:
        query: Search string.
        max_results: Max results to fetch (strictly capped at 2 for token budget).
        
    Returns:
        tuple: (results_list, error_message)
            results_list: list of dicts with keys 'title', 'snippet', 'url' (max 2).
            error_message: None if successful, or error description string if failed.
    """
    clean_query = query.strip() if query else ""
    if not clean_query:
        return [], "Search query is empty"

    if DDGS is None:
        return [], "ddgs library is not installed"

    results: List[Dict[str, str]] = []
    try:
        ddgs_client = DDGS()
        raw_items = list(ddgs_client.text(clean_query, max_results=max_results))
        
        for item in raw_items[:max_results]:
            title = (item.get("title") or "").strip()
            # body or snippet
            snippet = (item.get("body") or item.get("snippet") or "").strip()
            # Clean up newlines in snippet
            snippet = " ".join(snippet.split())
            url = (item.get("href") or item.get("link") or item.get("url") or "").strip()
            
            results.append({
                "title": title,
                "snippet": snippet,
                "url": url
            })
            
        return results, None
    except Exception as e:
        return [], f"Search failed: {str(e)}"


if __name__ == "__main__":
    import sys
    test_q = sys.argv[1] if len(sys.argv) > 1 else "latest Python release"
    print(f"Searching for: '{test_q}'...")
    res, err = web_search(test_q, max_results=2)
    if err:
        print(f"Error: {err}")
    else:
        print(f"Found {len(res)} results:")
        for i, r in enumerate(res, 1):
            print(f"[{i}] {r['title']}")
            print(f"    Snippet: {r['snippet'][:120]}...")
            print(f"    URL: {r['url']}")
