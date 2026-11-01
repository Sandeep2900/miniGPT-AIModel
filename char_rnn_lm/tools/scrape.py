"""
tools/scrape.py
Compact web scraper with strict SSRF protection, robots.txt check, and 200-char truncation.
Enforces User Constraint 4:
Scraping runs ONLY when a snippet is missing or < 40 characters; output truncated to 200 chars.
"""

import ipaddress
import socket
import urllib.robotparser
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def is_safe_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validates URL to protect against SSRF (Server-Side Request Forgery).
    Rejects:
    - Non http/https schemes (file://, gopher://, ftp://)
    - Localhost names ('localhost', '127.0.0.1', '::1')
    - Private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, loopback, link-local)
    """
    if not url or not isinstance(url, str):
        return False, "URL is empty or invalid"

    try:
        parsed = urlparse(url.strip())
        if parsed.scheme not in ("http", "https"):
            return False, f"Unsupported scheme '{parsed.scheme}'. Only http and https allowed."

        hostname = parsed.hostname
        if not hostname:
            return False, "Missing hostname in URL"

        hostname_lower = hostname.lower()
        if hostname_lower in ("localhost", "127.0.0.1", "::1"):
            return False, "SSRF Blocked: Localhost access is prohibited"

        # Resolve hostname to all associated IP addresses
        addr_info = socket.getaddrinfo(hostname, None)
        for family, socktype, proto, canonname, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local or ip.is_multicast:
                return False, f"SSRF Blocked: IP '{ip_str}' is within a private/restricted subnet"

        return True, None
    except Exception as e:
        return False, f"URL validation failed: {str(e)}"


def is_allowed_by_robots(url: str, user_agent: str = "*") -> bool:
    """
    Checks if crawling this URL path is permitted by the domain's robots.txt.
    Uses safe 3-second timeout. Defaults to True if robots.txt is missing/404.
    """
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        resp = requests.get(
            robots_url,
            headers={"User-Agent": "MiniGPT-Assistant/1.0"},
            timeout=3,
            allow_redirects=True
        )
        if resp.status_code == 200:
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(resp.text.splitlines())
            return rp.can_fetch(user_agent, url)
        return True
    except Exception:
        # If robots.txt cannot be fetched or times out, proceed safely
        return True


def should_scrape_snippet(snippet: Optional[str]) -> bool:
    """
    User Constraint 4:
    Scraping runs ONLY when a snippet is missing or < 40 characters.
    """
    if not snippet:
        return True
    return len(snippet.strip()) < 40


def scrape_page(url: str, max_chars: int = 200) -> Tuple[str, Optional[str]]:
    """
    Fetches and extracts clean, plain text from a web page.
    
    Rules:
    1. SSRF check: Refuses private/local IPs.
    2. robots.txt check: Checks site policy.
    3. HTML parsing: Removes scripts, styles, header, footer, nav.
    4. Truncation: Max 200 characters to prevent 128-token overflow.
    
    Returns:
        tuple: (clean_text, error_message)
    """
    safe, err = is_safe_url(url)
    if not safe:
        return "", err

    if not is_allowed_by_robots(url):
        return "", "Disallowed by site robots.txt"

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "MiniGPT-Assistant/1.0"},
            timeout=5,
            allow_redirects=True
        )
        if resp.status_code != 200:
            return "", f"HTTP error {resp.status_code}"

        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip noisy elements
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "form"]):
            tag.decompose()

        # Extract text from paragraph tags first, or body text
        paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if p])
        if not text:
            text = soup.get_text()

        # Normalize multiple spaces and newlines
        clean_text = " ".join(text.split())

        # Strict 200 character truncation
        truncated = clean_text[:max_chars].strip()
        return truncated, None
    except Exception as e:
        return "", f"Scraping error: {str(e)}"


if __name__ == "__main__":
    import sys
    test_target = sys.argv[1] if len(sys.argv) > 1 else "https://en.wikipedia.org/wiki/Python_(programming_language)"
    print(f"Testing Scraper on: {test_target}")
    scraped, error = scrape_page(test_target, max_chars=200)
    if error:
        print(f"Result: FAILED -> {error}")
    else:
        print(f"Result: SUCCESS ({len(scraped)} chars):")
        print(f"'{scraped}'")
