"""
Fetch Hindi Wikipedia articles for character-level language model training.
Fetches random articles from hi.wikipedia.org MediaWiki API with:
- User-Agent header
- 1-second sequential request rate limiting
- Filtering of short stubs (< 1500 chars)
- Removal of headings, citation markers, non-target characters
- NFC normalization
- Discarding lines with > 30% Latin characters or < 30 total characters
- Target: >= 300,000 Devanagari characters or 600 articles
- Saves to data/raw/hindi_wiki.txt (UTF-8)
"""

import os
import sys
import time
import json
import re
import unicodedata
import urllib.request
import urllib.parse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

USER_AGENT = "CharRNNMultilingualResearch/1.0 (https://github.com/my-research-project; student-nlp@project.local) python-urllib"
API_ENDPOINT = "https://hi.wikipedia.org/w/api.php"
OUTPUT_FILE = os.path.join("data", "raw", "hindi_wiki.txt")

TARGET_DEVANAGARI_CHARS = 300000
MAX_ARTICLES = 600
MIN_STUB_CHARS = 1500
REQUEST_DELAY_SEC = 1.0

# Allowed punctuation set: basic punctuation (‘ ’ “ ” – — … । ॥)
ALLOWED_EXTRA_CHARS = set("‘’“”–—…।॥\n\r\t")


def is_allowed_char(ch: str) -> bool:
    """Keep only Devanagari block, ASCII printable, and basic Hindi/typography punctuation."""
    code = ord(ch)
    # Devanagari Unicode Block: 0x0900 - 0x097F
    if 0x0900 <= code <= 0x097F:
        return True
    # ASCII printable (space to ~)
    if 32 <= code <= 126:
        return True
    # Newlines
    if ch in ("\n", "\r"):
        return True
    # Basic typography punctuation
    if ch in ALLOWED_EXTRA_CHARS:
        return True
    return False


def clean_article_text(raw_text: str) -> str:
    """Cleans raw text extract according to strict user specifications."""
    # 1. NFC Normalize
    normalized = unicodedata.normalize("NFC", raw_text)

    # 2. Remove citations [1], [2], [citation needed], etc.
    cleaned = re.sub(r"\[\d+\]", "", normalized)
    cleaned = re.sub(r"\[[a-zA-Z\s]+\]", "", cleaned)

    # 3. Process line-by-line
    kept_lines = []
    for line in cleaned.split("\n"):
        line_stripped = line.strip()

        # Remove '== Heading ==' lines
        if re.match(r"^==+.*==+$", line_stripped):
            continue

        # Filter characters: keep only Devanagari, ASCII printable, allowed punctuation
        char_filtered = "".join(ch for ch in line_stripped if is_allowed_char(ch))
        char_filtered = char_filtered.strip()

        # Discard lines shorter than 30 characters
        if len(char_filtered) < 30:
            continue

        # Discard lines where Latin letters > 30% of characters
        latin_letters = sum(1 for ch in char_filtered if ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'))
        if (latin_letters / len(char_filtered)) > 0.30:
            continue

        kept_lines.append(char_filtered)

    return "\n".join(kept_lines)


def make_api_request(params: dict, max_retries: int = 3) -> dict:
    """Executes rate-limited MediaWiki API request with custom User-Agent and backoff on 429."""
    import gzip
    query_string = urllib.parse.urlencode(params)
    url = f"{API_ENDPOINT}?{query_string}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(max_retries):
        time.sleep(REQUEST_DELAY_SEC)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw_bytes = resp.read()
                if resp.info().get("Content-Encoding") == "gzip":
                    raw_bytes = gzip.decompress(raw_bytes)
                data = raw_bytes.decode("utf-8")
                return json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries - 1:
                wait_time = (attempt + 1) * 3
                print(f"\n[RATE-LIMIT] 429 Too Many Requests received. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
            print(f"\n[ERROR] HTTP Error {e.code}: {e.reason}")
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"\n[ERROR] API request failed: {e}")
            raise


def fetch_random_titles(batch_size: int = 20) -> list:
    """Fetches random article titles from namespace 0."""
    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": str(batch_size),
        "format": "json"
    }
    data = make_api_request(params)
    pages = data.get("query", {}).get("pages", {})
    return [p["title"] for p in pages.values() if "title" in p]


def fetch_article_extract(title: str) -> str:
    """Fetches plain text extract for a given article title."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "titles": title,
        "format": "json"
    }
    data = make_api_request(params)
    pages = data.get("query", {}).get("pages", {})
    for p in pages.values():
        return p.get("extract", "")
    return ""


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("=" * 75, flush=True)
    print("🌐 Hindi Wikipedia Data Fetcher (MediaWiki API)", flush=True)
    print(f"Target : >= {TARGET_DEVANAGARI_CHARS:,d} Devanagari chars or {MAX_ARTICLES} articles", flush=True)
    print(f"Filter : Skip stubs (< {MIN_STUB_CHARS} chars), remove headings/citations, Latin > 30%", flush=True)
    print(f"Output : {OUTPUT_FILE}", flush=True)
    print("=" * 75, flush=True)

    accumulated_articles = []
    total_devanagari_chars = 0
    total_all_chars = 0
    articles_attempted = 0
    stubs_skipped = 0

    # Initialize / clear output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        pass

    try:
        while total_devanagari_chars < TARGET_DEVANAGARI_CHARS and len(accumulated_articles) < MAX_ARTICLES:
            titles = fetch_random_titles(batch_size=20)
            print(f"\n[INFO] Fetched batch of {len(titles)} candidate titles from Wikipedia API.", flush=True)

            for title in titles:
                articles_attempted += 1
                extract = fetch_article_extract(title)

                # Skip short stubs (< 1500 chars)
                if len(extract) < MIN_STUB_CHARS:
                    stubs_skipped += 1
                    print(f"  - Skip stub ({len(extract):4d} chars < {MIN_STUB_CHARS}): {title[:35]}", flush=True)
                    continue

                cleaned_article = clean_article_text(extract)
                dev_count = sum(1 for ch in cleaned_article if 0x0900 <= ord(ch) <= 0x097F)

                # If after cleaning too little Hindi remains, skip
                if dev_count < 300:
                    print(f"  - Skip low Hindi ({dev_count} dev chars): {title[:35]}", flush=True)
                    continue

                accumulated_articles.append(cleaned_article)
                total_devanagari_chars += dev_count
                total_all_chars += len(cleaned_article)

                # Incrementally append to file and flush to disk
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    if len(accumulated_articles) > 1:
                        f.write("\n\n")
                    f.write(cleaned_article)
                    f.flush()

                progress_pct = (total_devanagari_chars / TARGET_DEVANAGARI_CHARS) * 100.0
                print(
                    f"[{len(accumulated_articles):3d}/{MAX_ARTICLES}] "
                    f"Title: {title[:28]:28s} | "
                    f"+{dev_count:5,d} Hindi chars | "
                    f"Total: {total_devanagari_chars:7,d} / {TARGET_DEVANAGARI_CHARS:,d} ({progress_pct:5.1f}%)",
                    flush=True
                )

                if total_devanagari_chars >= TARGET_DEVANAGARI_CHARS or len(accumulated_articles) >= MAX_ARTICLES:
                    break

    except KeyboardInterrupt:
        print("\n[!] User interrupted fetch. Saving collected articles so far...", flush=True)
    except Exception as e:
        print(f"\n[FATAL ERROR] Fetching stopped due to network/API error: {e}", flush=True)
        sys.exit(1)

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        full_text = f.read()

    final_unique_chars = len(set(full_text))
    final_devanagari = sum(1 for ch in full_text if 0x0900 <= ord(ch) <= 0x097F)
    final_size_kb = os.path.getsize(OUTPUT_FILE) / 1024.0

    print("\n" + "=" * 75, flush=True)
    print("🎉 Hindi Wikipedia Fetch Completed Successfully!", flush=True)
    print("=" * 75, flush=True)
    print(f"Saved Output File    : {OUTPUT_FILE} ({final_size_kb:.2f} KB)", flush=True)
    print(f"Total Articles Saved : {len(accumulated_articles)}", flush=True)
    print(f"Total Characters     : {len(full_text):,d}", flush=True)
    print(f"Devanagari Characters: {final_devanagari:,d} (Target >= {TARGET_DEVANAGARI_CHARS:,d})", flush=True)
    print(f"Unique Characters    : {final_unique_chars}", flush=True)
    print(f"Stubs Skipped (<1500): {stubs_skipped} (from {articles_attempted} attempts)", flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
