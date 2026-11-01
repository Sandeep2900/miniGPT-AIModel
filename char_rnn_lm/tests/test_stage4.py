"""
tests/test_stage4.py
Unit tests for Stage 4 Compact Scraper:
- SSRF prevention (localhost, private subnets, non-http schemes)
- Snippet decision logic (< 40 chars or missing)
- Max 200-char truncation
"""

import unittest
from tools.scrape import is_safe_url, should_scrape_snippet, scrape_page


class TestStage4Scraper(unittest.TestCase):

    def test_ssrf_blocks_localhost(self):
        """Localhost and loopback IPs must be blocked."""
        urls = [
            "http://localhost:8080/admin",
            "http://127.0.0.1:5000",
            "http://127.0.0.1",
            "https://127.0.0.1:3000/api",
        ]
        for url in urls:
            safe, err = is_safe_url(url)
            self.assertFalse(safe, f"Failed to block localhost: {url}")
            self.assertIn("SSRF Blocked", err)

    def test_ssrf_blocks_private_networks(self):
        """Private IP subnets (10.x, 192.168.x, 172.16.x) must be blocked."""
        private_urls = [
            "http://192.168.1.1/router",
            "http://10.0.0.5:8080",
            "http://172.16.0.1/status",
        ]
        for url in private_urls:
            safe, err = is_safe_url(url)
            self.assertFalse(safe, f"Failed to block private subnet: {url}")
            self.assertIn("SSRF Blocked", err)

    def test_ssrf_blocks_unsupported_schemes(self):
        """Non-http/https schemes must be blocked."""
        schemes = [
            "file:///C:/Windows/win.ini",
            "ftp://ftp.example.com",
            "gopher://example.com",
        ]
        for url in schemes:
            safe, err = is_safe_url(url)
            self.assertFalse(safe, f"Failed to block invalid scheme: {url}")
            self.assertIn("Unsupported scheme", err)

    def test_safe_public_url_allowed(self):
        """Legitimate public domains must be approved."""
        safe, err = is_safe_url("https://en.wikipedia.org/wiki/Python")
        self.assertTrue(safe)
        self.assertIsNone(err)

    def test_should_scrape_snippet_rule(self):
        """Constraint 4: Scrapes ONLY when snippet is missing or < 40 characters."""
        # Missing
        self.assertTrue(should_scrape_snippet(None))
        self.assertTrue(should_scrape_snippet(""))
        self.assertTrue(should_scrape_snippet("   "))

        # Under 40 chars
        short_snippet = "Python is a language."  # 21 chars
        self.assertTrue(should_scrape_snippet(short_snippet))

        # Adequate snippet (>= 40 chars)
        good_snippet = "Python is a high-level, general-purpose programming language with dynamic semantics."  # 84 chars
        self.assertFalse(should_scrape_snippet(good_snippet))

    def test_scrape_localhost_rejected_without_network_call(self):
        """Calling scrape_page on localhost must fail safely with SSRF message."""
        text, err = scrape_page("http://127.0.0.1:9999/secret")
        self.assertEqual(text, "")
        self.assertIn("SSRF Blocked", err)

    def test_scrape_live_truncation(self):
        """Scraped output must never exceed max_chars (200 characters)."""
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        text, err = scrape_page(url, max_chars=200)
        if err is None:
            self.assertLessEqual(len(text), 200)
            self.assertGreater(len(text), 10)
        else:
            self.assertIsInstance(err, str)


if __name__ == "__main__":
    unittest.main()
