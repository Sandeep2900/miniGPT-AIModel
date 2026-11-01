"""
tests/test_stage3.py
Unit and integration tests for Stage 3 (Clock & Search Tools).
"""

import re
import unittest
from tools.clock import get_current_time
from tools.search import web_search


class TestStage3Tools(unittest.TestCase):

    def test_clock_format(self):
        """Clock must return valid YYYY-MM-DD HH:MM string."""
        now_str = get_current_time()
        self.assertIsInstance(now_str, str)
        # Regex check: 4 digits - 2 digits - 2 digits space 2 digits : 2 digits
        pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"
        self.assertTrue(re.match(pattern, now_str), f"Time format mismatch: {now_str}")

    def test_search_empty_query(self):
        """Empty query must return empty list and error message without crashing."""
        res, err = web_search("")
        self.assertEqual(res, [])
        self.assertIsNotNone(err)

        res_ws, err_ws = web_search("   ")
        self.assertEqual(res_ws, [])
        self.assertIsNotNone(err_ws)

    def test_search_live_query(self):
        """Live search query must return at most 2 structured results."""
        results, err = web_search("Python programming language", max_results=2)
        # Live network call: if internet is connected, it should return <= 2 items
        if err is None:
            self.assertLessEqual(len(results), 2)
            if len(results) > 0:
                first = results[0]
                self.assertIn("title", first)
                self.assertIn("snippet", first)
                self.assertIn("url", first)
                self.assertTrue(first["url"].startswith("http"))
        else:
            # If network error, must be safe string without crashing
            self.assertIsInstance(err, str)


if __name__ == "__main__":
    unittest.main()
