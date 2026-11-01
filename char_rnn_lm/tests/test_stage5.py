"""
tests/test_stage5.py
Unit tests for Stage 5 Orchestrator Agent:
- Keyword routing classification
- Token budget trimming
- Non-crashing fallback logic
"""

import unittest
from agent import is_time_query, is_search_query, trim_to_tokens
from tokenizers import Tokenizer


class TestStage5Orchestrator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file("data/bpe_tokenizer.json")

    def test_routing_time_queries(self):
        """Time and date queries must route to clock."""
        self.assertTrue(is_time_query("What time is it?"))
        self.assertTrue(is_time_query("Aaj kya date hai?"))
        self.assertTrue(is_time_query("Kitna samay hua"))
        self.assertTrue(is_time_query("Current time please"))
        self.assertFalse(is_time_query("Hello how are you"))

    def test_routing_search_queries(self):
        """Keywords like latest, news, price, etc. must route to search."""
        self.assertTrue(is_search_query("Latest iPhone price"))
        self.assertTrue(is_search_query("Aaj ka weather kaisa hai"))
        self.assertTrue(is_search_query("World cup match score"))
        self.assertTrue(is_search_query("Python release news"))
        self.assertFalse(is_search_query("Who wrote Hamlet?"))

    def test_token_trimming(self):
        """trim_to_tokens must strictly keep token count <= limit."""
        long_text = "This is a very long text designed to test token trimming under strict budgets " * 10
        trimmed = trim_to_tokens(long_text, max_tokens=15, tokenizer=self.tokenizer)
        tokens = self.tokenizer.encode(trimmed).ids
        self.assertLessEqual(len(tokens), 15)

    def test_routing_stage1_cases(self):
        """Tests the 6 specific Stage 1 route cases for is_time_query and is_search_query."""
        # 1. "Aaj mera din thoda mushkil tha." -> Neither clock nor search (model_direct)
        self.assertFalse(is_time_query("Aaj mera din thoda mushkil tha."))
        self.assertFalse(is_search_query("Aaj mera din thoda mushkil tha."))

        # 2. "Main thoda break lena chahta hoon." -> Neither clock nor search (model_direct)
        self.assertFalse(is_time_query("Main thoda break lena chahta hoon."))
        self.assertFalse(is_search_query("Main thoda break lena chahta hoon."))

        # 3. "Aaj ka weather kya hai?" -> search_rule
        self.assertFalse(is_time_query("Aaj ka weather kya hai?"))
        self.assertTrue(is_search_query("Aaj ka weather kya hai?"))

        # 4. "Latest Python release news kya hai?" -> search_rule
        self.assertFalse(is_time_query("Latest Python release news kya hai?"))
        self.assertTrue(is_search_query("Latest Python release news kya hai?"))

        # 5. "Abhi time kya hua hai?" -> clock
        self.assertTrue(is_time_query("Abhi time kya hua hai?"))
        self.assertFalse(is_search_query("Abhi time kya hua hai?"))

        # 6. "Hi, kaise ho?" -> Neither clock nor search (model_direct)
        self.assertFalse(is_time_query("Hi, kaise ho?"))
        self.assertFalse(is_search_query("Hi, kaise ho?"))


if __name__ == "__main__":
    unittest.main()
