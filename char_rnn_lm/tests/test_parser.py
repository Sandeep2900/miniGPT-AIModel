"""
tests/test_parser.py
Unit tests for the strict MiniGPT tool call parser.
Ensures zero false positives and rejects any malformed or multi-call tool strings.
"""

import unittest
from tools.parser import parse_tool_call


class TestToolCallParser(unittest.TestCase):

    def test_valid_tool_call(self):
        """Case 1: Standard valid tool call returns parsed dictionary."""
        raw = '[TOOL: search("python news")]'
        result = parse_tool_call(raw)
        self.assertEqual(result, {"name": "search", "query": "python news"})

        # Leading / trailing whitespace on the entire string is safely stripped
        raw_with_newlines = '  \n[TOOL: search("latest cricket score")]\n  '
        result_ws = parse_tool_call(raw_with_newlines)
        self.assertEqual(result_ws, {"name": "search", "query": "latest cricket score"})

    def test_missing_quotes(self):
        """Case 2: Missing quotes around query must be rejected."""
        raw = '[TOOL: search(python news)]'
        self.assertIsNone(parse_tool_call(raw))

        raw_single_quote = "[TOOL: search('python news')]"
        self.assertIsNone(parse_tool_call(raw_single_quote))

    def test_extra_spaces_in_format(self):
        """Case 3: Extra internal spaces in syntax must be rejected."""
        # Extra space before colon
        self.assertIsNone(parse_tool_call('[TOOL : search("test")]'))
        # Extra space after colon
        self.assertIsNone(parse_tool_call('[TOOL:  search("test")]'))
        # Space between search and parenthesis
        self.assertIsNone(parse_tool_call('[TOOL: search ("test")]'))

    def test_empty_query(self):
        """Case 4: Empty or whitespace-only query must be rejected."""
        self.assertIsNone(parse_tool_call('[TOOL: search("")]'))
        self.assertIsNone(parse_tool_call('[TOOL: search("   ")]'))

    def test_two_tool_calls_in_one_output(self):
        """Case 5: Multiple tool calls in a single turn must be strictly rejected."""
        raw = '[TOOL: search("a")] [TOOL: search("b")]'
        self.assertIsNone(parse_tool_call(raw))

        raw_multiline = '[TOOL: search("first")]\n[TOOL: search("second")]'
        self.assertIsNone(parse_tool_call(raw_multiline))

    def test_plain_conversational_text(self):
        """Case 6: Plain natural language outputs must return None."""
        self.assertIsNone(parse_tool_call('Hello! How are you?'))
        self.assertIsNone(parse_tool_call('Main theek hoon, aap bataiye?'))
        self.assertIsNone(parse_tool_call('I cannot help with that.'))

    def test_surrounding_text_rejected(self):
        """Tool call embedded within regular sentences must be rejected (strict exact match)."""
        self.assertIsNone(parse_tool_call('Let me check that for you: [TOOL: search("weather")]'))
        self.assertIsNone(parse_tool_call('[TOOL: search("weather")] here are the details.'))

    def test_null_and_empty_inputs(self):
        """Empty, whitespace, or invalid types must safely return None without throwing."""
        self.assertIsNone(parse_tool_call(""))
        self.assertIsNone(parse_tool_call("   "))
        self.assertIsNone(parse_tool_call(None))


if __name__ == "__main__":
    unittest.main()
