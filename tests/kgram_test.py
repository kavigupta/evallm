import unittest

import evallm


class TestLongestKgram(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(
            evallm.utils.longest_terminal_repeated_kgram("abcdefgabc"),
            (3, "abc", [0]),
        )

    def test_no_repeats(self):
        self.assertEqual(
            evallm.utils.longest_terminal_repeated_kgram("abcdefg"), (1, "g", [])
        )

    def test_with_repeated_character(self):
        self.assertEqual(
            evallm.utils.longest_terminal_repeated_kgram("abcaaaa"), (3, "aaa", [3])
        )

    def test_repeated_thrice(self):
        self.assertEqual(
            evallm.utils.longest_terminal_repeated_kgram("abcdeabcfgabc"),
            (3, "abc", [0, 5]),
        )
