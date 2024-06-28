import unittest

import evallm


class TestLongestKgram(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(
            evallm.utils.longest_terminal_repeated_kgrams("abcdefgabc"),
            [(1, "c", [2]), (2, "bc", [1]), (3, "abc", [0])],
        )

    def test_no_repeats(self):
        self.assertEqual(
            evallm.utils.longest_terminal_repeated_kgrams("abcdefg"), [(1, "g", [])]
        )

    def test_with_repeated_character(self):
        self.assertEqual(
            evallm.utils.longest_terminal_repeated_kgrams("abcaaaa"),
            [(1, "a", [0, 3, 4, 5]), (2, "aa", [3, 4]), (3, "aaa", [3])],
        )

    def test_repeated_thrice(self):
        self.assertEqual(
            evallm.utils.longest_terminal_repeated_kgrams("abcdeabcfgabc"),
            [(1, "c", [2, 7]), (2, "bc", [1, 6]), (3, "abc", [0, 5])],
        )

    def test_multiple_prefixes(self):
        self.assertEqual(
            evallm.utils.longest_terminal_repeated_kgrams(
                "bcd ef def bcdef cdef abcdef"
            ),
            [
                (1, "f", [5, 9, 15, 20]),
                (2, "ef", [4, 8, 14, 19]),
                (3, "def", [7, 13, 18]),
                (4, "cdef", [12, 17]),
                (5, "bcdef", [11]),
            ],
        )

    def test_with_heterogenous_values(self):
        self.assertEqual(
            evallm.utils.longest_terminal_repeated_kgrams(
                ["c", 0, "b", 0, "c", 0, "b", 0, "c", 1, "b", 0, "c"]
            ),
            [
                (1, ["c"], [0, 4, 8]),
                (2, [0, "c"], [3, 7]),
                (3, ["b", 0, "c"], [2, 6]),
            ],
        )


class TestPredictBasedOnKgram(unittest.TestCase):
    def test_basic(self):
        # kgram b0c -> 0, even though c -> 1
        self.assertEqual(
            evallm.utils.predict_from_sequence_based_on_kgram(
                ["b", 0, "c", 0, "c", 1, "b", 0, "c"]
            ),
            0,
        )

    def test_tie(self):
        # kgram b0c -> 0 then b0c -> 1, so tie
        self.assertEqual(
            evallm.utils.predict_from_sequence_based_on_kgram(
                ["x", 0, "b", 0, "c", 0, "y", 0, "b", 0, "c", 1, "z", 0, "b", 0, "c"]
            ),
            1,
        )
        # kgram b0c -> 1 then b0c -> 0, so tie
        self.assertEqual(
            evallm.utils.predict_from_sequence_based_on_kgram(
                ["x", 0, "b", 0, "c", 1, "y", 0, "b", 0, "c", 0, "z", 0, "b", 0, "c"]
            ),
            0,
        )
