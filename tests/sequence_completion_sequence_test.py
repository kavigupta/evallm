import unittest

import evallm
from evallm.experiments.sequence_completion.ngram_suffix_heuristic import (
    multiple_ngrams,
    ngram_completions,
)


class TestNgramSequenceCompletion(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(
            ngram_completions(
                ["abcdefabcghiabcjkl", "abacabcd", "defabcdefaaa", "abcjkl"],
                "abc",
                5,
            ),
            ["jkl", "d", "jkl"],
        )

    def test_at_end(self):
        self.assertEqual(
            ngram_completions(
                ["abcabc", "defabc"],
                "abc",
                5,
            ),
            ["abc"],
        )

    def test_multiple_valid_in_one_sequence(self):
        self.assertEqual(
            ngram_completions(
                ["abcdefabcghiabcjkl", "abacabcd", "defabcdefaaa", "abcjkl"],
                "abc",
                100,
            ),
            ["defabcghiabcjkl", "ghiabcjkl", "jkl", "d", "defaaa", "jkl"],
        )

    def test_duplicate_character(self):
        self.assertEqual(
            ngram_completions(
                ["aaaabbaa", "aabaaa", "abaaaa", "accccccaacccc"],
                "aa",
                100,
            ),
            ["aabbaa", "abbaa", "bbaa", "baaa", "a", "aa", "a", "cccc"],
        )

    def test_ngrams(self):
        self.assertEqual(
            multiple_ngrams(
                ["abcdX"] + ["0bcdY"] * 2 + ["00cdZ"] * 3 + ["000dX"] * 5,
                "abcd",
            ),
            ["X", "Z", "Y", "X"],
        )
