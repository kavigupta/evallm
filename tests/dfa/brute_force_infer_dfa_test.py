from fractions import Fraction
import unittest

import numpy as np

import evallm
import evallm.infer_dfa
import evallm.infer_dfa.inference


class TestBruteForceInfer(unittest.TestCase):

    def compute_consistent_universes(self, num_symbols, num_states, inp, out):
        sorted_symbols = sorted(set(inp))
        universes = evallm.infer_dfa.inference.consistent_universes(
            num_symbols, num_states, [sorted_symbols.index(i) for i in inp], out
        )
        result = [
            (
                Fraction.from_float(np.exp(logit)).limit_denominator(1000),
                dfa.json(sorted_symbols),
            )
            for logit, dfa in universes
        ]
        print(result)
        return result

    def test_basic(self):
        # trying to find the 1 symbol 3 state DFA that outputs 001001001001
        self.assertEqual(
            self.compute_consistent_universes(1, 3, "A" * 3, [False, False, True]),
            [
                (
                    Fraction(1, 9),
                    {
                        "current_state": 0,
                        "transitions": [{"A": 1}, {"A": 2}, {"A": 0}],
                        "accept_states": [1, 0, 0],
                    },
                ),
                (
                    Fraction(1, 9),
                    {
                        "current_state": 0,
                        "transitions": [{"A": 2}, {"A": 0}, {"A": 1}],
                        "accept_states": [1, 0, 0],
                    },
                ),
            ],
        )

    def test_basic_partial(self):
        # trying to find the 1 symbol 3 state DFA that outputs 001001001001
        self.assertEqual(
            self.compute_consistent_universes(1, 3, "A" * 2, [False, False]),
            [
                (
                    Fraction(1, 3),
                    {
                        "current_state": np.int8(0),
                        "transitions": [{"A": 0}, {"A": -1}, {"A": -1}],
                        "accept_states": [0, -1, -1],
                    },
                ),
                (
                    Fraction(1, 9),
                    {
                        "current_state": 0,
                        "transitions": [{"A": 1}, {"A": 0}, {"A": -1}],
                        "accept_states": [0, 0, -1],
                    },
                ),
                (
                    Fraction(1, 9),
                    {
                        "current_state": 1,
                        "transitions": [{"A": 1}, {"A": 1}, {"A": -1}],
                        "accept_states": [-1, 0, -1],
                    },
                ),
                (
                    Fraction(1, 9),
                    {
                        "current_state": 2,
                        "transitions": [{"A": 1}, {"A": 2}, {"A": -1}],
                        "accept_states": [-1, 0, 0],
                    },
                ),
                (
                    Fraction(1, 9),
                    {
                        "current_state": 0,
                        "transitions": [{"A": 2}, {"A": -1}, {"A": 0}],
                        "accept_states": [0, -1, 0],
                    },
                ),
                (
                    Fraction(1, 9),
                    {
                        "current_state": 1,
                        "transitions": [{"A": 2}, {"A": -1}, {"A": 1}],
                        "accept_states": [-1, 0, 0],
                    },
                ),
                (
                    Fraction(1, 9),
                    {
                        "current_state": 2,
                        "transitions": [{"A": 2}, {"A": -1}, {"A": 2}],
                        "accept_states": [-1, -1, 0],
                    },
                ),
            ],
        )

    def test_addition_dfa(self):
        # trying to find the 3 symbol 3 state DFA that outputs the sum of the input mod 3
        sequence = np.random.RandomState(0).randint(0, 3, 100).tolist()
        out = [sum(sequence[:i]) % 3 == 0 for i in range(1, 101)]
        universes = self.compute_consistent_universes(3, 3, sequence, out)
        self.assertEqual(
            universes,
            [
                (
                    Fraction(1, 192),
                    {
                        "current_state": np.int8(1),
                        "transitions": [
                            {0: 0, 1: 1, 2: 2},
                            {0: 1, 1: 2, 2: 0},
                            {0: 2, 1: 0, 2: 1},
                        ],
                        "accept_states": [1, 0, 0],
                    },
                ),
                (
                    Fraction(1, 192),
                    {
                        "current_state": np.int8(2),
                        "transitions": [
                            {0: 0, 1: 2, 2: 1},
                            {0: 1, 1: 0, 2: 2},
                            {0: 2, 1: 1, 2: 0},
                        ],
                        "accept_states": [1, 0, 0],
                    },
                ),
            ],
        )
