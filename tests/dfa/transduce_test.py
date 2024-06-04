import unittest

from automata.fa.dfa import DFA

import evallm


class TestTransduce(unittest.TestCase):
    # add modulo 3 == 0
    dfa = DFA(
        states={"0", "1", "2"},
        input_symbols={"0", "1", "2"},
        transitions={
            "0": {"0": "0", "1": "1", "2": "2"},
            "1": {"0": "1", "1": "2", "2": "0"},
            "2": {"0": "2", "1": "0", "2": "1"},
        },
        initial_state="0",
        final_states={"0"},
    )

    def test_basic(self):
        self.assertEqual(
            evallm.transduce(self.dfa, "012212"),
            [1, 0, 1, 0, 1, 0],
        )
