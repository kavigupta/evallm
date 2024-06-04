import unittest

import evallm


class TestRecognition(unittest.TestCase):
    # add modulo 3 == 0
    dfa = evallm.dfa_from_strings(
        {
            "0": {"0": "0", "1": "1", "2": "2"},
            "1": {"0": "1", "1": "2", "2": "0"},
            "2": {"0": "2", "1": "0", "2": "1"},
        },
        "0",
        {"0"},
    )

    def test_basic(self):
        self.assertEqual(self.dfa.run([0, 1, 2, 2, 1, 2]), [1, 0, 1, 0, 1, 0])
