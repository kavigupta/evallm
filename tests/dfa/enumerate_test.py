import unittest

import numpy as np
from parameterized import parameterized
import tqdm

from evallm.enumerate_dfa.enumerate import (
    all_io_permutations,
    all_state_permutations,
    enumerate_packed_dfas,
    enumerate_packed_dfas_no_permutations_valid,
)
from evallm.enumerate_dfa.pack_dfa import pack_dfa, unpack_dfa
from evallm.sample_dfa.naive_sample import naively_sample_dfa
from evallm.sample_dfa.transduce import transduce


class PackDfaTest(unittest.TestCase):

    def round_trip(self, dfa):
        unpacked = unpack_dfa(pack_dfa(dfa))
        self.assertEqual(dfa, unpacked)

    @parameterized.expand([(seed,) for seed in range(100)])
    def test_pack_random_dfa(self, seed):
        rng = np.random.RandomState(seed)
        dfa = naively_sample_dfa(rng.choice(10) + 1, rng.choice(10) + 1, rng)
        self.round_trip(dfa)


class PermutationsTest(unittest.TestCase):

    @parameterized.expand([(seed,) for seed in range(100)])
    def test_permutations(self, seed):
        rng = np.random.RandomState(seed)
        dfa = naively_sample_dfa(3, 3, rng)
        packed = pack_dfa(dfa)
        seq = "".join(rng.choice(list(dfa.input_symbols), size=100))
        for permuted in all_state_permutations(packed):
            self.assertEqual(
                transduce(unpack_dfa(packed), seq),
                transduce(unpack_dfa(permuted), seq),
            )


class FullEnumerationTest(unittest.TestCase):
    chunk_size = 1000
    enumerated = set(enumerate_packed_dfas(3, 3))
    enumerated_valid = set(enumerate_packed_dfas_no_permutations_valid(3, 3))
    enumerated_valid_list = list(enumerated_valid)

    @parameterized.expand([(seed,) for seed in range(100)])
    def test_pack_random_dfa(self, seed):
        rng = np.random.RandomState(seed)
        dfa = naively_sample_dfa(3, 3, rng)
        packed = pack_dfa(dfa)
        self.assertIn(packed, self.enumerated)

    @parameterized.expand(
        [(chunk,) for chunk in range(0, len(enumerated_valid_list), chunk_size)]
    )
    def test_symbol_permutations_present(self, chunk):
        for pdfa in tqdm.tqdm(
            self.enumerated_valid_list[chunk : chunk + self.chunk_size]
        ):
            for pdfa_sym in all_io_permutations(pdfa):
                assert pdfa_sym in self.enumerated_valid
