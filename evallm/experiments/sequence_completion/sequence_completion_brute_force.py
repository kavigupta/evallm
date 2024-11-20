from functools import lru_cache
import itertools

import numpy as np
from permacache import permacache, stable_hash
from evallm.enumerate_dfa.enumerate import enumerate_packed_dfas_no_permutations
from evallm.enumerate_dfa.pack_dfa import unpack_dfa


@lru_cache(maxsize=None)
def all_dfas(num_states, num_symbols):
    return [
        unpack_dfa(dfa)
        for dfa in enumerate_packed_dfas_no_permutations(num_states, num_symbols)
    ]


@lru_cache(maxsize=None)
def all_completions(symbols, num_sequence_symbols):
    return list(
        list(x)
        for x in itertools.product(
            symbols,
            repeat=num_sequence_symbols,
        )
    )


@permacache(
    "evallm/experiments/sequence_completion/sequence_completion_brute_force",
    key_function=dict(
        dfa_packed=stable_hash,
        sequences=stable_hash,
    ),
)
def sequence_completion_brute_force(dfa_packed, sequences, prefix):
    dfa = unpack_dfa(dfa_packed)
    relevant_dfas = all_dfas(len(dfa.states), len(dfa.input_symbols))
    completions = all_completions(
        tuple(sorted(dfa.input_symbols)), len(sequences[0]) - len(prefix)
    )
    for sequence in sequences:
        relevant_dfas = [d for d in relevant_dfas if d.accepts_input(sequence)]
    by_completion = [
        np.mean([d.accepts_input(prefix + c) for d in relevant_dfas])
        for c in completions
    ]
    best_completion = np.argmax(by_completion)
    return completions[best_completion], by_completion[best_completion]
