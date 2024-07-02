import numpy as np
from automata.fa.dfa import DFA

from evallm.sample_dfa.naive_sample import naively_sample_dfa


def sample_reachable_dfa(
    n_states: int, n_symbols: int, rng: np.random.RandomState
) -> DFA:
    while True:
        dfa = naively_sample_dfa(n_states, n_symbols, rng)
        if len(reachable_states(dfa)) == n_states:
            return dfa


def reachable_states(dfa):
    seen = set()
    queue = [dfa.initial_state]
    while queue:
        first = queue.pop()
        if first in seen:
            continue
        seen.add(first)
        queue += dfa.transitions[first].values()
    return seen
