import string

import numpy as np
from automata.fa.dfa import DFA


def naively_sample_dfa(
    n_states: int, n_symbols: int, rng: np.random.RandomState
) -> DFA:
    """
    Sample a DFA with the given number of states and symbols. The DFA
        distribution is not guaranteed to be uniform due to isomorphisms,
        and not all states are guaranteed to be reachable.

    Args:
        n_states: The number of states in the DFA.
        n_symbols: The number of symbols in the DFA.
        rng: The random number generator to use.

    Returns:
        A DFA with the given number of states and symbols.
    """
    transitions = rng.randint(0, n_states, size=(n_states, n_symbols))
    start_state = rng.randint(0, n_states)
    accept_states = rng.choice([True, False], size=n_states)

    symbols = string.ascii_letters[:n_symbols]
    states = [str(i) for i in range(n_states)]
    assert len(symbols) == n_symbols

    return DFA(
        states=set(states),
        input_symbols=set(symbols),
        transitions={
            states[i]: {symbols[j]: states[transitions[i, j]] for j in range(n_symbols)}
            for i in range(n_states)
        },
        initial_state=states[start_state],
        final_states={states[i] for i in range(n_states) if accept_states[i]},
    )
