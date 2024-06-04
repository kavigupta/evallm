from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class DFA:
    n_states: int
    n_symbols: int
    transitions: np.ndarray  # shape (n_states, n_symbols) contains next state
    start_state: int
    accept_states: np.ndarray  # shape (n_states,) dtype bool

    def __post_init__(self):
        assert self.transitions.dtype == np.int64
        assert self.accept_states.dtype == np.bool_
        assert self.transitions.shape == (self.n_states, self.n_symbols)
        assert self.accept_states.shape == (self.n_states,)

    def run(self, input_string: List[int]) -> List[bool]:
        """
        Run the DFA on the given input string.
        """

        state = self.start_state
        results = []
        for symbol in input_string:
            state = self.transitions[state, symbol]
            results.append(self.accept_states[state])
        return results


def dfa_from_strings(
    transitions: Dict[str, Dict[str, str]],
    start_state: str,
    accept_states: List[str],
    default_state: str = "0",
) -> DFA:
    """
    Create a DFA from a dictionary of transitions.

    Args:
        transitions: A dictionary of transitions. The keys are the current states,
            and the values are dictionaries where the keys are the symbols and the
            values are the next states.
        start_state: The start state.
        accept_states: A string of the accept states separated by commas.

    Returns:
        A DFA.
    """
    states = sorted(transitions.keys())
    symbols = sorted({symbol for state in transitions for symbol in transitions[state]})
    n_states = len(states)
    n_symbols = len(symbols)
    transitions_array = np.full((n_states, n_symbols), int(default_state))
    for i, state in enumerate(states):
        for j, symbol in enumerate(symbols):
            transitions_array[i, j] = int(
                transitions.get(state, {}).get(symbol, default_state)
            )
    start_state = states.index(start_state)
    accept_states = np.array([state in accept_states for state in states])
    return DFA(
        n_states=n_states,
        n_symbols=n_symbols,
        transitions=transitions_array,
        start_state=start_state,
        accept_states=accept_states,
    )
