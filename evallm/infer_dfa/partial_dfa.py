from dataclasses import dataclass

import numpy as np


@dataclass
class PartialDFA:
    """
    Represents a partial DFA, which is a DFA that may not have all transitions
    defined. The transitions that are not defined are represented by -1.
    """

    current_state: int
    transitions: np.ndarray
    accept_states: np.ndarray

    @classmethod
    def empty(cls, n_states: int, n_symbols: int):
        """
        Create an empty partial DFA with n_states states and n_symbols symbols.
        By default, the current state starts at 0.
        """
        return cls(
            current_state=0,
            transitions=np.zeros((n_states, n_symbols), dtype=np.int8) - 1,
            accept_states=np.zeros(n_states, dtype=np.int8) - 1,
        )

    def copy(self):
        """
        Copies this partial DFA.
        """
        return PartialDFA(
            current_state=self.current_state,
            transitions=self.transitions.copy(),
            accept_states=self.accept_states.copy(),
        )

    def output_for_step(self, symbol: int) -> bool:
        """
        Computes the output of the partial DFA for a given symbol.

        :param symbol: The symbol to be processed.
        :return: A tuple with the new state and the output of the new state.
            If the transition is not defined, returns (-1, -1).
        """
        new_state = self.transitions[self.current_state, symbol]
        if new_state == -1:
            return -1, -1
        return new_state, self.accept_states[new_state]

    def consistent_with_step(self, symbol, accept):
        """
        Return a list of partial DFAs that are consistent with the transition of
        the current DFA for the given symbol and accept value.
        """
        new_state, new_accept = self.output_for_step(symbol)
        if new_state != -1:
            if new_accept == accept:
                copy_self = self.copy()
                copy_self.current_state = new_state
                return [copy_self]
            return []
        results = []
        for new_state in range(self.transitions.shape[0]):
            if (
                self.accept_states[new_state] != -1
                and self.accept_states[new_state] != accept
            ):
                continue
            copy_self = self.copy()
            copy_self.transitions[self.current_state, symbol] = new_state
            copy_self.current_state = new_state
            copy_self.accept_states[new_state] = accept
            results.append(copy_self)
        return results

    def json(self, symbols=None):
        """
        Returns a JSON representation of the partial DFA.
        """
        if symbols is None:
            symbols = list(range(self.transitions.shape[1]))
        transitions = [
            {sym: int(self.transitions[i, j]) for j, sym in enumerate(symbols)}
            for i in range(self.transitions.shape[0])
        ]
        return {
            "current_state": self.current_state,
            "transitions": transitions,
            "accept_states": self.accept_states.tolist(),
        }
