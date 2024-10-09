from dataclasses import dataclass

import numpy as np


@dataclass
class PartialDFA:
    current_state: int
    transitions: np.ndarray
    accept_states: np.ndarray

    @classmethod
    def empty(cls, n_states: int, n_symbols: int):
        return cls(
            current_state=0,
            transitions=np.zeros((n_states, n_symbols), dtype=np.int8) - 1,
            accept_states=np.zeros(n_states, dtype=np.int8) - 1,
        )

    def copy(self):
        return PartialDFA(
            current_state=self.current_state,
            transitions=self.transitions.copy(),
            accept_states=self.accept_states.copy(),
        )

    def output_for_step(self, symbol: int) -> bool:
        new_state = self.transitions[self.current_state, symbol]
        if new_state == -1:
            return -1, -1
        return new_state, self.accept_states[new_state]

    def consistent_with_step(self, symbol, accept):
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
