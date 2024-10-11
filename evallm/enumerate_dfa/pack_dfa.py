from typing import Tuple

from automata.fa.dfa import DFA

PackedDFA = Tuple[int, Tuple[Tuple[int, ...], ...], Tuple[int, ...]]


def pack_dfa(dfa: DFA) -> PackedDFA:
    """
    Returns a tuple (transitions, accept_states) representing the DFA.
    """
    symbols = sorted(dfa.input_symbols)
    states = sorted(dfa.states)

    transitions = tuple(
        tuple(states.index(dfa.transitions[state][symbol]) for symbol in symbols)
        for state in states
    )
    accept_states = tuple(1 if state in dfa.final_states else 0 for state in states)

    initial_state = states.index(dfa.initial_state)
    return initial_state, transitions, accept_states


def unpack_dfa(packed_dfa: PackedDFA) -> DFA:
    """
    Returns a DFA from the tuple (transitions, accept_states).
    """
    start_state, transitions, accept_states = packed_dfa

    n_states = len(transitions)
    n_symbols = len(transitions[0])

    symbols = tuple(chr(ord("a") + i) for i in range(n_symbols))
    states = tuple(str(i) for i in range(n_states))

    dfa = DFA(
        states=set(states),
        input_symbols=set(symbols),
        transitions={
            states[i]: {symbols[j]: states[transitions[i][j]] for j in range(n_symbols)}
            for i in range(n_states)
        },
        initial_state=states[start_state],
        final_states={states[i] for i in range(n_states) if accept_states[i]},
    )

    return dfa
