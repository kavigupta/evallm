import itertools
from typing import Tuple

from permacache import permacache

from .pack_dfa import PackedDFA


def enumerate_packed_dfas(num_states: int, num_symbols: int) -> PackedDFA:
    """
    Enumerate all possible DFAs with the given number of states and symbols.
    """
    initial_states = range(num_states)
    transitions = itertools.product(
        list(itertools.product(range(num_states), repeat=num_symbols)),
        repeat=num_states,
    )
    accept_states = itertools.product(range(2), repeat=num_states)
    return itertools.product(initial_states, transitions, accept_states)


def all_state_permutations(pdfa: PackedDFA) -> Tuple[PackedDFA]:
    """
    Enumerate all possible permutations of states.
    """
    permutations = []
    for perm in itertools.permutations(range(len(pdfa[1]))):
        perm_back = {perm[i]: i for i in range(len(perm))}
        transitions = tuple(tuple(perm_back[j] for j in pdfa[1][i]) for i in perm)
        accept_states = tuple(pdfa[2][i] for i in perm)
        permutations.append((perm_back[pdfa[0]], transitions, accept_states))
    return tuple(permutations)


@permacache("evallm/enumerate_dfa/enumerate_packed_dfas_no_permutations")
def enumerate_packed_dfas_no_permutations(
    num_states: int, num_symbols: int
) -> Tuple[PackedDFA]:
    """
    Enumerate all possible DFAs with the given number of states and symbols. Does not
    include any two DFAs that are permutations of each other.
    """
    return [
        pdfa
        for pdfa in enumerate_packed_dfas(num_states, num_symbols)
        if sorted(all_state_permutations(pdfa))[0] == pdfa
    ]
