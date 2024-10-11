import itertools
import string
from typing import Tuple

from automata.fa.dfa import DFA
from permacache import permacache
import tqdm

from evallm.sample_dfa.reachable_sample import reachable_states
from evallm.sample_dfa.transduce import transduce

from .pack_dfa import PackedDFA, unpack_dfa


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


@permacache("evallm/enumerate_dfa/enumerate_packed_dfas_no_permutations_valid_2")
def enumerate_packed_dfas_no_permutations_valid(
    num_states: int, num_symbols: int
) -> Tuple[PackedDFA]:
    """
    Enumerate all possible DFAs with the given number of states and symbols.

    Does not include any two DFAs that
        - are permutations of each other
        - have unreachable states
        - are non-minimal
        - have all states accept or all states reject
    """
    result = []
    for pdfa in enumerate_packed_dfas_no_permutations(num_states, num_symbols):
        dfa = unpack_dfa(pdfa)
        if all(pdfa[2]):
            continue
        if not any(pdfa[2]):
            continue
        if reachable_states(dfa) != {str(s) for s in range(num_states)}:
            continue
        # minimality
        if len(dfa.minify().states) != num_states:
            continue
        result.append(pdfa)
    return result


@permacache("evallm/enumerate_dfa/enumerate_packed_dfas_distinguishable")
def enumerate_packed_dfas_distinguishable(num_states: int, num_symbols: int):
    """
    Like enumerate_packed_dfas_no_permutations_valid, but guarantees that all DFAs
    are distinguishable.
    """

    seqs = list(
        itertools.product(string.ascii_lowercase[:num_symbols], repeat=num_states + 2)
    )
    pdfas = enumerate_packed_dfas_no_permutations_valid(num_states, num_symbols)
    out_to_pdfa = {}
    for pdfa in tqdm.tqdm(pdfas):
        dfa = unpack_dfa(pdfa)
        out = tuple(tuple(transduce(dfa, seq)) for seq in seqs)
        if out not in out_to_pdfa:
            out_to_pdfa[out] = pdfa
            continue
        diff = dfa.symmetric_difference(unpack_dfa(out_to_pdfa[out]))
        diff = diff.difference(DFA.from_finite_language(diff.input_symbols, [""]))
        assert diff.isempty(), str(dfa, unpack_dfa(out_to_pdfa[out]), diff)
        out_to_pdfa[out] = min(out_to_pdfa[out], pdfa)
    return sorted(out_to_pdfa.values())


def all_io_permutations(pdfa: PackedDFA) -> Tuple[PackedDFA]:
    """
    Enumerate all the DFAs that are equivalent to the given DFA under input/output
    permutations.
    """
    permutations = set()
    for perm in itertools.permutations(range(len(pdfa[1][0]))):
        for flip_output in True, False:
            transitions = tuple(
                tuple(pdfa[1][i][perm_idx] for perm_idx in perm)
                for i in range(len(pdfa[1]))
            )
            out = pdfa[2] if not flip_output else tuple(not i for i in pdfa[2])
            pdfa_perm = (pdfa[0], transitions, out)
            permutations.add(min(all_state_permutations(pdfa_perm)))
    return sorted(permutations)


@permacache(
    "evallm/enumerate_dfa/enumerate_packed_dfas_no_permutations_valid_no_io_permutations"
)
def enumerate_packed_dfas_no_permutations_valid_no_io_permutations(
    num_states: int, num_symbols: int
) -> Tuple[PackedDFA]:
    """
    Enumerate all possible DFAs with the given number of states and symbols. Does not
    include any two DFAs that are permutations of each other or that are equivalent
    under input/output permutations.
    """
    return [
        pdfa
        for pdfa in enumerate_packed_dfas_no_permutations_valid(num_states, num_symbols)
        if all_io_permutations(pdfa)[0] == pdfa
    ]
