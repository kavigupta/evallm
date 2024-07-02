import numpy as np
from automata.fa.dfa import DFA
from evallm.sample_dfa.naive_sample import naively_sample_dfa
from evallm.sample_dfa.reachable_sample import sample_reachable_dfa
from evallm.utils.construct import construct


def sample_dfa(sample_dfa_spec, rng: np.random.RandomState) -> DFA:
    return construct(
        dict(
            naively_sample_dfa=naively_sample_dfa,
            sample_reachable_dfa=sample_reachable_dfa,
        ),
        sample_dfa_spec,
        rng=rng,
    )
