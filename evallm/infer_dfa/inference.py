import numpy as np
from scipy.special import logsumexp

from .partial_dfa import PartialDFA


def consistent_universes(num_symbols, num_states, inp, out):
    universes = [(0, PartialDFA.empty(num_states, num_symbols))]
    for i, o in zip(inp, out):
        new_universes = []
        for u_logit, u in universes:
            new_us = u.consistent_with_step(i, o)
            new_universes += [
                (u_logit - np.log(len(new_us)), new_u) for new_u in new_us
            ]
        universes = new_universes
    assert universes, "No consistent DFA found"
    return universes


def prob_1(num_states, inp, out):
    symbols = sorted(set(inp))
    inp = [symbols.index(i) for i in inp]
    assert len(inp) == len(out) + 1
    universes = consistent_universes(len(symbols), num_states, inp[:-1], out)
    logits = [[], []]
    for logit, u in universes:
        _, out = u.output_for_step(inp[-1])
        if out == -1:
            continue
        logits[out].append(logit)
    prob_each = [logsumexp(logit) for logit in logits]
    prob = np.exp(prob_each[1] - np.logaddexp(prob_each[0], prob_each[1]))
    return prob
