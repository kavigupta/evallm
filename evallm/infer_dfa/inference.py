import numpy as np
from scipy.special import logsumexp

from .partial_dfa import PartialDFA


def consistent_universes(num_symbols, num_states, inp, out):
    """
    Find all consistent DFAs for a given input/output sequence.

    :param num_symbols: The number of symbols in the alphabet.
    :param num_states: The number of states in the DFA.
    :param inp: The input sequence as a list of integers in the range [0, num_symbols).
    :param out: The output sequence as a list of booleans.

    :return: A list of tuples (logit, dfa) where logit is the log probability of the DFA
        and dfa is a PartialDFA object.
    """
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
    """
    Probability of 1 being outputted by the DFA for the given input/output sequence.

    :param num_states: The number of states in the DFA.
    :param inp: The input sequence as a string / list of strings
    :param out: The output sequence as a list of booleans.
    """
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
