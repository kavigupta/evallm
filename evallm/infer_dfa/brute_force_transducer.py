from permacache import permacache, stable_hash
import numpy as np
import tqdm.auto as tqdm
from evallm.experiments.sequence_completion.sequence_completion_brute_force import (
    all_dfas,
)
from evallm.sample_dfa.transduce import transduce


def consistent_with_dfa(dfa, i, o):
    # do not include the final state
    o = o[:-1]
    pred_o = transduce(dfa, i)
    pred_o, final_pred = pred_o[:-1], pred_o[-1]
    assert len(pred_o) == len(o)
    return (np.array(pred_o) == o).all(), final_pred


@permacache(
    "evallm/infer_dfa/consistent_dfa_predictions",
    key_function=dict(i=stable_hash, o=stable_hash),
)
def consistent_dfa_predictions(num_states, num_symbols, i, o):
    dfas = all_dfas(num_states=num_states, num_symbols=num_symbols)
    results = {0: 0, 1: 0}
    for dfa in dfas:
        consistent, final_pred = consistent_with_dfa(dfa, i, o)
        results[final_pred] += consistent
    return results


def brute_force_accuracy(num_states, num_symbols, inputs, outputs):
    results = []
    for i, o in zip(tqdm.tqdm(inputs), outputs):
        pred = consistent_dfa_predictions(num_states, num_symbols, i, o)
        pred = pred[1] > pred[0]
        results.append(pred == o[-1])
    return np.mean(results)