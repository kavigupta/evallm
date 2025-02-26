import numpy as np

from evallm.experiments.main_tables import arg_best_prompt, sanitize_names
from evallm.experiments.transducer_summary import transducer_results
from evallm.infer_dfa.brute_force_transducer import brute_force_on_instances
from evallm.utils.kgrams import predict_based_on_kgram


def transducer_correctness_by_individual(data):
    best_prompts_keys = arg_best_prompt(transducer_results())
    x = transducer_results(accuracy_summary=False)
    x = sanitize_names({k: x[k][best_prompts_keys[k]] for k in x})
    x = {k: np.array(v) for k, v in x.items()}
    x["\\textsc{BruteForce}"] = correct_brute_force_on_all(
        data, correctness_kind="correct"
    ).astype(np.float64)
    for ngram in range(2, 1 + 6):
        x[rf"{ngram}-\textsc{{Gram}}"] = correct_ngram_prediction_on_all(
            data, ngram
        ).astype(np.float64)
    return x


def correct_brute_force_on_all(data, *, correctness_kind):
    justified_correct_mask = []
    for datum in data:
        bfs = brute_force_on_instances(
            num_states=3, num_symbols=3, inputs=datum.inputs, outputs=datum.outputs
        )
        justified_correct_mask.append([])
        for o, bf in zip(datum.outputs, bfs):
            o = o[-1]
            assert bf[o]
            if correctness_kind == "justified":
                justified_correct_mask[-1].append(bf[1 - o] == 0)
            elif correctness_kind == "correct":
                justified_correct_mask[-1].append((bf[1] >= bf[0]) == (o == 1))
            else:
                raise ValueError(f"Unknown correctness kind: {correctness_kind}")
    justified_correct_mask = np.array(justified_correct_mask)
    return justified_correct_mask


def confident_ngram_prediction_on_instance(i, o, ngram):
    prediction = predict_based_on_kgram(i, o, uncertainty=True)
    # if prediction[ngram - 2] is not present, then there's no suffix of that length
    if ngram - 2 >= len(prediction) - 1:
        return False
    # check if the prediction is correct and confident
    return prediction[ngram - 2] == float(o[-1])


def justified_ngram_prediction_on_instance(dfa, i, o, ngram):
    sequence = [x for xs in zip(i, o) for x in xs][-ngram:-1]
    state_set = set(dfa.states)
    for element in sequence:
        if isinstance(element, str):
            state_set = {dfa.transitions[s][element] for s in state_set}
        else:
            assert isinstance(element, bool)
            state_set = {s for s in state_set if (s in dfa.final_states) == element}
    possibilities = sorted({s in dfa.final_states for s in state_set})
    if len(possibilities) > 1:
        return False
    [answer] = possibilities
    assert answer == o[-1]
    return confident_ngram_prediction_on_instance(i, o, ngram)


def correct_ngram_prediction_on_all(data, ngram):
    ngram_mask = []
    for datum in data:
        ngram_mask.append([])
        for i, o in zip(datum.inputs, datum.outputs):
            res = predict_based_on_kgram(i, o)
            res = res[ngram - 2] if ngram - 2 < len(res) else res[-1]
            ngram_mask[-1].append(res == o[-1])
    ngram_mask = np.array(ngram_mask)
    return ngram_mask


def justified_ngram_prediction(data, ngram):
    justified_ngram_mask = []
    for datum in data:
        justified_ngram_mask.append([])
        for i, o in zip(datum.inputs, datum.outputs):
            justified_ngram_mask[-1].append(
                justified_ngram_prediction_on_instance(datum.dfa, i, o, ngram)
            )
    justified_ngram_mask = np.array(justified_ngram_mask)
    return justified_ngram_mask


def difficulty_bin_masks_by_individual_accuracy(data):
    masks_each = [
        justified_ngram_prediction(data, ngram) for ngram in range(2, 1 + 5)
    ] + [correct_brute_force_on_all(data, correctness_kind="justified")]
    incremental_masks = []
    not_solved_by_smaller = np.ones_like(masks_each[0], bool)
    for m in masks_each:
        incremental_masks.append(m & not_solved_by_smaller)
        not_solved_by_smaller &= ~m
    return incremental_masks
