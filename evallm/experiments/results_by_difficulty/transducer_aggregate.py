import numpy as np


from .difficulty_categories import mask_names


def difficulty_bin_masks_by_aggregate_accuracy(table):
    ngrams = np.array([table[name] for name in mask_names])
    not_solved_by_smaller = np.ones(ngrams.shape[1], bool)
    masks = []
    for ng in ngrams:
        ng = ng >= 28 / 30
        masks.append(ng & not_solved_by_smaller)
        not_solved_by_smaller &= ~ng
    return masks


def compute_full_results(
    model_to_accuracy,
    masks,
    *,
    valid_accuracy_predicate=lambda x: np.ones_like(x, bool),
):
    results_full = {}
    for model in model_to_accuracy:
        if model == r"\textsc{Null}":
            continue
        res = 100 * np.array(model_to_accuracy[model])
        if res.shape[0] < 100:
            continue
        results_full[model] = [
            res[valid_accuracy_predicate(res) & mask[: len(res)]] for mask in masks
        ]

    return results_full
