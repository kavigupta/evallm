import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from evallm.experiments.main_tables import arg_best_prompt, sanitize_names
from evallm.experiments.transducer_plotting import setup_plot
from evallm.experiments.transducer_summary import transducer_results
from evallm.infer_dfa.brute_force_transducer import brute_force_on_instances
from evallm.utils.bootstrap import boostrap_mean
from evallm.utils.kgrams import predict_based_on_kgram

mask_names = ["%s-\\textsc{Gram}" % n for n in range(2, 1 + 5)] + [
    r"\textsc{BruteForce}"
]


def compute_masks(table):
    ngrams = np.array([table[name] for name in mask_names])
    not_solved_by_smaller = np.ones(ngrams.shape[1], bool)
    masks = []
    for ng in ngrams:
        ng = ng >= 28 / 30
        masks.append(ng & not_solved_by_smaller)
        not_solved_by_smaller &= ~ng
    return masks


def compute_full_results(table, masks):
    results_full = {}
    for model in table:
        # if model in grouped_models["Baselines"]:
        # continue
        if model == r"\textsc{Null}":
            continue
        res = 100 * np.array(table[model])
        if res.shape[0] < 100:
            continue
        results_full[model] = [res[mask[: len(res)]] for mask in masks]

    return results_full


def plot_results_by_difficulty(table):
    masks = compute_masks(table)
    results_full = compute_full_results(table, masks)
    category_description = r"First baseline that solves (acc $\geq \frac{28}{30}$) task"
    plot_results_by_diff_category(results_full, category_description)


def plot_results_by_diff_category(results_full, category_description):
    plt.figure(figsize=(8, 6), tight_layout=True)
    endings = [v[-1].mean() for v in results_full.values()]
    yfakes = np.linspace(max(endings), min(endings), len(results_full))
    setup_plot()
    for i, (yfake, model) in enumerate(
        zip(yfakes, sorted(results_full, key=lambda k: -results_full[k][-1].mean()))
    ):
        results = results_full[model]
        mu = np.array([r.mean() for r in results])
        ci = np.array([boostrap_mean(r) for r in results])
        c = THEME_COLORS[i % len(THEME_COLORS)]
        cd = modify_color(c, 0.5, 0.9)
        plt.plot(mu, label=model, color=cd)
        plt.fill_between(np.arange(len(mask_names)), *ci.T, alpha=0.25, color=c)
        plt.text(x=len(mu) - 0.4, y=yfake, s=model, size=10, color=cd, va="center")
        plt.arrow(x=len(mu) - 0.5, y=yfake, dx=-0.4, dy=mu[-1] - yfake, color=cd)
    plt.xticks(np.arange(len(mask_names)), mask_names)
    plt.xlim(0, len(mask_names) + 1)
    plt.xlabel(category_description)
    plt.ylabel(r"Accuracy on task collection [\%]")
    plt.title("Results by difficulty")


THEME_COLORS = [
    "#80cdff",  # blue
    "#ffca80",  # orange
    "#60e37a",  # green
    "#ff80b1",  # pink
    "#bd80ff",  # purple
    "#000000",  # black
]


def modify_color(color, saturation_change, value_change):
    m = mcolors.ColorConverter().to_rgb
    rgb = m(color)
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv[1] = 1 - (1 - hsv[1]) * saturation_change
    hsv[2] *= value_change
    color = mcolors.hsv_to_rgb(hsv)
    return color


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


def instance_difficulty_category_masks(data):
    masks_each = [
        justified_ngram_prediction(data, ngram) for ngram in range(2, 1 + 5)
    ] + [correct_brute_force_on_all(data, correctness_kind="justified")]
    incremental_masks = []
    not_solved_by_smaller = np.ones_like(masks_each[0], bool)
    for m in masks_each:
        incremental_masks.append(m & not_solved_by_smaller)
        not_solved_by_smaller &= ~m
    return incremental_masks
