import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from evallm.experiments.transducer_plotting import setup_plot
from evallm.utils.bootstrap import boostrap_mean

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
    plt.figure(figsize=(8, 6), tight_layout=True)
    masks = compute_masks(table)
    results_full = compute_full_results(table, masks)
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
        plt.fill_between(np.arange(len(masks)), *ci.T, alpha=0.25, color=c)
        plt.text(x=len(mu) - 0.4, y=yfake, s=model, size=10, color=cd, va="center")
        plt.arrow(x=len(mu) - 0.5, y=yfake, dx=-0.4, dy=mu[-1] - yfake, color=cd)
    plt.xticks(np.arange(len(masks)), mask_names)
    plt.xlim(0, len(masks) + 1)
    plt.xlabel(r"First baseline that solves (acc $\geq \frac{28}{30}$) task")
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
