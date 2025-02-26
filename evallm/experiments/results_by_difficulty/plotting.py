import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

from evallm.experiments.results_by_difficulty.transducer_aggregate import (
    compute_full_results,
    difficulty_bin_masks_by_aggregate_accuracy,
)
from evallm.experiments.results_by_difficulty.transducer_individual import (
    difficulty_bin_masks_by_individual_accuracy,
    transducer_correctness_by_individual,
)
from evallm.experiments.transducer_plotting import setup_plot
from evallm.utils.bootstrap import boostrap_mean

from .difficulty_categories import mask_names


def plot_aggregate_results_by_difficulty(table):
    setup_plot()
    plt.figure(figsize=(8, 6), tight_layout=True)
    ax = plt.gca()
    results_full = compute_full_results(
        table, difficulty_bin_masks_by_aggregate_accuracy(table)
    )
    category_description = (
        r"First baseline that solves (acc $\geq \frac{28}{30}$) the DFA"
    )
    plot_results_by_diff_category(results_full, category_description, ax)


def plot_individual_results_by_difficulty(data):
    setup_plot()
    plt.figure(figsize=(8, 6), tight_layout=True)
    correctness_by_individual = transducer_correctness_by_individual(data)
    masks = difficulty_bin_masks_by_individual_accuracy(data)
    results = compute_full_results(
        correctness_by_individual,
        masks,
        valid_accuracy_predicate=lambda x: (x == 0) | (x == 100),
    )
    ax = plt.gca()
    category_description = r"First model that makes a justified, confident, and correct prediction on instance"
    plot_results_by_diff_category(results, category_description, ax)


def plot_results_by_diff_category(results_full, category_description, ax):
    endings = [v[-1].mean() for v in results_full.values()]
    yfakes = np.linspace(max(endings), min(endings), len(results_full))
    for i, (yfake, model) in enumerate(
        zip(yfakes, sorted(results_full, key=lambda k: -results_full[k][-1].mean()))
    ):
        results = results_full[model]
        mu = np.array([r.mean() for r in results])
        ci = np.array([boostrap_mean(r) for r in results])
        c = THEME_COLORS[i % len(THEME_COLORS)]
        cd = modify_color(c, 0.5, 0.9)
        ax.plot(mu, label=model, color=cd)
        ax.fill_between(np.arange(len(mask_names)), *ci.T, alpha=0.25, color=c)
        ax.text(x=len(mu) - 0.4, y=yfake, s=model, size=10, color=cd, va="center")
        ax.arrow(x=len(mu) - 0.5, y=yfake, dx=-0.4, dy=mu[-1] - yfake, color=cd)
    ax.set_xticks(np.arange(len(mask_names)))
    ax.set_xticklabels(mask_names)
    ax.set_xlim(0, len(mask_names) + 1)
    ax.set_xlabel(category_description)
    ax.set_ylabel(r"Accuracy on task collection [\%]")
    ax.set_title("Results by difficulty")


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
