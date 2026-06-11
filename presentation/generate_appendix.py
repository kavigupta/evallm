"""Regenerate the two appendix figures used in the talk:

    images/more-examples-per-problem.png   (accuracy vs. number of examples)
    images/more-states-per-problem.png     (accuracy vs. number of DFA states)

Unlike the busy notebook version, each panel shows only two series:

    * the strongest n-gram baseline, in gray -- the infinity-gram where it
      exists (the transducer task); on sequence completion, which has no
      infinity-gram, this is the highest available n-gram;
    * mistral-nemo-minitron-8B, in black, labelled simply "minitron".

The other models (qwen, llama) and the lower n-grams are dropped.  Figures are
saved with a transparent ("alpha") background and the LaTeX font preset
(setup_plot), so they sit cleanly on the slide background.
"""

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from evallm.experiments.transducer_summary import (
    for_model_and_prompt as for_model_t,
    compute_model_results as compute_model_results_t,
    compute_deterministic_baseline_outcomes as dbo_t,
)
from evallm.experiments.sequence_completion_summary import (
    for_model as for_model_sc,
    current_setting as current_setting_s,
    results_for_baseline as rfb_s,
)
from evallm.experiments.transducer_plotting import setup_plot
from evallm.experiments.results_by_difficulty.plotting import THEME_COLORS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images")

COUNT = 1000
MINITRON = "mistral-nemo-minitron-8B"
GRAM_COLOR = "#808080"  # gray, for the infinity-gram
MINITRON_COLOR = "#000000"  # black


def produce_transducer_results(max_gram, include_infinity_gram, **kwargs):
    r = for_model_t(MINITRON, COUNT, "Basic", **kwargs)
    baselines = {
        mod: {prompt: np.mean(v) for prompt, v in by_prompt.items()}
        for mod, by_prompt in dbo_t(
            **kwargs,
            include_brute_force=False,
            include_null=False,
            min_gram=4,
            max_gram=max_gram,
            include_infinity_gram=include_infinity_gram,
        ).items()
    }
    models = {
        mod: {prompt: np.mean(res) for prompt, res in prompts_res.items()}
        for mod, prompts_res in compute_model_results_t(
            r, accuracy_summary=True
        ).items()
    }
    overall = {**baselines, **models}
    return {k: v["Basic"] for k, v in overall.items()}


def produce_sc_results(**kwargs):
    setting = {**current_setting_s, **kwargs}
    r = {
        **rfb_s(setting, include_non_ngram=False, min_ngram=4),
        **for_model_sc(MINITRON, COUNT, "Basic", na_mode="ignore", setting=setting),
    }
    results = defaultdict(dict)
    for mod, prompt in r:
        assert prompt == "Basic"
        results[mod] = np.mean(r[mod, prompt])
    return results


def style_for(column, i):
    """(color, label, linewidth) for a column. Finite n-grams keep their
    original THEME_COLORS; the infinity-gram is gray; minitron is black."""
    if column == MINITRON:
        return MINITRON_COLOR, "minitron", 2.3
    if "infty" in column:
        return GRAM_COLOR, r"$\infty$-gram", 2.0
    n = int(re.match(r"(\d+)", column).group(1))
    return THEME_COLORS[i % len(THEME_COLORS)], rf"${n}$-gram", 1.4


def plot_panel(ax, df, title, xlabel, xlog):
    # df columns are ordered: 4..N-gram, [infty-gram], minitron.
    for i, column in enumerate(df.columns):
        color, label, lw = style_for(column, i)
        ax.plot(df.index, 100 * df[column], label=label, color=color, linewidth=lw)
    ax.legend(fontsize=8, framealpha=0.6)
    if xlog:
        ax.set_xscale("log")
    ax.set_xticks(df.index, df.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Accuracy [\%]")
    ax.set_title(title)
    ax.grid(alpha=0.4)
    ax.set_ylim(ax.get_ylim()[0], 100)


def plot_both(transducer, sc, xlabel, xlog, path):
    setup_plot()
    size = 4
    fig, axs = plt.subplots(
        1, 2, figsize=(size * 2, size), dpi=200, tight_layout=True
    )
    plot_panel(axs[0], transducer, "Transducer", xlabel, xlog)
    plot_panel(axs[1], sc, "Sequence Completion", xlabel, xlog)
    fig.savefig(path, transparent=True)
    plt.close(fig)
    print("wrote", path)


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- varying the number of examples per problem ----
    nss_vals = [30, 60, 150, 300, 600, 1200]
    t_examples = pd.DataFrame(
        [
            produce_transducer_results(
                max_gram=8, include_infinity_gram=True, num_sequence_symbols=nss
            )
            for nss in nss_vals
        ],
        index=nss_vals,
    )
    sc_examples = pd.DataFrame(
        [produce_sc_results(num_sequences=nss) for nss in nss_vals],
        index=nss_vals,
    )
    plot_both(
        t_examples,
        sc_examples,
        "Number examples [log scale]",
        True,
        os.path.join(OUT, "more-examples-per-problem.png"),
    )

    # ---- varying the number of states in the DFA ----
    num_states_to_try = [3, 4, 5, 6, 7, 8]
    t_states = pd.DataFrame(
        [
            produce_transducer_results(
                max_gram=8,
                include_infinity_gram=True,
                num_states=num_states,
                num_sequence_symbols=30,
            )
            for num_states in num_states_to_try
        ],
        index=num_states_to_try,
    )
    sc_states = pd.DataFrame(
        [
            produce_sc_results(
                num_sequences=30,
                dfa_spec={**current_setting_s["dfa_spec"], "n_states": num_states},
            )
            for num_states in num_states_to_try
        ],
        index=num_states_to_try,
    )
    plot_both(
        t_states,
        sc_states,
        "Number states in DFA",
        False,
        os.path.join(OUT, "more-states-per-problem.png"),
    )


if __name__ == "__main__":
    main()
