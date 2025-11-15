from typing import List

import numpy as np
from matplotlib import pyplot as plt

from evallm.experiments.models_display import model_by_display_key
from evallm.experiments.transducer_experiment import (
    current_dfa_sample_spec,
    run_transducer_experiment,
)
from evallm.experiments.transducer_plotting import setup_plot
from evallm.prompting.transducer_prompt import BasicSequencePromptNoChat, RemappedPrompt

from .transducer_summary import (
    num_repeats_per_dfa,
    num_sequence_symbols_default,
    num_states_default,
)


def change_symbols_experiment(
    model_key: str,
    *,
    num_states: int = num_states_default,
    num_sequence_symbols: int = num_sequence_symbols_default,
    num_dfas: int,
    other_symbol_sets: List[str],
):
    setting_kwargs = dict(
        num_sequence_symbols=num_sequence_symbols,
        sample_dfa_spec=current_dfa_sample_spec(num_states=num_states),
        num_states=num_states,
    )
    abc_prompt = BasicSequencePromptNoChat.for_setting(setting_kwargs)

    def result_for_prompt(prompt):
        result = run_transducer_experiment(
            model_by_display_key[model_key],
            current_dfa_sample_spec(num_states=num_states),
            prompt,
            num_repeats_per_dfa=num_repeats_per_dfa,
            num_dfas=num_dfas,
        )
        return np.array([r.success_rate_each for r in result]).flatten() < 1

    def result_for_remap(chars):
        remapped_prompt = RemappedPrompt(
            abc_prompt, dict(zip("abc", chars)), {0: 0, 1: 1}
        )
        return result_for_prompt(remapped_prompt)

    result = {"abc": result_for_prompt(abc_prompt)}
    for symbol_set in other_symbol_sets:
        result[symbol_set] = result_for_remap(symbol_set)

    return result


def plot_symbol_correlations(results):
    setup_plot()
    labels = list(results.keys())
    matr = np.array(
        [[np.corrcoef(results[a], results[b])[0, 1] for b in labels] for a in labels]
    )
    fig = plt.gcf()
    ax = plt.gca()
    im = ax.imshow(matr, vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    # place x-axis on top
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(
        axis="x", which="both", top=True, bottom=False, labeltop=True, labelbottom=False
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matr[i, j]:.2f}", ha="center", va="center", color="k")
    ax.set_title("Correlation of Errors Between Symbol Sets")
    fig.colorbar(im)
