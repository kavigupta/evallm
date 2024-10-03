import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from evallm.utils.bootstrap import boostrap_mean


def plot_absolute_results(ax, which_llm, result_by_length, *, ignore_na):
    plot_model_result(ax, which_llm, result_by_length, ignore_na, "black")
    plot_baselines(ax, result_by_length)
    ax.set_title(which_llm)


def plot_result(ax, result_by_length, compute_outcome, **kwargs):
    lengths = sorted(result_by_length)
    results = (
        100
        * np.array(
            [
                [compute_outcome(r) for r in result_by_length[length]]
                for length in lengths
            ]
        ).T
    )
    lo, hi = boostrap_mean(results)
    ax.plot(lengths, results.mean(0), **kwargs)
    ax.fill_between(
        lengths, lo, hi, alpha=0.3, **{k: v for k, v in kwargs.items() if k == "color"}
    )


def plot_model_result(ax, which_llm, result_by_length, ignore_na, color):

    return plot_result(
        ax,
        result_by_length,
        lambda r: (
            r.success_rate_binary_ignore_na if ignore_na else r.success_rate_binary
        ),
        color=color,
        marker="o",
        label=which_llm,
    )


def plot_baselines(ax, result_by_length):
    ngrams = range(1, 1 + 5)
    count = len(ngrams) + 1
    linestyles = ["--", "-.", ":"] * 10
    colors = [mpl.colors.hsv_to_rgb((i / count, 0.5, 0.5)) for i in range(count)]
    plot_result(
        ax,
        result_by_length,
        lambda r: r.null_success_rate,
        color=colors.pop(0),
        linestyle=linestyles.pop(0),
        label="null",
    )
    for ngram in ngrams:
        plot_result(
            ax,
            result_by_length,
            lambda r: r.kgram_success_rates_each[ngram - 1],
            color=colors.pop(0),
            linestyle=linestyles.pop(0),
            label=f"{ngram}gram",
        )
    ax.legend()
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Success rate [%]")
    ax.grid()


def plot_all_absolute_results(results, num_states, *, ignore_na):
    _, axs = plt.subplots(
        1,
        len(results),
        figsize=(5 * len(results), 5),
        tight_layout=True,
        facecolor="white",
    )
    axs = [axs] if len(results) == 1 else axs.flatten()
    for ax, model_name in zip(axs, results):
        plot_absolute_results(
            ax, model_name, results[model_name][num_states], ignore_na=ignore_na
        )
    plt.suptitle(f"Prediction of {num_states}-state DFA")


def plot_all_absolute_results_single_graph(results, result_baselines, num_states, *, ignore_na):
    plt.figure(
        figsize=(8, 8),
        tight_layout=True,
        facecolor="white",
        dpi=200,
    )
    for i, model_name in enumerate(results):
        if num_states not in results[model_name]:
            continue
        plot_model_result(
            plt.gca(),
            model_name,
            results[model_name][num_states],
            ignore_na=ignore_na,
            color=f"C{i}",
        )
    plot_baselines(plt.gca(), result_baselines[num_states])
    plt.title(f"Prediction of {num_states}-state DFA")