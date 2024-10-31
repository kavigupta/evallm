import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from evallm.utils.bootstrap import boostrap_mean


def plot_absolute_results(ax, which_llm, result_by_length, *, ignore_na):
    plot_model_result(ax, which_llm, result_by_length, ignore_na, "black")
    plot_baselines(ax, result_by_length, ignore_na=ignore_na)
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


def result_calc(ignore_na, r):
    d = {
        "na_ignore": r.success_rate_binary_ignore_na,
        "na_wrong": r.success_rate_binary,
        # na_freq + (1 - na_freq) * na_wrong = na_ignore
        # na_freq * (1 - na_wrong) + na_wrong = na_ignore
        # na_freq = (na_ignore - na_wrong) / (1 - na_wrong)
        "na_freq": (
            (r.success_rate_binary_ignore_na - r.success_rate_binary)
            / (1 - r.success_rate_binary)
            if r.success_rate_binary < 1
            else 0
        ),
    }
    return d[ignore_na]


def plot_model_result(ax, which_llm, result_by_length, ignore_na, color):

    return plot_result(
        ax,
        result_by_length,
        compute_outcome=lambda r: result_calc(ignore_na, r),
        color=color,
        marker="o",
        label=which_llm,
    )


def plot_baselines(ax, result_by_length, *, ignore_na):
    if ignore_na != "na_freq":
        ngrams = range(1, 1 + 5)
        count = len(ngrams) + 2
        linestyles = ["--", "-.", ":"] * 10
        colors = baseline_colors(count)
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
                lambda r, ngram=ngram: r.kgram_success_rates_each[ngram - 1],
                color=colors.pop(0),
                linestyle=linestyles.pop(0),
                label=f"{ngram}gram",
            )
        plot_result(
            ax,
            result_by_length,
            lambda r: getattr(r, "brute_force_inference", np.nan),
            color=colors.pop(0),
            linestyle=linestyles.pop(0),
            label="brute force inference",
        )
    ax.legend()
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(get_ylabel(ignore_na))
    ax.grid()


def baseline_colors(count):
    return [mpl.colors.hsv_to_rgb((i / count, 0.5, 0.5)) for i in range(count)]


def get_ylabel(ignore_na):
    return {
        "na_ignore": "Success rate (N/A = ignored) [%]",
        "na_wrong": "Success rate (N/A = wrong) [%]",
        "na_freq": "N/A frequency",
    }[ignore_na]


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


def plot_all_absolute_results_single_graph(
    results, result_baselines, num_states, *, ignore_na
):
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
    plot_baselines(plt.gca(), result_baselines[num_states], ignore_na=ignore_na)
    plt.title(f"Prediction of {num_states}-state DFA")


def plot_relative_results(relative, name, ax=None):
    if ax is None:
        ax = plt.gca()
    for k in relative:
        ax.plot(relative[k].index, relative[k] * 100, label=f"{k} states")
    ax.legend()
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(f"Meets {name} %")
    ax.axhline(50, color="black")
    ax.grid()


def plot_absolute_results_barchart(
    results, result_baseline, num_states, num_sequence_symbols, *, ignore_na
):
    plt.figure(
        figsize=(8, 8),
        tight_layout=True,
        facecolor="white",
        dpi=200,
    )

    bc = baseline_colors(5 + 2)
    c = BarChartBuilder()

    c.add_result(
        result_baseline[num_states][num_sequence_symbols],
        "null",
        bc.pop(0),
        lambda r: r.null_success_rate,
    )
    for i, ngram in enumerate(range(1, 1 + 5)):
        c.add_result(
            result_baseline[num_states][num_sequence_symbols],
            f"{ngram}gram",
            bc.pop(0),
            lambda r, ngram=ngram: r.kgram_success_rates_each[ngram - 1],
        )

    c.add_result(
        result_baseline[num_states][num_sequence_symbols],
        "brute force inference",
        bc.pop(0),
        lambda r: getattr(r, "brute_force_inference", np.nan),
    )

    models = [model_name(run_name) for run_name in results]
    models = sorted(set(models), key=lambda x: models.index(x))

    for run_name in results:
        if num_states not in results[run_name]:
            continue
        if num_sequence_symbols not in results[run_name][num_states]:
            continue

        res = results[run_name][num_states][num_sequence_symbols]
        c.add_result(
            res,
            run_name,
            f"C{models.index(model_name(run_name))}",
            lambda r: result_calc(ignore_na, r),
        )

    c.sort()
    c.plot_bars()

    plt.title(
        f"Transducer of {num_states}-state DFA on {num_sequence_symbols}-length sequences"
    )
    plt.ylabel(get_ylabel(ignore_na))
    plt.xticks(rotation=90)


def model_name(run_name):
    return run_name.split(" ")[0]


class BarChartBuilder:

    def __init__(self):
        self.names = []
        self.means_each = []
        self.low_high = []
        self.colors = []

    def add_result(self, res, model_name, color, extract):
        values = 100 * np.array([extract(r) for r in res])
        self.names.append(model_name)
        self.means_each.append(np.mean(values))
        [lh] = boostrap_mean(values[:, None]).T
        self.low_high.append(lh)
        self.colors.append(color)

    def sort(self):
        order = np.argsort(self.means_each)
        self.names = [self.names[i] for i in order]
        self.means_each = [self.means_each[i] for i in order]
        self.low_high = [self.low_high[i] for i in order]
        self.colors = [self.colors[i] for i in order]

    def plot_bars(self):
        low_high = np.array(self.low_high).T
        plt.bar(self.names, self.means_each, color=self.colors)
        plt.errorbar(
            self.names,
            (low_high[0] + low_high[1]) / 2,
            yerr=(low_high[1] - low_high[0]) / 2,
            fmt="none",
            ecolor="black",
            capsize=5,
        )
        plt.grid(axis="y")
        plt.yticks(np.arange(50, 101, 5))
        plt.ylim(50, 100)
