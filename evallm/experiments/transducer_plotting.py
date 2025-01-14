import matplotlib.pyplot as plt
import numpy as np

from evallm.utils.bootstrap import boostrap_mean


def produce_table(accuracies, ordered_prompts):
    accuracies_mean = {
        mod: {
            prompt: np.mean(accuracies[mod][prompt])
            for prompt in accuracies[mod]
            if prompt in ordered_prompts
        }
        for mod in accuracies
    }

    best_acc_mean_by_mod = {
        k: max(np.nan_to_num(x, -np.inf) for x in v.values())
        for k, v in accuracies_mean.items()
    }
    models_sorted = sorted(
        accuracies, key=lambda k: np.nan_to_num(best_acc_mean_by_mod[k], -np.inf)
    )[::-1]

    format_by_mod = {}
    format_by_mod[models_sorted[0]] = r"\bf "

    table_alignments = "|" + "|".join("r" + "c" * len(ordered_prompts)) + "|"
    table = ""
    table += r"\begin{tabular}{%s}" % table_alignments + "\n"
    table += r"\hline" + "\n"
    table += " & ".join(["Model"] + ordered_prompts) + r"\\" + "\n"
    table += r"\hline" + "\n"
    for mod in models_sorted:
        table += format_by_mod.get(mod, "") + mod + " &"
        for prompt in ordered_prompts:
            if prompt not in accuracies[mod]:
                table += "--&"
                continue
            table += display_acc(format_by_mod, accuracies[mod][prompt], mod) + "&"
        assert table[-1] == "&"
        table = table[:-1]
        table += r"\\" + "\n"
        table += r"\hline" + "\n"
    table += r"\end{tabular}"

    print(table)


def display_acc(format_by_mod, acc, mod):
    if isinstance(acc, (float, np.float32, np.float64)) and np.isnan(acc):
        return "N/A"
    acc = np.array(acc) * 100
    mu = np.mean(acc)
    lo, hi = boostrap_mean(acc)
    prefix = format_by_mod.get(mod, "")
    return prefix + f"{mu:.1f} ({lo:.1f}--{hi:.1f})"


def plot_bootstrap_means(ax, xs, ys, *, scatter_kwargs, err_kwargs):
    xs, ys = np.array(xs), np.array(ys)
    mu_x, mu_y = np.mean(xs), np.mean(ys)
    lo_x, hi_x = boostrap_mean(xs)
    lo_y, hi_y = boostrap_mean(ys)

    ax.scatter(mu_x, mu_y, **scatter_kwargs)

    err_x_center = (lo_x + hi_x) / 2
    err_y_center = (lo_y + hi_y) / 2

    err_x_bar = (hi_x - lo_x) / 2
    err_y_bar = (hi_y - lo_y) / 2

    ax.errorbar(mu_x, err_y_center, yerr=err_y_bar, **err_kwargs)
    ax.errorbar(err_x_center, mu_y, xerr=err_x_bar, **err_kwargs)

    return mu_x, mu_y


def setup_plot():
    plt.rc("text", usetex=True)
