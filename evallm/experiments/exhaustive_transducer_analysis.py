import numpy as np
import pandas as pd
from frozendict import frozendict
from matplotlib import pyplot as plt

from evallm.enumerate_dfa.enumerate import (
    enumerate_packed_dfas_no_permutations_valid_no_io_permutations,
)


def res_nan_bad(summary_for_pdfa):
    return (
        100
        * summary_for_pdfa.model_summary[1]
        / sum(summary_for_pdfa.model_summary.values())
    )


def compute_summary_stats(summary):
    pdfas = sorted(enumerate_packed_dfas_no_permutations_valid_no_io_permutations(3, 3))
    model_res_no_nan = [
        (
            100
            * summary[pdfa].model_summary[1]
            / (summary[pdfa].model_summary[0] + summary[pdfa].model_summary[1])
            if pdfa in summary
            else np.nan
        )
        for pdfa in pdfas
    ]
    model_res_nan_bad = np.array(
        [res_nan_bad(summary[pdfa]) if pdfa in summary else np.nan for pdfa in pdfas]
    )
    result_5gram = np.array(
        [
            100 * summary[pdfa].ngram_each[5 - 1] if pdfa in summary else np.nan
            for pdfa in pdfas
        ]
    )

    return {
        "pdfas": pdfas,
        "model_res_no_nan": model_res_no_nan,
        "model_res_nan_bad": model_res_nan_bad,
        "result_5gram": result_5gram,
    }


def compute_all_summary_stats(summaries):
    summary_stats_each = {
        model: compute_summary_stats(summary) for model, summary in summaries.items()
    }

    pdfas_each = [
        summary_stats["pdfas"] for summary_stats in summary_stats_each.values()
    ]
    pdfas = pdfas_each[0]
    assert all(pdfas == pdfas_i for pdfas_i in pdfas_each)

    kgram_each = [
        summary_stats["result_5gram"] for summary_stats in summary_stats_each.values()
    ]
    assert all(all_non_nans_close(kgram, kgram_each[0]) for kgram in kgram_each)

    kgram = kgram_each[0]

    model_res_no_nan_each = {
        model: summary_stats["model_res_no_nan"]
        for model, summary_stats in summary_stats_each.items()
    }

    model_res_nan_bad_each = {
        model: summary_stats["model_res_nan_bad"]
        for model, summary_stats in summary_stats_each.items()
    }

    return pdfas, kgram, model_res_no_nan_each, model_res_nan_bad_each


def all_non_nans_close(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    return np.allclose(a[mask], b[mask])


def plot_comparison(x, y, xlabel, ylabel, mask_fns=frozendict({})):
    plt.scatter(x, y, alpha=0.1, marker=".", color="black")
    xlim, ylim = plt.xlim(), plt.ylim()
    plt.plot([50, 110], [50, 110], color="black", lw=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    masks = {}
    for label, mask_fn in mask_fns.items():
        mask = mask_fn(x, y)
        masks[label] = mask
        plt.fill_between(
            [x[mask].min() - 0.5, x[mask].max() + 0.5],
            [y[mask].min() - 0.5] * 2,
            [y[mask].max() + 0.5] * 2,
            alpha=0.5,
            label=label,
        )
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend()
    plt.show()
    return masks


def plot_correlations(by_model):
    for_corr = pd.DataFrame(by_model)
    for_corr = for_corr[~np.isnan(for_corr.T).any()]
    correlation_array = np.array(
        [
            [
                np.corrcoef(for_corr[k1], for_corr[k2])[0, 1] if i <= j else np.nan
                for i, k1 in enumerate(for_corr)
            ]
            for j, k2 in enumerate(for_corr)
        ]
    )
    vmax = np.nanmax(correlation_array)
    plt.imshow(correlation_array, vmin=0, vmax=vmax, cmap="viridis")
    ticks, labels = zip(*enumerate(for_corr))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.colorbar()
    for i in range(correlation_array.shape[1]):
        for j in range(correlation_array.shape[0]):
            val = correlation_array[j, i]
            if np.isnan(val):
                continue
            plt.text(
                i,
                j,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if val > 0.5 else "white",
            )
    plt.title("Correlation ($r$) between models'\nperformance on different DFAs")
