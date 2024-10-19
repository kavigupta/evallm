from frozendict import frozendict
from matplotlib import pyplot as plt
import numpy as np


def res_nan_bad(summary_for_pdfa):
    return (
        100
        * summary_for_pdfa.model_summary[1]
        / sum(summary_for_pdfa.model_summary.values())
    )


def compute_summary_stats(summary):
    pdfas = sorted(summary.keys())
    model_res_no_nan = [
        100
        * summary[pdfa].model_summary[1]
        / (summary[pdfa].model_summary[0] + summary[pdfa].model_summary[1])
        for pdfa in pdfas
    ]
    model_res_nan_bad = np.array([res_nan_bad(summary[pdfa]) for pdfa in pdfas])
    result_5gram = np.array([100 * summary[pdfa].ngram_each[5 - 1] for pdfa in pdfas])

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
    assert all(np.allclose(kgram, kgram_each[0]) for kgram in kgram_each)

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
