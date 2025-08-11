import itertools
from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evallm.experiments.transducer_plotting import (
    display_acc,
    plot_bootstrap_means,
    setup_plot,
)

stretch = 1.25


@dataclass
class ModelMetadata:
    model_size: str | None
    rlhf_finetuning: bool
    coding_model: bool

    @property
    def rlhf_finetuning_check(self):
        if self.rlhf_finetuning is None:
            return None
        return r"\checkmark" if self.rlhf_finetuning else ""

    @property
    def coding_model_check(self):
        if self.coding_model is None:
            return None
        return r"\checkmark" if self.coding_model else ""


metadata_for_models = {
    "llama3-8B": ModelMetadata("8.0B", False, False),
    "llama3-70B": ModelMetadata("70.6B", False, False),
    "llama3.1-8B-Instruct": ModelMetadata("8.0B", True, False),
    "llama3.1-70B": ModelMetadata("70.0B", True, False),
    "llama3.1-8B": ModelMetadata("8.0B", True, False),
    "mistral-nemo-minitron-8B": ModelMetadata("8.4B", False, False),
    "mistral-nemo-base-12B": ModelMetadata("12.2B", False, False),
    "mistral-nemo-instruct-12B": ModelMetadata("12.2B", True, False),
    "starcoder2-15b": ModelMetadata("16.0B", False, True),
    "codestral-22B": ModelMetadata("22.2B", False, True),
    "deepseek-coder-33b-instruct": ModelMetadata("33.3B", True, True),
    "qwen-2.5-coder-7B": ModelMetadata("7.6B", False, True),
    "qwen-2.5-coder-instruct-7B": ModelMetadata("7.6B", True, True),
    "qwen-2.5-7B": ModelMetadata("7.6B", False, False),
    "qwen-2.5-32B": ModelMetadata("32.5B", False, False),
    "qwen-2.5-coder-instruct-32B": ModelMetadata("32.8B", True, True),
    "gemma-7b": ModelMetadata("8.5B", False, False),
    "falcon-7b": ModelMetadata("7.2B", False, False),
    "gpt-3.5-instruct": ModelMetadata("?", True, False),
    "gpt-3.5-chat": ModelMetadata("?", True, False),
    "gpt-4o-mini": ModelMetadata("?", True, False),
    "gpt-4o": ModelMetadata("?", True, False),
    "claude-3.5": ModelMetadata("?", True, False),
    # "o1-preview": ModelMetadata("?", True, False),
    "o3-mini": ModelMetadata("?", True, False),
    "gpt-5": ModelMetadata("?", True, False),
}

metadata_baseline = ModelMetadata("--", False, False)

random_null = r"\textsc{Random}$_S$/\textsc{Null}$_T$"

grouped_models = {
    "Baselines": [
        r"\textsc{BruteForce}",
        r"6-\textsc{Gram}",
        r"5-\textsc{Gram}",
        r"4-\textsc{Gram}",
        r"3-\textsc{Gram}",
        r"2-\textsc{Gram}",
        r"\textsc{Common-Suffix}",
        random_null,
    ],
    "Open Weight Completion": [
        "llama3-8B",
        "llama3-70B",
        "llama3.1-8B-Instruct",
        "llama3.1-8B",
        "llama3.1-70B",
        "qwen-2.5-7B",
        "qwen-2.5-32B",
        "mistral-nemo-minitron-8B",
        "mistral-nemo-base-12B",
        "mistral-nemo-instruct-12B",
        "gemma-7b",
        "falcon-7b",
    ],
    "Open Weight Code": [
        "starcoder2-15b",
        "codestral-22B",
        "deepseek-coder-33b-instruct",
        "qwen-2.5-coder-7B",
        "qwen-2.5-coder-instruct-7B",
        "qwen-2.5-coder-instruct-32B",
    ],
    "Proprietary": [
        "gpt-3.5-instruct",
        "gpt-3.5-chat",
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3.5",
        # "o1-preview",
        "o3-mini",
        "gpt-5",
    ],
}

models_considered = [name for names in grouped_models.values() for name in names]
models_to_category = {
    name: category for category, names in grouped_models.items() for name in names
}


def check_all_accounted(names):
    for name in names:
        assert name in models_considered, name


def display_result(results, name):
    format_by_mod = {}
    brute_forces = [x for x in results if "BruteForce" in x]
    if brute_forces:
        [brute_force] = brute_forces
        format_by_mod[brute_force] = r"\cellcolor{lightgray} "
    maximum_acc_model_not_brute_force = max(
        (
            x
            for x in results
            if "BruteForce" not in x
            and models_to_category[x] == models_to_category[name]
        ),
        key=lambda x: np.mean(results[x]),
    )
    format_by_mod[maximum_acc_model_not_brute_force] = r"\bf "
    if name not in results:
        return "-- & --"
    mean_results_non_nan = {
        k: np.mean(v) for k, v in results.items() if not np.isnan(np.mean(v))
    }
    ordered = sorted(mean_results_non_nan, key=mean_results_non_nan.get, reverse=True)
    ordinal = str(ordered.index(name) + 1) if name in ordered else "--"
    return display_acc(format_by_mod, results[name], name) + " & " + ordinal


def replace_null_random(results):
    r"""
    Set \textsc{Random} or \textsc{Null} with a row saying \textsc{Random}$_S$/\textsc{Null}$_T$
    """
    results = results.copy()

    keys = [k for k in results.keys() if "Random" in k or "Null" in k]
    assert len(keys) == 1, keys

    key = keys[0]

    results[random_null] = results.pop(key)

    return results


def main_table_of_results(transducer_results, sequence_completion_results):
    transducer_results = replace_null_random(transducer_results)
    sequence_completion_results = replace_null_random(sequence_completion_results)

    check_all_accounted(transducer_results)
    check_all_accounted(sequence_completion_results)

    table = r"{\renewcommand{\arraystretch}{$stretch}".replace("$stretch", str(stretch))
    # model name, parameters, finetuning, coding model, transducer results, sequence completion results
    table += r"\begin{tabular}{l|ccc|ccccc}" + "\n"
    table += r"\hline" + "\n"
    table += (
        r"\bf Model & \bf Size & \bf IT? & \bf Code? & \bf Sequence Completion & \bf SR & \bf Transducer & \bf TR\\"
        + "\n"
    )
    for group_name, group in grouped_models.items():
        # double line rule for each header group
        table += r"\hline" + "\n"
        table += r"\multicolumn{8}{c}{ \bf " + group_name + r"} \\" + "\n"
        table += r"\hline" + "\n"
        for name, is_last in zip(group, [False] * (len(group) - 1) + [True]):
            metadata = (
                metadata_baseline
                if group_name == "Baselines"
                else metadata_for_models[name]
            )
            table += f"{name} & {metadata.model_size} & {metadata.rlhf_finetuning_check} & {metadata.coding_model_check} & "
            table += (
                f"{display_result(sequence_completion_results, name)} & {display_result(transducer_results, name)} \\\\"
                + "\n"
            )
            # table += r"\vspace{0.1cm}" + "\n"
            if not is_last:
                table += r"\hline" + "\n"
    table += r"\hline" + "\n"
    table += r"\end{tabular}" + "\n"
    table += r"}" + "\n"
    print(table)


def multi_prompt_table_of_results(
    transducer_results, sequence_completion_results, *, bold_best=True
):
    check_all_accounted(transducer_results)
    check_all_accounted(sequence_completion_results)
    prompts = ["Basic", "More-Expl", "COT", "Red-Green"]
    prompts = [rf"\textsc{{{prompt}}}" for prompt in prompts]
    # model name, *prompts
    # Transducer Results then Sequence Completion Results
    results = {
        "Sequence Completion": sequence_completion_results,
        "Transducer": transducer_results,
    }
    table = r"{\renewcommand{\arraystretch}{$stretch}".replace("$stretch", str(stretch))
    table += r"\begin{tabular}{l|" + "".join("c" * len(prompts)) + "}" + "\n"
    table += r"\hline" + "\n"
    table += (
        r"\bf Model & "
        + " & ".join([r"\bf " + prompt for prompt in prompts])
        + r" \\"
        + "\n"
    )
    # table += r"\hline" + "\n"
    for group_name, group in results.items():
        # double line rule for each header group
        table += r"\hline" + "\n"
        table += (
            r"\multicolumn{"
            + str(len(prompts) + 1)
            + r"}{l}{ \bf "
            # + r"\vspace{0.1cm}"
            + group_name
            # + r"\vspace{0.1cm}"
            + r"} \\"
            + "\n"
        )
        table += r"\hline" + "\n"
        for name, is_last in zip(group, [False] * (len(group) - 1) + [True]):
            assert all(p in prompts for p in group[name]), group[name].keys()
            format_by_prompt = {}
            best_p = max(
                group[name],
                key=lambda x, group=group, name=name: (
                    np.mean(group[name][x])
                    if ~np.isnan(np.mean(group[name][x])).all()
                    else -np.inf
                ),
            )

            format_by_prompt[best_p] = r"\bf " if bold_best else ""
            table += f"{name} & "
            for prompt in prompts:
                if prompt in group[name]:
                    acc = display_acc(format_by_prompt, group[name][prompt], prompt)
                    table += f"{acc} & "
                else:
                    table += "-- & "
            table = table[:-2] + r"\\" + "\n"
            if not is_last:
                table += r"\hline" + "\n"

    table += r"\hline" + "\n"
    table += r"\end{tabular}" + "\n"
    table += r"}" + "\n"
    print(table)


def arg_best_prompt(results):
    return {
        k: max(
            v.items(),
            key=lambda yx: (
                np.mean(yx[1])
                if not (isinstance(yx[1], float) and np.isnan(yx[1]))
                else -np.inf
            ),
        )[0]
        for k, v in results.items()
    }


def sanitize_names(results):
    return {k.replace("$_T$", "").replace("$_S$", ""): v for k, v in results.items()}


def best_prompt(results):
    best_prompt_key = arg_best_prompt(results)
    return sanitize_names({k: results[k][best_prompt_key[k]] for k in results})


def multi_prompts(results, *, minimum_number_prompts=2):
    results = {
        k: {
            p.replace("$_T$", "").replace("$_S$", ""): for_kp
            for p, for_kp in for_k.items()
            if "Commas" not in p
        }
        for k, for_k in results.items()
    }
    return {k: v for k, v in results.items() if len(v) >= minimum_number_prompts}


def plot_transducer_vs_sequence_completion(results_sc, results_t):

    category_to_color = {
        "Baselines": "#009bff",
        "Open Weight Completion": "#7a00ff",
        "Open Weight Code": "#ff9500",
        "Proprietary": "#ff0062",
    }

    summary_t = best_prompt(results_t)
    summary_sc = best_prompt(results_sc)

    summary_t = replace_null_random(summary_t)
    summary_sc = replace_null_random(summary_sc)

    common_keys = set(summary_t) & set(summary_sc)
    ordered_keys = [
        x
        for x in models_to_category
        if x in common_keys
        and not isinstance(summary_sc[x], float)
        and not isinstance(summary_t[x], float)
    ]

    to_display = set(grouped_models["Baselines"]) | set(grouped_models["Proprietary"])
    for category, ms in grouped_models.items():
        ms = [m for m in ms if m in ordered_keys]
        for summary in summary_t, summary_sc:
            means = [np.mean(summary[m]) for m in ms]
            to_display.add(ms[np.argmax(means)])
            to_display.add(ms[np.argmin(means)])

    setup_plot()

    ax = plt.gca()

    margin_text = 0.01
    margin_point = 0.05
    off_dist = 2.5

    directions = {
        "\\textsc{BruteForce}": (-1, -1, 0.1),
        "6-\\textsc{Gram}": (1, -1, 0.1),
        "5-\\textsc{Gram}": (-1, 1, 0.1),
        "4-\\textsc{Gram}": (1, -1, 0.1),
        "3-\\textsc{Gram}": (1, 1, 0.1),
        "2-\\textsc{Gram}": (1, 1, 0.1),
        "mistral-nemo-minitron-8B": (-0.0001, 1, 1),
        "qwen-2.5-coder-7B": (0.0001, -1, 1),
        "qwen-2.5-coder-instruct-7B": (0.0001, 1, 1),
        "gpt-3.5-instruct": (-1, 1, 0.1),
        "gpt-4o-mini": (1, -1, 0.1),
        "gpt-4o": (1, -1, 0.1),
        "gemma-7b": (-1, -1, 0.1),
        "falcon-7b": (-1, -1, 0.1),
        "starcoder2-15b": (-1, 1, 1),
        "deepseek-coder-33b-instruct": (1, -1, 1),
        "claude-3.5": (1, -1, 1),
        random_null: (1, 1, 0.1),
    }

    texts = []
    xs, ys = [], []
    for model in ordered_keys:
        color = category_to_color[models_to_category[model]]

        x, y = plot_bootstrap_means(
            ax,
            100 * np.array(summary_sc[model]),
            100 * np.array(summary_t[model]),
            scatter_kwargs=dict(color=color, marker="."),
            err_kwargs=dict(color=color, lw=0.5),
        )
        if models_to_category[model] != "Baselines":
            xs.append(x)
            ys.append(y)
        if model in to_display:
            dirx, diry, rel_dist = directions.get(model, (1, 1, 1))
            # margin_text_this should be the distance required to put margin_text
            # vertical units away from the point
            margin_text_this = margin_text / np.abs(np.sin(np.arctan2(diry, dirx)))
            radius_adj = rel_dist * off_dist / np.sqrt(dirx**2 + diry**2)
            offx, offy = dirx * radius_adj, diry * radius_adj

            model_display = model.replace(r"\textsc{", "").replace("}", "")

            texts += [
                plt.text(
                    x=x + offx,
                    y=y + offy,
                    s=model_display,
                    size=5,
                    ha="right" if dirx < 0 else "left",
                    va="top" if diry < 0 else "bottom",
                    color=darken(color, 0.5),
                )
            ]

            # start near x + offx, but with the margin_text_this applied
            start_pos = (
                x + offx * (1 - margin_text_this),
                y + offy * (1 - margin_text_this),
            )
            # end near x but with the margin_text applied
            end_pos = (
                x + offx * margin_point,
                y + offy * margin_point,
            )

            if rel_dist > 0.2:
                patch = mpl.patches.FancyArrowPatch(
                    start_pos,
                    end_pos,
                    arrowstyle="-|>,head_width=0.2,head_length=0.4",
                    color="black",
                    lw=0.4,
                    mutation_scale=5,
                    zorder=10,
                )
                ax.add_patch(patch)

    lo_x, lo_y = ax.get_xlim()[0], ax.get_ylim()[0]

    for category, color in category_to_color.items():
        plt.scatter([], [], color=color, label=category)

    x_model_max, y_model_max = max(xs), max(ys)

    plt.fill_between(
        [x_model_max, 100],
        y_model_max,
        100,
        color=category_to_color["Baselines"],
        alpha=0.2,
        zorder=0,
        label="Better than all LLMs on both tasks",
        lw=0,
    )

    plt.xlabel("Sequence Completion Result")
    plt.ylabel("Transducer Result")
    plt.xlim(lo_x, 100)
    plt.ylim(lo_y, 100)
    plt.legend()


def darken(color, factor):
    color = np.array(mpl.colors.to_rgba(color))
    return mpl.colors.to_hex(color * factor)


def flat_pandas_table(results_ignore_na, results_null):
    flat_ignore_na = {}
    flat_null = {}
    for model, prompts in results_ignore_na.items():
        for prompt, results in prompts.items():
            flat_ignore_na[(model, prompt)] = np.mean(results)
    for model, prompts in results_null.items():
        for prompt, results in prompts.items():
            flat_null[(model, prompt)] = np.mean(results)

    for k in set(flat_ignore_na) - set(flat_null):
        assert k[0].replace("$_T$", "").replace("$_S$", "") in grouped_models[
            "Baselines"
        ] + [r"\textsc{Null}", r"\textsc{Random}"], k
        flat_null[k] = 0

    assert set(flat_ignore_na) == set(flat_null)
    result = pd.DataFrame(dict(acc_ignore_na=flat_ignore_na, null=flat_null))
    result = result.loc[[x for x in result.index if "Commas" not in x[1]]]
    result = result[result.null < 0.25]
    result["acc_full"] = result.acc_ignore_na * (1 - result.null)
    result["best_prompt"] = False
    for model, prompt in result.index:
        others = result.loc[[x for x in result.index if x[0] == model]]
        if result.loc[model, prompt].acc_ignore_na == others.acc_ignore_na.max():
            result.loc[(model, prompt), "best_prompt"] = True
    return result


def reorderings(flat_table, same_model=False, only_best_prompt=False):
    if only_best_prompt:
        # compute for each model the best score in each column, grouped by prompt
        flat_table = flat_table.groupby(level=0).max()
    for i1, i2 in itertools.combinations(flat_table.index, 2):
        if same_model and i1[0] != i2[0]:
            continue
        r1, r2 = flat_table.loc[i1], flat_table.loc[i2]
        if r1.acc_ignore_na > r2.acc_ignore_na:
            r1, r2 = r2, r1
            i1, i2 = i2, i1
        # now we have that r1 is the one with the lower accuracy on our metric
        if r1.acc_full > r2.acc_full:
            print(
                f"{i1} is better than {i2} on full accuracy; despite being worse on ignore-na accuracy"
            )


def compute_non_directional_p_value(a, b):
    if isinstance(b, float):
        assert np.isnan(b)
        return 0
    a, b = np.array(a), np.array(b)
    if a.mean() < b.mean():
        # guarantee a > b
        a, b = b, a
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    delta = a - b
    p = (
        np.random.RandomState(0).choice(delta, size=(10_000, len(delta))).mean(1) < 0
    ).mean()
    return p * 2


def all_p_values(flat_res):
    return {
        (k1, k2): compute_non_directional_p_value(flat_res[k1], flat_res[k2])
        for k1 in flat_res
        for k2 in flat_res
    }


def plot_significance(ax, flat_res, ps):
    # pylint: disable=consider-using-enumerate,cyclic-import
    from evallm.experiments.results_by_difficulty.plotting import (
        THEME_COLORS,
        modify_color,
    )

    very_significant, significant, not_significant = [
        modify_color(THEME_COLORS[idx], 0.5, 0.9) for idx in (0, 1, 3)
    ]
    # significant = blue
    # not_significant = red
    outer = 0.05
    alpha = 0.75
    sorted_keys = sorted(flat_res, key=lambda k: np.mean(flat_res[k]))[::-1]
    for row in range(len(sorted_keys)):
        for col in range(len(sorted_keys)):
            p = ps[sorted_keys[row], sorted_keys[col]]
            if col >= row:
                color = "white"
            elif p < 0.01:
                color = very_significant
            elif p < 0.05:
                color = significant
            else:
                color = not_significant
            ax.fill_between(
                [col + outer, col - outer + 1],
                [row + outer] * 2,
                [row - outer + 1] * 2,
                color=color,
                alpha=alpha,
                lw=0,
            )
    ax.set_xticks(0.5 + np.arange(len(sorted_keys)), sorted_keys, rotation=90)
    ax.set_yticks(0.5 + np.arange(len(sorted_keys)), sorted_keys)
    pad = 4
    ax.set_ylim(len(sorted_keys), -pad)
    ax.set_xlim(0, len(sorted_keys) + pad)

    dx = 0.2
    for row_col in range(len(sorted_keys)):
        ax.text(
            s=sorted_keys[row_col],
            x=row_col + dx,
            y=row_col + 1 - dx,
            rotation=45,
            size=7,
            ha="left",
            va="bottom",
        )

    # create a legend
    legend_elements = [
        mpl.patches.Patch(color=very_significant, label="$p < 0.01$", alpha=alpha),
        mpl.patches.Patch(
            color=significant, label=r"$0.01 \leq p < 0.05$", alpha=alpha
        ),
        mpl.patches.Patch(color=not_significant, label=r"$0.05 \leq p$", alpha=alpha),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        # bbox_to_anchor=(1, 1),
        fontsize=7,
        title="Significance",
    )
