from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from evallm.experiments.transducer_plotting import display_acc, plot_bootstrap_means


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
    "mistral-nemo-minitron-8B": ModelMetadata("8.4B", False, False),
    "mistral-nemo-base-12B": ModelMetadata("12.2B", False, False),
    "mistral-nemo-instruct-12B": ModelMetadata("12.2B", True, False),
    "starcoder2-15b": ModelMetadata("16.0B", False, True),
    "codestral-22B": ModelMetadata("22.2B", False, True),
    "deepseek-coder-33b-instruct": ModelMetadata("33.3B", True, True),
    "qwen-2.5-coder-7B": ModelMetadata("7.6B", False, True),
    "qwen-2.5-coder-instruct-7B": ModelMetadata("7.6B", True, True),
    "qwen-2.5-coder-instruct-32B": ModelMetadata("32.8B", True, True),
    "gemma-7b": ModelMetadata("8.5B", False, False),
    "falcon-7b": ModelMetadata("7.2B", False, False),
    "gpt-3.5-instruct": ModelMetadata("?", True, False),
    "gpt-3.5-chat": ModelMetadata("?", True, False),
    "gpt-4o-mini": ModelMetadata("?", True, False),
    "gpt-4o": ModelMetadata("?", True, False),
    "claude-3.5": ModelMetadata("?", True, False),
}

metadata_baseline = ModelMetadata("--", False, False)

grouped_models = {
    "Baselines": [
        r"\textsc{BruteForce}",
        r"6-\textsc{Gram}",
        r"5-\textsc{Gram}",
        r"4-\textsc{Gram}",
        r"3-\textsc{Gram}",
        r"2-\textsc{Gram}",
        r"\textsc{Common-Suffix}",
        r"\textsc{Null}",
        r"\textsc{Random}",
    ],
    "Open Source Completion": [
        "llama3-8B",
        "llama3-70B",
        "llama3.1-8B-Instruct",
        "mistral-nemo-minitron-8B",
        "mistral-nemo-base-12B",
        "mistral-nemo-instruct-12B",
        "gemma-7b",
        "falcon-7b",
    ],
    "Open Source Code": [
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


def main_table_of_results(transducer_results, sequence_completion_results):
    check_all_accounted(transducer_results)
    check_all_accounted(sequence_completion_results)
    table = ""
    # model name, parameters, finetuning, coding model, transducer results, sequence completion results
    table += r"\begin{tabular}{|l|c|c|c|c|c|c|c|c|}" + "\n"
    table += r"\hline" + "\n"
    table += (
        r"\bf Model & \bf Size & \bf IT? & \bf Code? & \bf Sequence Completion & \bf SR & \bf Transducer & \bf TR\\"
        + "\n"
    )
    for group_name, group in grouped_models.items():
        # double line rule for each header group
        table += r"\hline" + "\n"
        table += r"\multicolumn{8}{|c|}{ \bf " + group_name + r"} \\" + "\n"
        table += r"\hline" + "\n"
        for name in group:
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
            table += r"\hline" + "\n"
    table += r"\end{tabular}" + "\n"
    print(table)


def multi_prompt_table_of_results(transducer_results, sequence_completion_results):
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
    table = ""
    table += r"\begin{tabular}{|l|" + "|".join("c" * len(prompts)) + "|}" + "\n"
    table += r"\hline" + "\n"
    table += (
        r"\bf Model & "
        + " & ".join([r"\bf " + prompt for prompt in prompts])
        + r" \\"
        + "\n"
    )
    table += r"\hline" + "\n"
    for group_name, group in results.items():
        # double line rule for each header group
        table += r"\hline" + "\n"
        table += (
            r"\multicolumn{"
            + str(len(prompts) + 1)
            + r"}{|l|}{ \bf "
            + group_name
            + r"} \\"
            + "\n"
        )
        table += r"\hline" + "\n"
        for name in group:
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

            format_by_prompt[best_p] = r"\bf "
            table += f"{name} & "
            for prompt in prompts:
                if prompt in group[name]:
                    acc = display_acc(format_by_prompt, group[name][prompt], prompt)
                    table += f"{acc} & "
                else:
                    table += "-- & "
            table = table[:-2] + r"\\" + "\n"
            table += r"\hline" + "\n"

    table += r"\end{tabular}" + "\n"
    print(table)


def best_prompt(results):
    return {
        k.replace("$_T$", "").replace("$_S$", ""): max(
            v.values(),
            key=lambda x: (
                np.mean(x) if not (isinstance(x, float) and np.isnan(x)) else -np.inf
            ),
        )
        for k, v in results.items()
    }


def multi_prompts(results):
    results = {
        k: {
            p.replace("$_T$", "").replace("$_S$", ""): for_kp
            for p, for_kp in for_k.items()
            if "Commas" not in p
        }
        for k, for_k in results.items()
    }
    return {k: v for k, v in results.items() if len(v) > 1}


def plot_transducer_vs_sequence_completion(results_sc, results_t):

    category_to_color = {
        "Baselines": "#009bff",
        "Open Source Code": "#ff9500",
        "Open Source Completion": "#7a00ff",
        "Proprietary": "#ff0062",
    }

    summary_t = best_prompt(results_t)
    summary_sc = best_prompt(results_sc)
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

    plt.figure(figsize=(7, 4), dpi=400, facecolor="white", tight_layout=True)
    plt.rc("text", usetex=True)

    ax = plt.gca()

    margin_text = 0.01
    margin_point = 0.05
    off_dist = 2.5

    directions = {
        "\\textsc{BruteForce}": (-1, -1, 0.1),
        "6-\\textsc{Gram}": (1, -1, 0.1),
        "5-\\textsc{Gram}": (-1, 1, 0.1),
        "4-\\textsc{Gram}": (1, -1, 0.1),
        "3-\\textsc{Gram}": (1, -1, 0.1),
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
    }

    texts = []
    for model in ordered_keys:
        color = category_to_color[models_to_category[model]]

        x, y = plot_bootstrap_means(
            ax,
            100 * np.array(summary_sc[model]),
            100 * np.array(summary_t[model]),
            scatter_kwargs=dict(color=color, marker="."),
            err_kwargs=dict(color=color, lw=0.5),
        )
        if model in to_display:
            dirx, diry, rel_dist = directions.get(model, (1, 1, 1))
            # margin_text_this should be the distance required to put margin_text
            # vertical units away from the point
            margin_text_this = margin_text / np.abs(np.sin(np.arctan2(diry, dirx)))
            radius_adj = rel_dist * off_dist / np.sqrt(dirx**2 + diry**2)
            offx, offy = dirx * radius_adj, diry * radius_adj

            texts += [
                plt.text(
                    x=x + offx,
                    y=y + offy,
                    s=model,
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

    plt.xlabel("Sequence Completion Result")
    plt.ylabel("Transducer Result")
    plt.ylim(plt.ylim()[0], 100)
    # legend for each category
    for category, color in category_to_color.items():
        plt.scatter([], [], color=color, label=category)
    plt.legend()


def darken(color, factor):
    color = np.array(mpl.colors.to_rgba(color))
    return mpl.colors.to_hex(color * factor)
