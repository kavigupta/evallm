from dataclasses import dataclass

import numpy as np

from evallm.experiments.transducer_plotting import display_acc


@dataclass
class ModelMetadata:
    model_size: float = None
    rlhf_finetuning: bool = None
    coding_model: bool = None


metadata_for_models = {
    "llama3-8B": ModelMetadata(),
    "llama3-70B": ModelMetadata(),
    "llama3.1-8B-Instruct": ModelMetadata(),
    "mistral-nemo-minitron-8B": ModelMetadata(),
    "mistral-nemo-base-12B": ModelMetadata(),
    "mistral-nemo-instruct-12B": ModelMetadata(),
    "starcoder2-15b": ModelMetadata(),
    "codestral-22B": ModelMetadata(),
    "deepseek-coder-33b-instruct": ModelMetadata(),
    "qwen-2.5-coder-7B": ModelMetadata(),
    "qwen-2.5-coder-instruct-7B": ModelMetadata(),
    "qwen-2.5-coder-instruct-32B": ModelMetadata(),
    "gemma-7b": ModelMetadata(),
    "falcon-7b": ModelMetadata(),
    "gpt-3.5-instruct": ModelMetadata(),
    "gpt-3.5-chat": ModelMetadata(),
    "gpt-4o-mini": ModelMetadata(),
    "gpt-4o": ModelMetadata(),
    "claude-3.5": ModelMetadata(),
}

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
    "Closed Source": [
        "gpt-3.5-instruct",
        "gpt-3.5-chat",
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3.5",
    ],
}

models_considered = [name for names in grouped_models.values() for name in names]


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
        [x for x in results if "BruteForce" not in x],
        key=lambda x: np.mean(results[x]),
    )
    format_by_mod[maximum_acc_model_not_brute_force] = r"\bf "
    if name not in results:
        return "--"
    return display_acc(format_by_mod, results[name], name)


def main_table_of_results(transducer_results, sequence_completion_results):
    check_all_accounted(transducer_results)
    check_all_accounted(sequence_completion_results)
    table = ""
    # model name, parameters, finetuning, coding model, transducer results, sequence completion results
    table += r"\begin{tabular}{|l|r|c|c|c|c|}" + "\n"
    table += r"\hline" + "\n"
    table += (
        r"\bf Model & \bf Size & \bf RLHF? & \bf Coding? & \bf Transducer & \bf Sequence Completion \\"
        + "\n"
    )
    for group_name, group in grouped_models.items():
        # double line rule for each header group
        table += r"\hline" + "\n"
        table += r"\multicolumn{6}{|l|}{ \bf " + group_name + r"} \\" + "\n"
        table += r"\hline" + "\n"
        for name in group:
            if group_name == "Baselines":
                table += f"{name}&&&&"
            else:
                metadata = metadata_for_models[name]
                table += f"{name} & {metadata.model_size} & {metadata.rlhf_finetuning} & {metadata.coding_model} & "
            table += (
                f"{display_result(transducer_results, name)} & {display_result(sequence_completion_results, name)} \\\\"
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
        "Transducer": transducer_results,
        "Sequence Completion": sequence_completion_results,
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
            best_prompt = max(
                group[name],
                key=lambda x: (
                    np.mean(group[name][x])
                    if ~np.isnan(np.mean(group[name][x])).all()
                    else -np.inf
                ),
            )

            format_by_prompt[best_prompt] = r"\bf "
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
