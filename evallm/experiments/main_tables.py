from dataclasses import dataclass

import numpy as np

from evallm.experiments.transducer_plotting import display_acc


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
        return "--"
    return display_acc(format_by_mod, results[name], name)


def main_table_of_results(transducer_results, sequence_completion_results):
    check_all_accounted(transducer_results)
    check_all_accounted(sequence_completion_results)
    table = ""
    # model name, parameters, finetuning, coding model, transducer results, sequence completion results
    table += r"\begin{tabular}{|l|c|c|c|c|c|}" + "\n"
    table += r"\hline" + "\n"
    table += (
        r"\bf Model & \bf Size & \bf RLHF? & \bf Coding? & \bf Transducer & \bf Sequence Completion \\"
        + "\n"
    )
    for group_name, group in grouped_models.items():
        # double line rule for each header group
        table += r"\hline" + "\n"
        table += r"\multicolumn{6}{|c|}{ \bf " + group_name + r"} \\" + "\n"
        table += r"\hline" + "\n"
        for name in group:
            metadata = (
                metadata_baseline
                if group_name == "Baselines"
                else metadata_for_models[name]
            )
            table += f"{name} & {metadata.model_size} & {metadata.rlhf_finetuning_check} & {metadata.coding_model_check} & "
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
