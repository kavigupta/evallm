from collections import defaultdict

import numpy as np

from evallm.experiments.sequence_completion.sequence_completion_prompt import (
    WithTemperatureSequenceCompletionPrompt,
)
from evallm.experiments.sequence_completion_summary import for_model as for_model_sc
from evallm.experiments.transducer_summary import (
    compute_model_results as compute_model_results_t,
)
from evallm.experiments.transducer_summary import for_model_and_prompt as for_model_t
from evallm.prompting.transducer_prompt import WithTemperatureTransducerPrompter
from evallm.utils.bootstrap import boostrap_mean


def produce_transducer_results(**kwargs):
    r = {
        **for_model_t(
            "mistral-nemo-minitron-8B",
            1000,
            "Basic",
            **kwargs,
        ),
        **for_model_t(
            "claude-3.5",
            30,
            "Basic",
            "More-Expl",
            "COT",
            "Red-Green",
            **kwargs,
        ),
    }

    return {
        mod: {prompt: res for prompt, res in prompts_res.items()}
        for mod, prompts_res in compute_model_results_t(
            r, accuracy_summary=True
        ).items()
    }


def produce_sc_results(**kwargs):
    r = {
        **for_model_sc(
            "mistral-nemo-minitron-8B",
            1000,
            "Basic",
            na_mode="ignore",
            **kwargs,
        ),
        **for_model_sc(
            "claude-3.5",
            30,
            # "Basic", "More-Expl" not included becasue they are NaN
            "COT",
            "Red-Green",
            na_mode="ignore",
            **kwargs,
        ),
    }
    results = defaultdict(dict)
    for mod, prompt in r:
        results[mod][prompt] = r[mod, prompt]
    return results


def both_conditions(produce_results_fn, wrapper):
    result_temp_0 = produce_results_fn()
    result_temp_nonzero = produce_results_fn(
        wrapper=wrapper,
    )
    return result_temp_0, result_temp_nonzero


def compute_results():
    sc = both_conditions(
        produce_sc_results,
        wrapper=lambda prompt: lambda args: WithTemperatureSequenceCompletionPrompt(
            prompt(args), 0.1
        ),
    )
    t = both_conditions(
        produce_transducer_results,
        lambda prompt: WithTemperatureTransducerPrompter(prompt, 0.1),
    )
    return sc, t


def temperature_comparison_subtable(zero_results, nonzero_results):
    """
    Table with the following columns:
    - Model
    - Zero temperature results
    - Nonzero temperature results
    - Difference
    """

    def render(vals):
        mean = np.mean(vals)
        lo, hi = boostrap_mean(vals)
        return f"{mean:.2%} ({lo:.2%} -- {hi:.2%})".replace("%", r"\%")

    table = []
    for model in zero_results.keys():
        for prompt in zero_results[model].keys():
            prompt_disp = rf"\textsc{{{prompt}}}"
            zr = np.array(zero_results[model][prompt])
            nr = np.array(nonzero_results[model][prompt])
            table.append([model, prompt_disp, render(zr), render(nr), render(nr - zr)])
    return table


def emit_table(cells):
    table = r"{\renewcommand{\arraystretch}{1.25}" + "\n"
    table += r"\begin{tabular}{llrrr}" + "\n"
    table += r"\hline" + "\n"
    for row in cells:
        table += " & ".join(row) + r" \\" + "\n"
        table += r"\hline" + "\n"
    table += r"\end{tabular}" + "\n"
    table += r"}" + "\n"
    print(table)


def temperature_comparison_tables(sc, t):
    table = [
        [
            r"\textbf{Model}",
            r"\textbf{Prompt}",
            r"\textbf{Zero Temp}",
            r"\textbf{Nonzero Temp}",
            r"\textbf{Difference}",
        ]
    ]
    table += [[r"\multicolumn{5}{l}{\textbf{Sequence Completion}}"]]
    table += temperature_comparison_subtable(*sc)
    table += [[r"\multicolumn{5}{l}{\textbf{Transducer}}"]]
    table += temperature_comparison_subtable(*t)
    emit_table(table)
