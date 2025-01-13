from typing import Dict, List

import numpy as np
import pandas as pd
from automata.fa.dfa import DFA
from matplotlib_venn import venn3

from evallm.enumerate_dfa.pack_dfa import pack_dfa, unpack_dfa
from evallm.experiments.exhaustive_transducer_experiment import (
    TransducerExperimentResultPacked,
)
from evallm.experiments.sequence_completion.sample_sequences import (
    sample_task_instances_given_dfa,
)
from evallm.experiments.transducer_plotting import setup_plot

blue = "#009bff"
green = "#26d94a"
red = "#ff0062"


def transducer_example_csv(
    results: TransducerExperimentResultPacked, samples: Dict[str, List[int]]
):
    completions = [x.message.content for x in results.completions]
    return pd.concat(
        [
            pd.DataFrame(
                [
                    dict(
                        id=f"{k}-{i}",
                        transducer_trace="".join(
                            x
                            for xy in zip(
                                np.array(list("abc"))[results.inputs_packed[idx]],
                                (str(int(x)) for x in results.outputs_packed[idx]),
                            )
                            for x in xy
                        ),
                        completion=completions[idx],
                    )
                    for i, idx in enumerate(samples[k])
                ]
            )
            for k in samples
        ]
    )


def plot_errors(transducer_masks: Dict[str, np.ndarray]):
    setup_plot()
    venn3(
        subsets=[set(np.where(x == 0)[0]) for x in transducer_masks.values()],
        set_labels=[f"{k}: {v.mean():.1%}" for k, v in transducer_masks.items()],
        set_colors=(red, green, blue),
    )


def generate_data_for_dfa(rule, current_setting_s):
    dfa = unpack_dfa(
        pack_dfa(
            DFA(
                states={0, 1, 2},
                input_symbols={0, 1, 2},
                transitions={a: {b: rule(a, b) for b in range(3)} for a in range(3)},
                initial_state=0,
                final_states={0},
            )
        )
    )
    sequence_tasks = sample_task_instances_given_dfa(
        np.random.RandomState(0),
        num_sequences=current_setting_s["num_sequences"],
        num_sequence_symbols=current_setting_s["num_sequence_symbols"],
        num_sequence_symbols_prompt=current_setting_s["num_sequence_symbols_prompt"],
        try_limit=50,
        num_instances=1000,
        dfa=dfa,
    )
    return dfa, sequence_tasks


def qualitative_results_table(table):
    result = ""
    result += r"\begin{tabular}{|r|r|r|}" + "\n"
    result += r"\hline" + "\n"
    for c in table.columns:
        result += f"{c} & "
    result = result[:-2] + r"\\"
    result += r"\hline" + "\n"
    for i in range(len(table)):
        for c in table.columns:
            result += f"{table[c][i]} & "
        result = result[:-2] + r"\\"
        result += "\n"
        result += r"\hline" + "\n"
    result += r"\end{tabular}" + "\n"
    return result
