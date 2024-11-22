from typing import Counter

import numpy as np
from evallm.experiments.transducer_experiment import (
    current_dfa_sample_spec,
    current_transducer_experiments,
    run_transducer_experiment,
)


def standard_experiment(num_states, num_sequence_symbols, model, prompt_class):
    setting = dict(num_states=num_states, num_sequence_symbols=num_sequence_symbols)
    result = run_transducer_experiment(
        model,
        current_dfa_sample_spec(num_states=num_states),
        prompt_class.for_setting(setting),
        num_repeats_per_dfa=30,
        num_dfas=30,
    )
    return [
        dict(
            Counter(
                [
                    int(np.sum(np.diag(x)) > 0.5) if x.max() > 0.5 else 0.5
                    for x in r.confusion_each
                ]
            ).items()
        )
        for r in result
    ]


def standard_experiment_curve(
    num_states_options, num_sequence_symbols_options, model, prompt_class
):
    return dict(
        model=model,
        prompt=prompt_class.__name__,
        results=[
            dict(
                num_states=num_states,
                num_sequence_symbols=num_sequence_symbols,
                results=standard_experiment(
                    num_states, num_sequence_symbols, model, prompt_class
                ),
            )
            for num_states in num_states_options
            for num_sequence_symbols in num_sequence_symbols_options
        ],
    )


def process_stats_to_dict(stats):
    result = dict(
        null_success_rate=float(stats.null_success_rate),
        kgram_success_rates_each=[float(x) for x in stats.kgram_success_rates_each],
    )
    if hasattr(stats, "brute_force_inference"):
        result["brute_force_inference_old"] = float(stats.brute_force_inference)
    return result

def baseline_results():
    none = current_transducer_experiments(
        "none",
        num_states_options=(3, 5, 7),
        num_dfas=1000,
        num_sequence_symbol_options=range(30, 500 + 1, 10),
        just_stats=True,
    )
    results = []
    for num_states in none:
        for num_sequence_symbols in none[num_states]:
            results.append(
                dict(
                    num_states=num_states,
                    num_sequence_symbols=num_sequence_symbols,
                    results=[
                        process_stats_to_dict(r)
                        for r in none[num_states][num_sequence_symbols]
                    ],
                )
            )
    return results
