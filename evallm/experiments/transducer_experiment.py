import itertools
from dataclasses import dataclass
from functools import cached_property
from types import SimpleNamespace

import numpy as np
import tqdm.auto as tqdm
from permacache import permacache

from evallm.infer_dfa.brute_force_transducer import brute_force_accuracy
from evallm.prompting.prompter import TrivialProblemError
from evallm.prompting.transducer_prompt import BasicInstructionTransducerPrompter
from evallm.sample_dfa.sample_dfa import sample_dfa
from evallm.utils import predict_based_on_kgram


@dataclass
class TransducerExperimentResult:
    inputs: list
    outputs: list
    prompts: list
    confusion_each: np.ndarray

    @classmethod
    def of(cls, metas, prompts, results):
        inputs, outputs = zip(*metas)
        res = cls(inputs, outputs, prompts, results)
        # populate cache
        # pylint: disable=pointless-statement
        res.kgram_successes_each
        res.success_rate
        res.null_success_rate
        return res

    @cached_property
    def confusion(self):
        return np.mean(self.confusion_each, axis=0)

    @cached_property
    def null_success_rate(self):
        p = np.mean([out[-1] for out in self.outputs])
        return np.max([p, 1 - p])

    @cached_property
    def success_rate(self):
        return np.sum(np.diag(self.confusion))

    @cached_property
    def success_rate_binary(self):
        return np.mean([np.sum(np.diag(x)) > 0.5 for x in self.confusion_each])

    @cached_property
    def success_rate_binary_ignore_na(self):
        return np.mean(
            [np.sum(np.diag(x)) > 0.5 for x in self.confusion_each if x.max() > 0.5]
        )

    @cached_property
    def success_rate_each(self):
        return [np.sum(np.diag(confusion)) for confusion in self.confusion_each]

    @cached_property
    def kgram_successes_each(self):
        return [
            [x == out[-1] for x in predict_based_on_kgram(inp, out)]
            for inp, out in zip(self.inputs, self.outputs)
        ]

    @property
    def kgram_success_rates_each(self):
        num_k = max(len(successes) for successes in self.kgram_successes_each)
        success_rates = []
        for k in range(num_k):
            success_rates.append(
                np.mean(
                    [
                        successes[k] if k < len(successes) else successes[-1]
                        for successes in self.kgram_successes_each
                    ]
                )
            )
        return success_rates

def single_transducer_experiment(
    *,
    seed,
    model,
    num_repeats_per_dfa,
    sample_dfa_spec,
    prompter,
    print_completions=False,
):
    rng = np.random.RandomState(seed)
    dfa = sample_dfa(sample_dfa_spec, rng)
    completions, metas, prompts, scores = prompter.run_experiment(
        dfa, rng, model, num_repeats_per_dfa
    )
    if print_completions:
        print(completions)
    return TransducerExperimentResult.of(metas, prompts, scores)


@permacache(
    "evallm/experiments/transducer_experiment_10", key_function=dict(prompter=repr)
)
def run_transducer_experiment(
    model,
    sample_dfa_spec,
    prompter,
    num_repeats_per_dfa,
    num_dfas,
):
    results = []
    print(f"Model: {model}, Sampling: {sample_dfa_spec}, Prompter: {prompter}")
    pbar = tqdm.tqdm(
        total=num_dfas,
    )
    for seed in itertools.count():
        try:
            result = single_transducer_experiment(
                seed=seed,
                model=model,
                num_repeats_per_dfa=num_repeats_per_dfa,
                sample_dfa_spec=sample_dfa_spec,
                prompter=prompter,
            )
        except TrivialProblemError:
            continue
        results.append(result)
        pbar.update()
        if len(results) == num_dfas:
            pbar.close()
            break
    return results


@permacache(
    "evallm/experiments/transducer_experiment_just_stats_3",
    key_function=dict(prompter=repr),
)
def run_transducer_experiment_just_stats(
    model,
    sample_dfa_spec,
    prompter,
    num_repeats_per_dfa,
    num_dfas,
):
    print(f"Model: {model}, Sampling: {sample_dfa_spec}, Prompter: {prompter}")
    args = (
        model,
        sample_dfa_spec,
        prompter,
        num_repeats_per_dfa,
        num_dfas,
    )
    results = run_transducer_experiment.function(*args)
    new_results = []
    for result in tqdm.tqdm(results):
        res_stats = SimpleNamespace(
            null_success_rate=result.null_success_rate,
            kgram_success_rates_each=result.kgram_success_rates_each,
        )
        new_results.append(res_stats)
    return new_results


num_sequence_symbol_options_default = (30, 120, 500)


def current_dfa_sample_spec(num_states):
    return dict(type="sample_reachable_dfa", n_states=num_states, n_symbols=3)


@permacache(
    "evallm/experiments/run_brute_force_transducer",
    key_function=dict(prompter=repr),
)
def run_brute_force_transducer(
    sample_dfa_spec,
    num_states,
    num_symbols,
    num_sequence_symbols,
    num_repeats_per_dfa,
    num_dfas,
):
    results = run_transducer_experiment.function(
        "none",
        sample_dfa_spec,
        BasicInstructionTransducerPrompter(num_sequence_symbols, strip=True),
        num_repeats_per_dfa,
        num_dfas,
    )
    results = [
        brute_force_accuracy(num_states, num_symbols, rr.inputs, rr.outputs)
        for rr in tqdm.tqdm(results)
    ]
    return results
