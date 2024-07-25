import itertools
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import tqdm.auto as tqdm
from permacache import permacache

from evallm.prompting.prompter import TrivialProblemError
from evallm.prompting.transducer_prompt import (
    BasicInstructionTransducerPrompter,
    ChainOfThoughtPrompt,
)
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

    @cached_property
    def success_rate_meets_kgram(self):
        if self.success_rate >= self.kgram_success_rates_each[-1]:
            return np.inf
        for k in range(len(self.kgram_success_rates_each) - 1, -1, -1):
            if self.success_rate >= self.kgram_success_rates_each[k]:
                return k + 1
        return 0

    @cached_property
    def success_rate_binary_meets_kgram(self):
        if self.success_rate_binary >= self.kgram_success_rates_each[-1]:
            return np.inf
        for k in range(len(self.kgram_success_rates_each) - 1, -1, -1):
            if self.success_rate_binary >= self.kgram_success_rates_each[k]:
                return k + 1
        return 0


def single_transducer_experiment(
    *, seed, model, num_repeats_per_dfa, sample_dfa_spec, prompter
):
    rng = np.random.RandomState(seed)
    dfa = sample_dfa(sample_dfa_spec, rng)
    return TransducerExperimentResult.of(
        *prompter.run_experiment(dfa, rng, model, num_repeats_per_dfa)
    )


@permacache(
    "evallm/experiments/transducer_experiment_9", key_function=dict(prompter=repr)
)
def run_transducer_experiment(
    model,
    sample_dfa_spec,
    prompter,
    num_repeats_per_dfa,
    num_dfas,
):
    results = []
    pbar = tqdm.tqdm(
        total=num_dfas,
        desc=f"Sampling: {sample_dfa_spec}, " f"Prompter: {prompter}",
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


def current_transducer_experiments():
    """
    Updated regularly to reflect the current experiments being run.
    """
    num_sequence_symbol_options = [
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        120,
        140,
        160,
        180,
        200,
        250,
        300,
        350,
        400,
        450,
        500,
    ]
    num_states_options = range(3, 1 + 8)
    results = {}
    for num_states in num_states_options:
        results[num_states] = {}
        for num_sequence_symbols in num_sequence_symbol_options:
            results[num_states][num_sequence_symbols] = run_transducer_experiment(
                "meta-llama/Meta-Llama-3-8B",
                sample_dfa_spec=dict(
                    type="sample_reachable_dfa", n_states=num_states, n_symbols=3
                ),
                prompter=BasicInstructionTransducerPrompter(num_sequence_symbols),
                num_repeats_per_dfa=30,
                num_dfas=100,
            )
    return results


def chatgpt_transducer_experiments():
    """
    Updated regularly to reflect the current experiments being run.
    """
    num_sequence_symbol_options = [
        30,
        # 40,
        # 50,
        # 60,
        # 70,
        # 80,
        # 90,
        # 100,
        # 120,
        # 140,
        # 160,
        # 180,
        # 200,
        # 250,
        # 300,
        # 350,
        # 400,
        # 450,
        # 500,
    ]
    # num_states_options = range(3, 1 + 8)
    num_states_options = [5]
    results = {}
    for num_states in num_states_options:
        results[num_states] = {}
        for num_sequence_symbols in num_sequence_symbol_options:
            results[num_states][num_sequence_symbols] = run_transducer_experiment(
                "gpt-3.5-turbo-0125",
                sample_dfa_spec=dict(
                    type="sample_reachable_dfa", n_states=num_states, n_symbols=3
                ),
                prompter=ChainOfThoughtPrompt(num_sequence_symbols),
                num_repeats_per_dfa=30,
                num_dfas=10,
            )
    return results
