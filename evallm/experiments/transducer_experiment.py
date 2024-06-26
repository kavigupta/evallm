import itertools
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import tqdm.auto as tqdm
from permacache import permacache

from evallm.prompting.prompter import TrivialProblemError
from evallm.sample_dfa.naive_sample import naively_sample_dfa
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
        res.kgram_success_rate
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
    def success_rate_each(self):
        return [np.sum(np.diag(confusion)) for confusion in self.confusion_each]

    @cached_property
    def kgram_successes_each(self):
        return [
            predict_based_on_kgram(inp, out) == out[-1]
            for inp, out in zip(self.inputs, self.outputs)
        ]

    @cached_property
    def kgram_success_rate(self):
        return np.mean(self.kgram_successes_each)


@permacache(
    "evallm/experiments/transducer_experiment_5", key_function=dict(prompter=repr)
)
def run_transducer_experiment(
    model,
    num_states,
    num_alphabet_symbols,
    prompter,
    num_repeats_per_dfa,
    num_dfas,
):
    results = []
    pbar = tqdm.tqdm(
        total=num_dfas,
        desc=f"Num states: {num_states}, "
        f"Num symbols: {num_alphabet_symbols}, "
        f"Prompter: {prompter}",
    )
    for seed in itertools.count():
        rng = np.random.RandomState(seed)
        dfa = naively_sample_dfa(num_states, num_alphabet_symbols, rng)
        try:
            result = TransducerExperimentResult.of(
                *prompter.run_experiment(dfa, rng, model, num_repeats_per_dfa)
            )
        except TrivialProblemError:
            continue
        results.append(result)
        pbar.update()
        if len(results) == num_dfas:
            pbar.close()
            break
    return results
