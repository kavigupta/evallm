from dataclasses import dataclass
from functools import cached_property
import itertools

import numpy as np
import tqdm.auto as tqdm

from evallm.prompting.prompter import TrivialProblemError
from evallm.prompting.transducer_prompt import BasicInstructionTransducerPrompter
from evallm.sample_dfa.naive_sample import naively_sample_dfa


@dataclass
class TransducerExperimentResult:
    inputs: list
    outputs: list
    prompts: list
    confusion: np.ndarray

    @classmethod
    def of(cls, metas, prompts, results):
        inputs, outputs = zip(*metas)
        return cls(inputs, outputs, prompts, np.mean(results, 0))

    @cached_property
    def null_success_rate(self):
        p = np.mean([out[-1] for out in self.outputs])
        return np.max([p, 1 - p])

    @cached_property
    def success_rate(self):
        return np.sum(np.diag(self.confusion))


def run_transducer_experiment(
    model,
    num_states,
    num_alphabet_symbols,
    num_sequence_symbols,
    num_repeats_per_dfa,
    num_dfas,
):
    results = []
    pbar = tqdm.tqdm(total=num_dfas)
    for seed in itertools.count():
        rng = np.random.RandomState(seed)
        dfa = naively_sample_dfa(num_states, num_alphabet_symbols, rng)
        prompter = BasicInstructionTransducerPrompter(num_sequence_symbols)
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
