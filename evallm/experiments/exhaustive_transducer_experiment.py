from collections import defaultdict
from dataclasses import dataclass
import string
from permacache import permacache
import numpy as np
import tqdm.auto as tqdm


from evallm.enumerate_dfa.enumerate import (
    all_io_permutations,
    enumerate_packed_dfas_no_permutations_valid_no_io_permutations,
)
from evallm.enumerate_dfa.pack_dfa import unpack_dfa
from evallm.experiments.transducer_experiment import TransducerExperimentResult
from evallm.prompting.transducer_prompt import BasicSequencePrompt


@dataclass
class TransducerExperimentResultPacked:
    """
    Represents the result of a transducer experiment with packed inputs and outputs.

    The inputs are packed as a 2D array of integers, where each row is a sequence of
    symbols. The outputs are packed as a 1D array of booleans, where each element is
    the output for the corresponding input sequence. The confusion matrix is a 2D array
    where the element (i, j) is the likelihood of the completion outputting j when the
    correct output is i.
    """

    inputs_packed: np.ndarray
    outputs_packed: np.ndarray
    confusion: np.ndarray

    @classmethod
    def of(cls, metas, prompts, results):
        ascii_to_int = {c: i for i, c in enumerate(string.ascii_lowercase)}
        inputs, outputs = zip(*metas)
        return cls(
            inputs_packed=np.array(
                [[ascii_to_int[c] for c in s] for s in inputs], dtype=np.uint8
            ),
            outputs_packed=np.array(outputs, dtype=bool),
            confusion=np.array(results),
        )


@permacache(
    "evallm/experiments/exhaustive_transducer_experiment/run_experiment_for_dfa_2",
    key_function=dict(prompter=repr),
)
def run_experiment_for_dfa(prompter, pdfa, count, model, sequence_seed):
    return TransducerExperimentResultPacked.of(
        *prompter.run_experiment(
            unpack_dfa(pdfa), np.random.RandomState(sequence_seed), model, count
        )
    )


def run_experiment_for_all_dfas(prompter, count, model, sequence_seed):
    pdfa_and_io = [
        (pdfa, pdfa_io)
        for pdfa in enumerate_packed_dfas_no_permutations_valid_no_io_permutations(3, 3)
        for pdfa_io in all_io_permutations(pdfa)
    ]
    result = defaultdict(list)
    for pdfa, pdfa_io in tqdm.tqdm(pdfa_and_io):
        result[pdfa].append(
            run_experiment_for_dfa(prompter, pdfa_io, count, model, sequence_seed)
        )
    return dict(result)


def main():
    num_sequence_symbols = 30
    model = "gpt-4o-mini-2024-07-18"

    prompter = BasicSequencePrompt.for_setting(
        dict(num_sequence_symbols=num_sequence_symbols)
    )
    run_experiment_for_all_dfas(prompter, count=100, model=model, sequence_seed=0)


if __name__ == "__main__":
    main()
