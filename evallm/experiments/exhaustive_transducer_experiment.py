import os
import string
from dataclasses import dataclass

import numpy as np
from permacache import drop_if_equal, permacache

from evallm.enumerate_dfa.pack_dfa import unpack_dfa
from evallm.utils.kgrams import predict_based_on_kgram

from ..cachedir import cache_dir


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
    completions: np.ndarray | None

    @classmethod
    def of(cls, completions, metas, prompts, results, keep_completions=False):
        del prompts
        ascii_to_int = {c: i for i, c in enumerate(string.ascii_lowercase)}
        inputs, outputs = zip(*metas)
        return cls(
            inputs_packed=np.array(
                [[ascii_to_int[c] for c in s] for s in inputs], dtype=np.uint8
            ),
            outputs_packed=np.array(outputs, dtype=bool),
            confusion=np.array(results),
            completions=(completions if keep_completions else None),
        )

    @property
    def accuracy_each(self):
        return self.confusion[:, np.eye(2, dtype=bool)].sum(-1)


def compute_ngram_each(result):
    all_ngram_preds = [
        [
            x == o[-1]
            for x in predict_based_on_kgram([string.ascii_lowercase[c] for c in i], o)
        ]
        for i, o in zip(result.inputs_packed, result.outputs_packed)
    ]
    max_ngram = max(len(t) for t in all_ngram_preds)
    for x in all_ngram_preds:
        x += [x[-1]] * (max_ngram - len(x))
    ngram_each = np.array(all_ngram_preds)
    return ngram_each


@permacache(
    os.path.join(cache_dir, "run_experiment_for_dfa"),
    key_function=dict(prompter=repr, keep_completions=drop_if_equal(False)),
    shelf_type="individual-file",
)
def run_experiment_for_dfa(
    prompter, pdfa, count, model, sequence_seed, keep_completions=False
):
    return TransducerExperimentResultPacked.of(
        *prompter.run_experiment(
            unpack_dfa(pdfa), np.random.RandomState(sequence_seed), model, count
        ),
        keep_completions=keep_completions
    )
