import string
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import tqdm.auto as tqdm
from permacache import (
    permacache,
    renamed_symbol_unpickler,
    stable_hash,
    swap_unpickler_context_manager,
)

from evallm.enumerate_dfa.enumerate import (
    all_io_permutations,
    enumerate_packed_dfas_no_permutations_valid_no_io_permutations,
)
from evallm.enumerate_dfa.pack_dfa import unpack_dfa
from evallm.prompting.prompter import TrivialProblemError
from evallm.prompting.transducer_prompt import (
    BasicSequencePrompt,
    BasicSequencePromptNoChat,
    RedGreenRoomPrompt1,
)
from evallm.utils.kgrams import predict_based_on_kgram


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
        del prompts
        ascii_to_int = {c: i for i, c in enumerate(string.ascii_lowercase)}
        inputs, outputs = zip(*metas)
        return cls(
            inputs_packed=np.array(
                [[ascii_to_int[c] for c in s] for s in inputs], dtype=np.uint8
            ),
            outputs_packed=np.array(outputs, dtype=bool),
            confusion=np.array(results),
        )


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
    ngram_each = np.array(all_ngram_preds).mean(0)
    return ngram_each


@dataclass
class SummaryStats:
    model_summary: Counter
    ngram_each: np.ndarray

    @classmethod
    def of(cls, result):
        diags = result.confusion[:, np.eye(2, dtype=bool)].sum(-1)
        return cls(model_summary=Counter(diags), ngram_each=compute_ngram_each(result))


@permacache(
    "evallm/experiments/exhaustive_transducer_experiment/run_experiment_for_dfa_2",
    key_function=dict(prompter=repr),
    read_from_shelf_context_manager=swap_unpickler_context_manager(
        renamed_symbol_unpickler(
            {
                (
                    "__main__",
                    "TransducerExperimentResultPacked",
                ): TransducerExperimentResultPacked
            }
        )
    ),
    multiprocess_safe=True,
)
def run_experiment_for_dfa(prompter, pdfa, count, model, sequence_seed):
    return TransducerExperimentResultPacked.of(
        *prompter.run_experiment(
            unpack_dfa(pdfa), np.random.RandomState(sequence_seed), model, count
        )
    )


@permacache(
    "evallm/experiments/exhaustive_transducer_experiment/summary_experiment_for_dfa_3",
    key_function=dict(prompt=repr),
    read_from_shelf_context_manager=swap_unpickler_context_manager(
        renamed_symbol_unpickler({("__main__", "SummaryStats"): SummaryStats})
    ),
)
def summary_experiment_for_dfa(prompt, pdfa, count, model, sequence_seed):
    result = run_experiment_for_dfa(
        prompt, pdfa, count, model, sequence_seed=sequence_seed
    )
    return SummaryStats.of(result)


@permacache(
    "evallm/experiments/exhaustive_transducer_experiment/summary_experiment_for_dfas_3",
    key_function=dict(prompt=repr, pdfas=stable_hash),
    read_from_shelf_context_manager=swap_unpickler_context_manager(
        renamed_symbol_unpickler({("__main__", "SummaryStats"): SummaryStats})
    ),
)
def summary_experiment_for_dfas(prompt, pdfas, count, model, sequence_seed):
    result = {}
    for pdfa in tqdm.tqdm(pdfas):
        try:
            r = summary_experiment_for_dfa(prompt, pdfa, count, model, sequence_seed)
        except TrivialProblemError:
            r = None
        result[pdfa] = r
    return dict(result)


def run_experiment_for_all_dfas(prompt, count, model, sequence_seed, limit=None):
    pdfa_and_io = [
        (pdfa, pdfa_io)
        for pdfa in enumerate_packed_dfas_no_permutations_valid_no_io_permutations(3, 3)
        for pdfa_io in all_io_permutations(pdfa)[:limit]
    ]
    result_flat = summary_experiment_for_dfas(
        prompt,
        sorted({pdfa_io for _, pdfa_io in pdfa_and_io}),
        count,
        model,
        sequence_seed,
    )
    result = defaultdict(list)
    for pdfa, pdfa_io in pdfa_and_io:
        result[pdfa].append(result_flat[pdfa_io])
    return dict(result)


def exhaustive_gpt_4o_mini(limit):
    num_sequence_symbols = 30
    model = "gpt-4o-mini-2024-07-18"

    prompter = BasicSequencePrompt.for_setting(
        dict(num_sequence_symbols=num_sequence_symbols)
    )
    return run_experiment_for_all_dfas(
        prompter, count=100, model=model, sequence_seed=0, limit=limit
    )


def exhaustive_gpt_4o_mini_red_green(limit):
    num_sequence_symbols = 30
    model = "gpt-4o-mini-2024-07-18"

    prompter = RedGreenRoomPrompt1.for_setting(
        dict(
            num_sequence_symbols=num_sequence_symbols,
            num_states=3,
            sample_dfa_spec=dict(type="sample_reachable_dfa", n_states=3, n_symbols=3),
        )
    )
    return run_experiment_for_all_dfas(
        prompter, count=100, model=model, sequence_seed=0, limit=limit
    )


def exhaustive_llama(limit):
    num_sequence_symbols = 30
    model = "meta-llama/Meta-Llama-3-8B"

    prompter = BasicSequencePromptNoChat.for_setting(
        dict(num_sequence_symbols=num_sequence_symbols)
    )
    return run_experiment_for_all_dfas(
        prompter, count=100, model=model, sequence_seed=0, limit=limit
    )


if __name__ == "__main__":
    # exhaustive_llama(1)
    # exhaustive_llama(None)
    # exhaustive_gpt_4o_mini(1)
    # exhaustive_gpt_4o_mini(None)
    exhaustive_gpt_4o_mini_red_green(1)
