from types import SimpleNamespace
from typing import Counter
import numpy as np
from evallm.experiments.sequence_completion.sequence_completion_experiments import (
    collate_model_responses,
    get_examples as sequence_completion_get_examples,
)
from evallm.experiments.transducer_summary import (
    prompt_by_key as transducer_prompt_by_key,
    sample_dfa_spec as default_transducer_sample_dfa_spec,
    num_repeats_per_dfa as default_transducer_num_repeats_per_dfa,
)
from evallm.experiments.sequence_completion_summary import (
    current_setting as sequence_completion_default_setting,
    prompts_by_key as sequence_completion_prompts_by_key,
)
from evallm.prompting.prompter import TrivialProblemError
from evallm.sample_dfa.sample_dfa import sample_dfa


def transducer_prompt(prompt_key, model_is_chat):
    if prompt_key not in transducer_prompt_by_key:
        raise ValueError(
            f"Unknown prompt key: {prompt_key}. Available keys are: {list(transducer_prompt_by_key.keys())}"
        )
    is_chat_key = "chat" if model_is_chat else "non-chat"
    if is_chat_key not in transducer_prompt_by_key[prompt_key]:
        raise ValueError(
            f"Prompt key '{prompt_key}' does not have a {is_chat_key} version."
        )
    return transducer_prompt_by_key[prompt_key][is_chat_key]


def sequence_completion_prompt(prompt_key):
    return sequence_completion_prompts_by_key[prompt_key](
        sequence_completion_default_setting
    )


def transducer_metas_prompts_answers(
    *,
    seed,
    model_is_chat,
    num_repeats_per_dfa=default_transducer_num_repeats_per_dfa,
    # pylint: disable=dangerous-default-value
    sample_dfa_spec=default_transducer_sample_dfa_spec,
    prompt_key,
):
    rng = np.random.RandomState(seed)
    while True:
        dfa = sample_dfa(sample_dfa_spec, rng)
        try:
            return transducer_prompt(prompt_key, model_is_chat).metas_prompts_answers(
                dfa, rng, model_is_chat, num_repeats_per_dfa
            )
        except TrivialProblemError:
            pass


def transducer_evaluate_responses(
    *,
    responses,
    answers,
    model_is_chat,
    prompt_key,
):
    responses = [
        (
            SimpleNamespace(message=SimpleNamespace(content=response))
            if model_is_chat
            else SimpleNamespace(text=response)
        )
        for response in responses
    ]
    confusions = np.array(
        transducer_prompt(prompt_key, model_is_chat).score_all(answers, responses)
    )
    # only keep the diagonal, then sum across it.
    accs_each = confusions[..., np.eye(2, dtype=bool)].sum(axis=-1)
    counter = Counter(accs_each)
    # make it consistent with the sequence completion format
    return {k: float(counter[k]) for k in [0.0, 0.5, 1.0]}


def sequence_completion_dfa_prompts_answers(
    *,
    model_is_chat,
    seed,
    prompt_key,
):
    dfa, sequences_prefixes = sequence_completion_get_examples(
        seed, sequence_completion_default_setting
    )
    prompts = [
        sequence_completion_prompt(prompt_key).display_prompt(
            dfa, sequences, prefix, is_chat=model_is_chat
        )
        for sequences, prefix in sequences_prefixes
    ]
    return dfa, prompts, sequences_prefixes


def sequence_completion_evaluate_responses(
    *,
    responses,
    dfa,
    sequences_prefixes,
    prompt_key,
):
    return collate_model_responses(
        sequence_completion_prompt(prompt_key),
        dfa,
        sequences_prefixes,
        responses,
    )
