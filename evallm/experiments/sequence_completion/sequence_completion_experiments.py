import os
from typing import Counter

import numpy as np
import tqdm.auto as tqdm
from permacache import permacache

from evallm.enumerate_dfa.pack_dfa import pack_dfa
from evallm.experiments.models_display import full_path
from evallm.experiments.sequence_completion.ngram_suffix_heuristic import (
    common_suffix_heuristic,
    multiple_ngrams,
)
from evallm.experiments.sequence_completion.sample_sequences import (
    sample_sequence_completion_problem,
)
from evallm.experiments.sequence_completion.sequence_completion_brute_force import (
    sequence_completion_brute_force,
)
from evallm.llm.llm import model_specs, run_prompt

from ...cachedir import cache_dir


@permacache(
    os.path.join(cache_dir, "compute_common_suffix_heuristic"),
    shelf_type="individual-file",
)
def compute_common_suffix_heuristic(num_seeds, setting):
    return np.array(
        [
            compute_ngram_score(seed, setting=setting, function=common_suffix_heuristic)
            for seed in tqdm.trange(num_seeds)
        ]
    )


def compute_ngram_score(seed, *, setting, function):
    dfa, examples = get_examples(seed, setting)
    return np.mean(
        [
            dfa.accepts_input(prefix + function(sequences, prefix))
            for sequences, prefix in examples
        ]
    )


@permacache(
    os.path.join(cache_dir, "compute_true_ngrams"),
    shelf_type="individual-file",
)
def compute_true_ngrams(ngram, num_seeds, setting):
    assert ngram >= 2
    return np.array(
        [
            compute_ngram_score(
                seed,
                setting=setting,
                function=lambda seq, pre: multiple_ngrams(seq, pre)[ngram - 2],
            )
            for seed in tqdm.trange(num_seeds)
        ]
    )


@permacache(
    os.path.join(cache_dir, "compute_brute_force_scores"),
    shelf_type="individual-file",
)
def compute_brute_force_scores(num_seeds, setting):
    return np.array(
        [
            compute_brute_force_score(seed, setting=setting)
            for seed in tqdm.trange(num_seeds)
        ]
    )


@permacache(
    "evallm/experiments/sequence_completion/sequence_completion_experiments/compute_brute_force_score"
)
def compute_brute_force_score(seed, setting):
    dfa, examples = get_examples(seed, setting)
    return np.mean(
        [
            dfa.accepts_input(
                prefix
                + sequence_completion_brute_force(pack_dfa(dfa), sequences, prefix)[0]
            )
            for sequences, prefix in tqdm.tqdm(examples, desc=f"{seed=}")
        ]
    )


@permacache(
    os.path.join(cache_dir, "compute_random_baseline_scores"),
    shelf_type="individual-file",
)
def compute_random_baseline_scores(num_seeds, setting):
    return np.array(
        [
            compute_random_baseline_score(seed, setting=setting)
            for seed in tqdm.trange(num_seeds)
        ]
    )


def compute_random_baseline_score(seed, *, setting):
    dfa, examples = get_examples(seed, setting)
    rng = np.random.RandomState(np.random.RandomState(seed).randint(2**32))
    return np.mean(
        [
            dfa.accepts_input(
                prefix
                + [
                    sorted(dfa.input_symbols)[sym]
                    for sym in rng.choice(
                        len(dfa.input_symbols), len(sequences[0]) - len(prefix)
                    )
                ]
            )
            for sequences, prefix in examples
        ]
    )


def compute_model_scores(num_seeds, setting, model, prompt_fn, *, na_mode="ignore"):
    prompt = prompt_fn(setting)
    res = compute_model_score_cached(num_seeds, setting, model, prompt)
    if na_mode == "ignore":
        if res[0.5].sum() > 0.25 * sum(res.values()).sum():
            return np.nan
        return res[1] / (res[0] + res[1])
    if na_mode == "zero":
        return res[1] / (res[0] + res[1] + res[0.5])
    if na_mode == "count-na":
        return res[0.5] / sum(res.values())
    raise ValueError(f"Invalid na_mode: {na_mode}")


@permacache(
    os.path.join(cache_dir, "compute_model_score_cached"),
    key_function=dict(prompt=lambda prompt: prompt.hash_prompt()),
    shelf_type="individual-file",
)
def compute_model_score_cached(num_seeds, setting, model, prompt):
    results = {0: [], 0.5: [], 1: []}
    for seed in tqdm.trange(num_seeds):
        scores = compute_model_score(
            seed, setting=setting, model=model, prompt=prompt
        )
        for k, v in scores.items():
            results[k].append(v)
    return {k: np.array(v) for k, v in results.items()}


@permacache(
    "evallm/experiments/sequence_completion/sequence_completion_experiments/compute_model_score_4",
    key_function=dict(prompt=lambda prompt: prompt.hash_prompt()),
)
def compute_model_score(seed, *, setting, model, prompt):
    """
    Returns counts of correct, incorrect, and N/A completions.
    """
    dfa, sequences_prefixes = get_examples(seed, setting)
    responses = run_model(model, prompt, dfa, sequences_prefixes)
    return collate_model_responses(prompt, dfa, sequences_prefixes, responses)


def score_model_responses(prompt, dfa, sequences_prefixes, responses):
    return [
        prompt.score_response(dfa, sequences, prefix, response)
        for (sequences, prefix), response in zip(sequences_prefixes, responses)
    ]


def collate_model_responses(prompt, dfa, sequences_prefixes, responses):
    results = Counter(score_model_responses(prompt, dfa, sequences_prefixes, responses))
    return {k: results[k] for k in [0.0, 0.5, 1.0]}


def run_model(model, prompt, dfa, sequences_prefixes):
    is_chat = model_specs[model].is_chat
    prompts = [
        prompt.display_prompt(dfa, sequences, prefix, is_chat=is_chat)
        for sequences, prefix in sequences_prefixes
    ]
    responses = run_prompt(model, prompts, prompt.model_kwargs())
    responses = responses.choices
    if is_chat:
        responses = [x.message.content for x in responses]
    else:
        responses = [x.text for x in responses]
    return responses


def get_examples(seed, setting):
    rng = np.random.RandomState(seed)
    return sample_sequence_completion_problem(rng, **setting, try_limit=50)
