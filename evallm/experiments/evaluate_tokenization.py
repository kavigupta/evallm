import json

from permacache import permacache

import numpy as np
import tqdm.auto as tqdm

import evallm

from evallm.experiments.sequence_completion.sequence_completion_experiments import (
    get_examples,
)
from evallm.experiments.sequence_completion.sequence_completion_prompt import (
    ANSWER_PATTERN,
)
from evallm.experiments.sequence_completion_summary import current_setting

prompt_prefix = (
    "How many 'a' characters appear in each line? Return your answers"
    " as a python list within <answer> tags, e.g., <answer>[1, 2, 3]</answer>"
)


def evaluate(res):
    for t in ANSWER_PATTERN.findall(res):
        try:
            return json.loads(t)
        # pylint: disable=broad-except
        except Exception:
            pass
    return None


def compute_counting_ability(eg_s, sep, model):
    prompt = "\n".join([prompt_prefix] + [""] + [sep.join(x) for x in eg_s])
    true_counts = compute_true_counts(eg_s)
    if evallm.llm.llm.model_specs[model].is_chat:
        prompt = {"system": "", "user": prompt}
    [res] = evallm.llm.run_prompt(
        model,
        [prompt],
        {"max_tokens": 5000, "temperature": 0.0},
    ).choices
    print(res)
    assert res.finish_reason == "stop"
    res = res.message.content
    res = evaluate(res)
    assert res is not None
    res = res[: len(true_counts)]
    res = res + [0] * (len(true_counts) - len(res))
    return (np.array(res) == np.array(true_counts)).mean()


def compute_true_counts(eg_s):
    return [x.count("a") for x in eg_s]


@permacache(
    "evallm/experiments/evaluate_tokenization/compute_random_baseline_counting_ability",
    shelf_type="individual-file",
    driver="pickle.gz",
)
def compute_random_baseline_counting_ability(count):
    res = []
    for seed in tqdm.trange(count):
        eg_s = get_eg_s(seed)
        tc = compute_true_counts(eg_s)
        random = (
            np.random.RandomState(seed).rand(len(eg_s), len(eg_s[0])) < 1 / 3
        ).sum(axis=1)
        res.append((np.array(random) == np.array(tc)).mean())
    return np.array(res)


def get_eg_s(seed):
    _, ([eg_s, _], *_) = get_examples(seed, current_setting)
    return eg_s


@permacache(
    "evallm/experiments/evaluate_tokenization/compute_counting_ability_many",
    shelf_type="individual-file",
    driver="pickle.gz",
)
def compute_counting_ability_many(count, sep, model):
    res = []
    for seed in tqdm.trange(count):
        res.append(compute_counting_ability(get_eg_s(seed), sep, model))
    return np.array(res)
