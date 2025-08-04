import os
import re

import numpy as np
import tqdm.auto as tqdm
from permacache import permacache

import evallm

from ..cachedir import cache_dir

prompt_template = (
    "I will give you a string. Tell me whether it matches the following regular"
    " expression: '{regexp}' (without quotes). Just answer YES or NO."
)


def evaluate_model_regexp_matching(model, regexp, test_str):
    prompt = prompt_template.format(regexp=regexp) + "\n" + test_str
    is_chat = evallm.llm.llm.model_specs[model].is_chat

    if is_chat:
        prompt = {"system": "", "user": prompt}
    [response] = evallm.llm.run_prompt(
        model,
        [prompt],
        {"max_tokens": 10, "temperature": 0.0},
    ).choices
    if hasattr(response, "finish_reason"):
        assert response.finish_reason == "stop"
    if is_chat:
        response = response.message.content
    else:
        response = response.text
    is_yes = "YES" in response
    is_no = "NO" in response
    assert is_yes != is_no
    return is_yes


regexp_for_demo = r"^ab(abc)+$"


def sample_string_for_demo(seed):
    """
    Note: this sampling procedure depends on the regexp, the results are technically correct
    no matter what but if the regexp changes, we don't have ~50% positive examples.
    """
    rng = np.random.RandomState(seed)
    repeats = rng.choice(4) + 1
    s = "ab" + "abc" * repeats
    if rng.rand() < 0.5:
        s = list(s)
        idx = rng.choice(len(s))
        s[idx] = rng.choice(sorted(set("abc") - {s[idx]}))
        s = "".join(s)
    return s


@permacache(
    os.path.join(
        cache_dir,
        "evaluate_tokenization_regexp_matching/evaluate_model_regexp_matching_multiple",
    ),
    shelf_type="individual-file",
    driver="pickle.gz",
)
def evaluate_model_regexp_matching_multiple(model, count):
    results_true = []
    results_pred = []
    for i in tqdm.trange(count):
        st = sample_string_for_demo(i)
        results_pred.append(evaluate_model_regexp_matching(model, regexp_for_demo, st))
        results_true.append(bool(re.match(regexp_for_demo, st)))
    results_pred = np.array(results_pred)
    results_true = np.array(results_true)
    return results_pred, results_true


def summary(results_pred, results_true):
    print(f"Percent positive: {results_true.mean():.0%}")
    print(f"Percent correct : {(results_true == results_pred).mean():.0%}")
